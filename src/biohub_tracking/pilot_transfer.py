"""Small, explicit Kaggle file requests. No full download, submission, retries or credentials in public logs."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import hashlib, json, os, resource, shutil, subprocess, sys, tempfile, time, zipfile
from .io import atomic_json, atomic_text, sha256_file
from .pilot_contract import checked_relative
from .pilot_runtime import emit, utc, verified_file


def unpack_single(download: Path, remote_key: str, max_bytes: int) -> bytes:
    """Accept the CLI's one raw file or its single-member ZIP; never extract paths."""
    if download.stat().st_size>max_bytes:raise ValueError('Downloaded file exceeds its cap.')
    if not zipfile.is_zipfile(download):return download.read_bytes()
    with zipfile.ZipFile(download) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        if len(members)!=1:raise ValueError('Expected one file, not an archive of the dataset.')
        item=members[0];name=checked_relative(item.filename)
        allowed={checked_relative(remote_key),PurePosixPath(remote_key).name}
        if name not in allowed or item.file_size>max_bytes or (item.external_attr>>16)&0o170000==0o120000:
            raise ValueError('Unexpected, oversized or unsafe ZIP member.')
        with z.open(item) as f:
            data=f.read(max_bytes+1)
        if len(data)!=item.file_size or len(data)>max_bytes:raise ValueError('ZIP member size mismatch.')
        return data


def fetch_file(root: Path, cfg: dict, remote_key: str, metadata_only=False) -> Path:
    remote_key=checked_relative(remote_key)
    prefix=f'train/{cfg["sample_id"]}.zarr/'
    if not remote_key.startswith(prefix):raise ValueError('Remote key is outside this single training crop.')
    dest=root/'data/pilot_snapshot'/remote_key
    receipt_path=root/'outputs/pilot/downloads'/(hashlib.sha256(remote_key.encode()).hexdigest()+'.json')
    max_bytes=2*1024**2 if metadata_only else cfg['max_download_file_bytes']
    if receipt_path.is_file():
        old=json.loads(receipt_path.read_text())
        if old.get('status')!='completed' or old.get('remote_key')!=remote_key or old.get('competition')!=cfg['competition']:
            raise ValueError('Incompatible download receipt.')
        p=verified_file(root,old['relative_path'],old['sha256'],max_bytes)
        if p.resolve()!=dest.resolve():raise ValueError('Download receipt points at a different file.')
        emit(root,'download','reused_verified_file',key=remote_key,bytes=p.stat().st_size,snapshot_utc=old['utc'])
        return p
    if dest.exists():raise ValueError('A file exists without a completion receipt; preserve it and inspect, do not overwrite.')
    cache=root/'data/pilot_snapshot';cache.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in cache.rglob('*')):raise ValueError('Symlinks are not allowed in the partial snapshot.')
    existing=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if existing>=cfg['max_download_cache_bytes']:raise ValueError('Private sample cache reached 256 MiB.')
    if shutil.disk_usage(root).free<2*1024**3:raise ValueError('At least 2 GiB free disk required for this pilot.')
    cli=Path(sys.executable).parent/'kaggle'
    if not cli.is_file():raise RuntimeError('Use the existing Biohub .venv kernel/terminal.')
    private=root/'outputs/pilot/private';private.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();proc=None
    with tempfile.TemporaryDirectory(prefix='download_',dir=private) as temp:
        folder=Path(temp);downloads=folder/'payload';downloads.mkdir()
        stdout=folder/'stdout_private.txt';stderr=folder/'stderr_private.txt'
        def file_cap():resource.setrlimit(resource.RLIMIT_FSIZE,(max_bytes,max_bytes))
        try:
            with stdout.open('w') as out,stderr.open('w') as err:
                proc=subprocess.Popen([str(cli),'competitions','download',cfg['competition'],
                    '-f',remote_key,'-p',str(downloads)],cwd=root,stdin=subprocess.DEVNULL,
                    stdout=out,stderr=err,preexec_fn=file_cap)
                last=start
                while proc.poll() is None:
                    now=time.monotonic()
                    if now-start>cfg['per_request_timeout_seconds']:raise TimeoutError('Single-file Kaggle request exceeded 45 seconds.')
                    observed=sum(p.stat().st_size for p in downloads.rglob('*') if p.is_file())
                    if observed>max_bytes or existing+observed>cfg['max_download_cache_bytes']:
                        raise ValueError('Download byte cap exceeded.')
                    if now-last>=10:
                        emit(root,'download','waiting',key=remote_key,elapsed_seconds=round(now-start,1),bytes_observed=observed)
                        last=now
                    time.sleep(0.2)
            if proc.returncode!=0:raise RuntimeError(f'Kaggle file request exited {proc.returncode}; verify entry/authentication/key before retry.')
            files=[p for p in downloads.rglob('*') if p.is_file()]
            if len(files)!=1 or any(p.is_symlink() for p in downloads.rglob('*')):
                raise ValueError('Expected exactly one raw file or one ZIP from this single-file request.')
            payload=unpack_single(files[0],remote_key,max_bytes)
            if not payload or existing+len(payload)>cfg['max_download_cache_bytes']:raise ValueError('Empty file or cache cap exceeded.')
            if metadata_only:
                meta=json.loads(payload.decode('utf-8'))
                if not isinstance(meta,dict):raise ValueError('Metadata must contain an object.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            tempdest=dest.with_name(dest.name+'.partial')
            with tempdest.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
            tempdest.replace(dest)
            receipt={'utc':utc(),'status':'completed','competition':cfg['competition'],'remote_key':remote_key,
                 'relative_path':dest.relative_to(root).as_posix(),'bytes':len(payload),'sha256':sha256_file(dest),
                 'remote_dataset_version':'not_exposed_by_this_capture; this is a timestamped local snapshot',
                 'elapsed_seconds':round(time.monotonic()-start,3)}
            atomic_json(receipt_path,receipt)
            emit(root,'download','completed_verified_file',key=remote_key,bytes=len(payload),sha256=receipt['sha256'])
            return dest
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
            tag=hashlib.sha256(remote_key.encode()).hexdigest()[:12]
            for p in (stdout,stderr):
                if p.exists():shutil.copyfile(p,private/(tag+'_'+p.name))
            raise
