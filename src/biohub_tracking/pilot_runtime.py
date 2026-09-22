"""User-triggered child-process supervision, bounded logs and verifiable receipts."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, signal, subprocess, sys, time, uuid
from importlib import metadata
from .io import atomic_json, atomic_text, project_root, sha256_file


def utc(): return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def identifier(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def emit(root,stage,state,**details):
    line=json.dumps({'utc':utc(),'stage':stage,'state':state,**details},allow_nan=False)
    path=Path(root)/'outputs/pilot/progress.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)

def versions():
    return {name:metadata.version(name) for name in ('numpy','pandas','scipy','zarr','plotly','kaggle')}

def code_signature(root):
    files=sorted((Path(root)/'src/biohub_tracking').glob('pilot*.py'))
    return canonical_hash({p.name:sha256_file(p) for p in files})

def verified_file(root,relative,expected_hash,max_bytes=256*1024**2):
    base=Path(root).resolve();path=base/relative
    if not path.is_file() or path.is_symlink() or base not in path.resolve().parents:
        raise ValueError(f'Missing/unsafe artifact: {relative}')
    if path.stat().st_size>max_bytes or sha256_file(path)!=expected_hash:
        raise ValueError(f'Artifact size/checksum mismatch: {relative}')
    return path

def terminate_group(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=5)


def public_failure_detail(root, stage):
    """Return a limited, redacted project failure message; never publish raw stderr."""
    path = Path(root) / 'outputs/pilot' / (stage + '_failure.json')
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 16384:
            return "No small structured failure record; inspect private stderr locally."
        record = json.loads(path.read_text(encoding='utf-8'))
        kind = record.get('error_type')
        if kind not in {'ValueError', 'RuntimeError', 'TimeoutError', 'MemoryError',
                        'KeyError', 'FileNotFoundError', 'TypeError'}:
            return "Worker failed; private details retained locally."
        message = str(record.get('message', ''))[:800]
        # Do not display unknown third-party error text or authentication output.
        prefixes = ('STOP:', 'No candidates', 'No adjacent-frame', 'Flat ROI:',
                    'Invalid ', 'Unsupported ', 'Missing/unsafe artifact:',
                    'Artifact size/checksum', 'Decoded array', 'ROI cache',
                    'Kaggle file request exited', 'Configuration/metadata changed',
                    'Incompatible download receipt', 'Sample ', 'Metadata ')
        if not message.startswith(prefixes):
            return kind + ': details retained locally; no automatic retry.'
        if re.search(r'(?i)(token|secret|password|authorization|credential|api[_ -]?key)', message):
            return kind + ': sensitive detail omitted; inspect locally.'
        message = re.sub(r'https?://[^\s]+', '[URL omitted]', message)
        message = re.sub(r'[A-Za-z0-9_+/=-]{40,}', '[long value omitted]', message)
        message = ' '.join(message.split())
        return kind + ': ' + message[:500]
    except (OSError, ValueError, TypeError):
        return 'Failure detail unavailable; inspect private stderr locally.'


def run_bounded(stage,root=None):
    """Run one allowed stage. An interrupted parent always terminates its process group."""
    import psutil
    root=Path(root or project_root()).resolve()
    cfg=json.loads((root/'configs/pilot.json').read_text())
    allowed={'metadata':120,'sample':240,'features':180}
    if stage not in allowed:raise ValueError('Unknown bounded stage.')
    seconds=cfg['stage_timeouts_seconds'][stage]
    if not isinstance(seconds,int) or not 1<=seconds<=allowed[stage]:raise ValueError('Expanded deadline refused.')
    folder=root/'outputs/pilot/processes'/identifier();folder.mkdir(parents=True)
    stdout=folder/'stdout.txt';stderr=folder/'stderr_private.txt'
    env=os.environ.copy()
    env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
    start=time.monotonic();last=start;position=0;proc=None
    try:
        with stdout.open('w') as out,stderr.open('w') as err:
            proc=subprocess.Popen([sys.executable,'-u','-m','biohub_tracking.pilot','--stage',stage],cwd=root,
                    env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            while proc.poll() is None:
                now=time.monotonic()
                if now-start>seconds:raise TimeoutError(f'{stage}: exceeded {seconds}s; stop and return logs.')
                if stdout.stat().st_size+stderr.stat().st_size>4*1024**2:raise RuntimeError('Stage log cap exceeded.')
                try:
                    processes=[psutil.Process(proc.pid),*psutil.Process(proc.pid).children(recursive=True)]
                    rss=sum(p.memory_info().rss for p in processes if p.is_running())
                except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
                if rss>cfg['max_rss_bytes']:raise MemoryError('Stage process tree exceeded 8 GiB RSS guard.')
                with stdout.open() as read:
                    read.seek(position);new=read.read();position=read.tell()
                if new:print(new,end='',flush=True)
                if now-last>=10:
                    print(json.dumps({'utc':utc(),'stage':stage,'state':'supervisor_waiting',
                          'elapsed_seconds':round(now-start,1),'process_tree_rss_MiB':round(rss/1024**2,1),
                          'note':'Only completed-stage/chunk events represent progress.'}),flush=True)
                    last=now
                time.sleep(0.25)
        with stdout.open() as read:read.seek(position);print(read.read(),end='',flush=True)
        if proc.returncode!=0:
            detail = public_failure_detail(root, stage)
            print('WORKER_FAILURE_DETAIL: ' + detail, flush=True)
            raise RuntimeError(f'STOP: {stage} exited {proc.returncode}. {detail} '
                               f'Private log: {stderr.relative_to(root)}. Do not rerun unchanged.')
        return json.loads((root/f'outputs/pilot/{stage}_receipt.json').read_text())
    except BaseException as exc:
        if proc is not None:terminate_group(proc)
        atomic_json(folder/'supervisor_failure.json',{'utc':utc(),'stage':stage,'error_type':type(exc).__name__,
                     'elapsed_seconds':round(time.monotonic()-start,3),'child_group_termination_requested':proc is not None})
        raise
    finally:
        if proc is not None and proc.poll() is None:terminate_group(proc)
