"""Local-only provenance, input contracts and atomic writes for two feature rounds."""
from __future__ import annotations
import hashlib,json,os,re,sys
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError


def utc():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda:handle.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        handle.write(data if isinstance(data,bytes) else data.encode('utf-8'));handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def write_json(path,value):atomic(path,json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n')


def safe(root,relative,max_bytes=32*1024**2):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe relative artifact path.')
    p=root/relative
    if not p.is_file() or p.is_symlink() or root not in p.resolve().parents:raise ValueError(f'Missing or unsafe artifact: {relative}')
    if p.stat().st_size>max_bytes:raise ValueError(f'Artifact exceeds limit: {relative}')
    return p


def verify(root,relative,expected,max_bytes=32*1024**2):
    p=safe(root,relative,max_bytes)
    if digest(p)!=expected:raise ValueError(f'Artifact hash mismatch: {relative}')
    return p


def artifacts(root,paths):return {Path(p).relative_to(root).as_posix():digest(p) for p in paths}


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_signature(root):
    root=Path(root);paths=[*sorted((root/'src/biohub_tracking').glob('feature_*.py')),
        *sorted((root/'tests').glob('test_feature_*.py')),
        root/'configs/feature_rounds.json',root/'configs/feature_catalog.json',
        root/'scripts/run_feature_tests.py',root/'scripts/run_feature_rounds.py',root/'tests/feature_fixtures.py']
    return fingerprint(artifacts(root,paths))


def versions():
    result={'python':sys.version.split()[0]}
    for package in ['numpy','pandas','scipy','plotly','IPython','psutil','pytest']:
        try:result[package]=version(package)
        except PackageNotFoundError:result[package]='missing'
    return result


def check_tests(root):
    r=read_json(safe(root,'outputs/features/tests_receipt.json'))
    if r.get('status')!='passed' or r.get('source_signature')!=source_signature(root):
        raise ValueError('Run scripts/run_feature_tests.py for this exact source first.')
    if any(r['counts'].get(k,0) for k in ['failures','errors','skipped']) or r['counts'].get('tests',0)==0:
        raise ValueError('Feature tests did not all pass.')
    verify(root,r['report'],r['report_sha256'])
    return r


def load_inputs(root):
    """Check existing receipts without calling any of the earlier pipeline stages."""
    import numpy as np
    import pandas as pd
    root=Path(root).resolve();cfg=read_json(root/'configs/feature_rounds.json')
    sr=read_json(safe(root,'outputs/pilot/sample_receipt.json'))
    fr=read_json(safe(root,'outputs/pilot/features_receipt.json'))
    if sr.get('status')!='completed' or fr.get('status')!='completed':raise ValueError('Completed sample and pilot feature receipts required.')
    paths=['data/pilot_snapshot/selected_roi.npz','outputs/pilot/node_features.csv','outputs/pilot/pair_features.csv']
    expected={**sr['artifacts'],**fr['artifacts']}
    for p in paths:verify(root,p,expected[p])
    with np.load(root/paths[0],allow_pickle=False) as a:
        images,frames,offsets,spacing=[a[k] for k in ['images','frames','offsets','spacing_um']]
    if list(images.shape)!=cfg['required_roi_shape'] or frames.tolist()!=cfg['frames']:
        raise ValueError('Unexpected ROI shape or frame selection; do not guess offsets.')
    if not np.isfinite(images).all() or spacing.shape!=(3,) or not np.isfinite(spacing).all() or np.any(spacing<=0):raise ValueError('Nonfinite image or invalid spacing.')
    if not np.allclose(spacing,[1.625,.40625,.40625],atol=1e-12):raise ValueError('Spacing differs from this reviewed sample.')
    if sr.get('sample_id')!=cfg['sample_id'] or list(offsets)!=sr['roi_offsets_zyx']:raise ValueError('Sample/offset provenance mismatch.')
    nodes=pd.read_csv(root/paths[1]);pairs=pd.read_csv(root/paths[2]);validate_tables(nodes,pairs,images.shape,offsets,spacing,frames,cfg)
    provenance={'files':{p:expected[p] for p in paths},'sample_receipt':digest(root/'outputs/pilot/sample_receipt.json'),
                'features_receipt':digest(root/'outputs/pilot/features_receipt.json'),'source':source_signature(root),'versions':versions()}
    return cfg,images,frames,offsets,spacing,nodes,pairs,provenance


def validate_tables(nodes,pairs,shape,offsets,spacing,frames,cfg):
    import numpy as np
    if len(nodes)==0 or len(nodes)>cfg['round1']['max_nodes'] or len(pairs)==0 or len(pairs)>cfg['round2']['max_pairs']:raise ValueError('Candidate row count outside bounded pilot.')
    required_n=['node_id','t','z','y','x','z_um','y_um','x_um','core_mean_raw','robust_contrast','best_sigma_um','roi_boundary_distance_um','weighted_shape_ratio','log_response']
    required_p=['source_id','target_id','t_source','t_target','distance_um']
    if any(c not in nodes for c in required_n) or any(c not in pairs for c in required_p):raise ValueError('Missing pilot feature columns.')
    if nodes.node_id.duplicated().any() or pairs.duplicated(['source_id','target_id']).any():raise ValueError('Duplicate node or pair identifiers.')
    if not np.isfinite(nodes[required_n].to_numpy(float)).all() or not np.isfinite(pairs[required_p].to_numpy(float)).all():raise ValueError('Nonfinite required candidate values.')
    if not set(nodes.t)==set(frames):raise ValueError('Every selected frame must have candidates.')
    coords=nodes[['z','y','x']].to_numpy(float);local=coords-offsets
    if np.any(coords!=np.rint(coords)) or np.any(local<0) or np.any(local>=np.asarray(shape)[1:]):raise ValueError('Candidate coordinate outside retained ROI.')
    if not np.allclose(coords*spacing,nodes[['z_um','y_um','x_um']],atol=1e-8):raise ValueError('Voxel-to-physical coordinate mismatch.')
    index=nodes.set_index('node_id')
    if not set(pairs.source_id).issubset(index.index) or not set(pairs.target_id).issubset(index.index):raise ValueError('Unknown node in pair table.')
    for _,p in pairs.iterrows():
        if index.loc[p.source_id,'t']!=p.t_source or index.loc[p.target_id,'t']!=p.t_target or p.t_target!=p.t_source+1:raise ValueError('Misaligned temporal pair.')
        d=np.linalg.norm(index.loc[p.target_id,['z_um','y_um','x_um']].to_numpy(float)-index.loc[p.source_id,['z_um','y_um','x_um']].to_numpy(float))
        if not np.isclose(d,p.distance_um,atol=1e-7):raise ValueError('Recorded candidate distance mismatch.')


def emit(root,stage,event,**fields):
    record={'utc':utc(),'stage':stage,'event':event,**fields}
    line=json.dumps(record,allow_nan=False,sort_keys=True)
    p=Path(root)/'outputs/features/progress.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(line+'\n')
    print(line,flush=True)


def checkpoint_valid(root,path,signature):
    if not Path(path).exists():return None
    r=read_json(path)
    if r.get('signature')!=signature or r.get('status')!='completed':return None
    for name,sha in r['artifacts'].items():verify(root,name,sha)
    return r


def clean_message(exc):
    message=f'{type(exc).__name__}: {str(exc)[:1000]}'
    message=re.sub(r'https?://\S+','[URL]',message)
    message=re.sub(r'(?i)(token|secret|password|authorization|code_challenge)\s*[=:]\s*\S+',r'\1=[REDACTED]',message)
    message=re.sub(r'[A-Za-z0-9_+/=-]{90,}','[LONG_VALUE_REDACTED]',message)
    return message
