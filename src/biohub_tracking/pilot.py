"""Bounded CLI stages for the first REAL-image representation pilot; zero model fits."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from .io import atomic_json, atomic_text, project_root, sha256_file
from .pilot_contract import validate_config, image_geometry, chunk_plan, centered_roi
from .pilot_runtime import emit, utc, canonical_hash, code_signature, verified_file, versions
from .pilot_transfer import fetch_file


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def records(root,paths):return {p.relative_to(root).as_posix():sha256_file(p) for p in paths}

def completed(root,stage,signature):
    path=root/f'outputs/pilot/{stage}_receipt.json'
    if not path.exists():return None
    old=read_json(path)
    if old.get('status')!='completed' or old.get('signature')!=signature:return None
    for relative,digest in old['artifacts'].items():verified_file(root,relative,digest)
    emit(root,stage,'reused_verified_stage',snapshot_utc=old['utc'])
    return old


def receipt(root,stage,signature,artifacts,**details):
    value={'utc':utc(),'stage':stage,'status':'completed','signature':signature,
        'training_fits':0,'official_score':None,'leaderboard_score':None,
        'artifacts':records(root,artifacts),**details}
    atomic_json(root/f'outputs/pilot/{stage}_receipt.json',value)
    emit(root,stage,'completed',**{k:v for k,v in details.items() if k not in ('artifact_checksums',)})
    return value


def metadata_stage(root,cfg):
    prefix=f'train/{cfg["sample_id"]}.zarr/'
    paths=[fetch_file(root,cfg,prefix+'zarr.json',metadata_only=True),
           fetch_file(root,cfg,prefix+'0/zarr.json',metadata_only=True)]
    raw=[read_json(p) for p in paths]
    geometry=image_geometry(*raw,cfg);plan=chunk_plan(raw[1],cfg)
    signature=canonical_hash({'config':cfg,'metadata':records(root,paths),
        'contract_code':sha256_file(Path(__file__).with_name('pilot_contract.py'))})
    old=completed(root,'metadata',signature)
    if old:return old
    plan.update(geometry,sample_id=cfg['sample_id'],embryo_id=cfg['sample_id'].split('_')[0],
        split='train',scope='four-frame image-only development pilot; NOT validation',
        crop_selection_source=cfg['sample_source'],code_signature=code_signature(root))
    target=root/'outputs/pilot/sample_plan.json';atomic_json(target,plan)
    return receipt(root,'metadata',signature,[*paths,target],chunk_files_planned=plan['chunk_file_count'],
        selected_decoded_MiB=round(plan['selected_decoded_bytes']/1024**2,2),
        spacing_source=plan['spacing_source'],full_dataset_downloaded=False)


def metadata_guard(root,cfg):
    r=read_json(root/'outputs/pilot/metadata_receipt.json')
    if r.get('status')!='completed':raise ValueError('Metadata gate is incomplete.')
    for relative,digest in r['artifacts'].items():verified_file(root,relative,digest)
    p=read_json(root/'outputs/pilot/sample_plan.json')
    prefix=f'train/{cfg["sample_id"]}.zarr/'
    rawpaths=[root/'data/pilot_snapshot'/prefix/'zarr.json',root/'data/pilot_snapshot'/prefix/'0/zarr.json']
    expected=canonical_hash({'config':cfg,'metadata':records(root,rawpaths),
        'contract_code':sha256_file(Path(__file__).with_name('pilot_contract.py'))})
    if r['signature']!=expected:raise ValueError('Configuration/metadata changed; rerun only the metadata gate, not training.')
    return p,r


def sample_stage(root,cfg):
    import zarr
    plan,meta=metadata_guard(root,cfg);prefix=f'train/{cfg["sample_id"]}.zarr/'
    chunks=[]
    for i,key in enumerate(plan['chunk_keys']):
        chunks.append(fetch_file(root,cfg,prefix+key))
        emit(root,'sample','chunk_ready',completed_chunks=i+1,total_chunks=len(plan['chunk_keys']))
    # This must precede any Zarr indexing: missing chunks can silently become fill values.
    for p in chunks:
        remote=p.relative_to(root/'data/pilot_snapshot').as_posix()
        r=read_json(root/'outputs/pilot/downloads'/(hashlib.sha256(remote.encode()).hexdigest()+'.json'))
        verified_file(root,r['relative_path'],r['sha256'],cfg['max_download_file_bytes'])
    signature=canonical_hash({'metadata_signature':meta['signature'],'chunks':records(root,chunks),
                             'zarr_version':zarr.__version__,'roi':cfg['roi_shape_zyx']})
    old=completed(root,'sample',signature)
    if old:return old
    store=root/'data/pilot_snapshot'/prefix
    array=zarr.open_array(str(store/'0'),mode='r')
    if list(array.shape)!=plan['shape_tzyx'] or np.dtype(array.dtype)!=np.dtype(plan['dtype']):
        raise ValueError('Decoded array metadata does not match the validated plan.')
    slices,offsets=centered_roi(array.shape[1:],cfg['roi_shape_zyx'])
    rois=[];qc=[]
    for t in cfg['frames']:
        frame=np.asarray(array[t,:,:,:])
        if list(frame.shape)!=plan['shape_tzyx'][1:] or not np.isfinite(frame).all():
            raise ValueError('Bad decoded shape/nonfinite values; do not interpret as biological data.')
        roi=np.array(frame[slices],copy=True)
        if roi.max()<=roi.min():raise ValueError('Flat ROI: stop and inspect acquisition/selection, do not expand the run.')
        rois.append(roi)
        values=np.quantile(roi,[.01,.5,.998])
        qc.append({'t':t,'roi_min':float(roi.min()),'roi_max':float(roi.max()),
            'roi_q01':float(values[0]),'roi_median':float(values[1]),'roi_q998':float(values[2]),
            'roi_mean':float(roi.mean()),'roi_std':float(roi.std()),
            'roi_saturation_fraction':float(np.mean(roi==np.iinfo(roi.dtype).max)) if np.issubdtype(roi.dtype,np.integer) else None,
            'roi_sha256':hashlib.sha256(roi.tobytes()).hexdigest()})
        emit(root,'sample','decoded_frame',frame=t,completed_frames=len(rois),total_frames=len(cfg['frames']))
    target=root/'data/pilot_snapshot/selected_roi.npz';temporary=target.with_suffix('.npz.partial')
    with temporary.open('wb') as f:np.savez_compressed(f,images=np.stack(rois),frames=np.asarray(cfg['frames']),offsets=np.asarray(offsets),spacing_um=np.asarray(plan['spacing_um']))
    temporary.replace(target)
    table=root/'outputs/pilot/image_quality.csv';atomic_text(table,pd.DataFrame(qc).to_csv(index=False))
    note=root/'data/pilot_snapshot/README_PARTIAL_STORE.md'
    atomic_text(note,'This is an INCOMPLETE Zarr store. Only the receipt-listed chunks were acquired.\nNever read the full movie or treat absent chunks as real blank frames.\nThe guarded pilot loader reads exactly four validated frames, then keeps a central ROI.\n')
    return receipt(root,'sample',signature,[target,table],metadata_signature=meta['signature'],sample_id=cfg['sample_id'],frames=cfg['frames'],
        roi_shape_tzyx=list(np.stack(rois).shape),roi_offsets_zyx=offsets,full_dataset_downloaded=False,
        elapsed_scope='image decoding and caching, not model training')


def feature_stage(root,cfg):
    from .pilot_features import normalize_volume,physical_log,select_candidates,candidate_features,phase_shift,pair_features
    plan,meta=metadata_guard(root,cfg)
    sr=read_json(root/'outputs/pilot/sample_receipt.json')
    if sr.get('status')!='completed':raise ValueError('Sample gate is incomplete.')
    if sr.get('metadata_signature')!=meta['signature']:raise ValueError('Sample metadata changed; cached ROIs are not compatible.')
    for relative,digest in sr['artifacts'].items():verified_file(root,relative,digest)
    current_versions=versions()
    signature=canonical_hash({'sample':sr['signature'],'config':cfg,'code':code_signature(root),'versions':current_versions})
    old=completed(root,'features',signature)
    if old:return old
    cache=root/'data/pilot_snapshot/selected_roi.npz'
    if cache.stat().st_size>cfg['max_selected_decoded_bytes']:raise ValueError('Unexpected ROI file size.')
    with np.load(cache,allow_pickle=False) as z:
        images=z['images'];frames=z['frames'];offsets=z['offsets'];spacing=z['spacing_um']
    if images.shape!=(4,*cfg['roi_shape_zyx']) or frames.tolist()!=cfg['frames'] or not np.isfinite(images).all():
        raise ValueError('ROI cache violates its recorded shape/frame contract.')
    folder=root/'outputs/pilot/frame_features';folder.mkdir(parents=True,exist_ok=True)
    all_nodes=[];all_detectors=[];checkpoints=[];summaries=[]
    for i,t in enumerate(frames):
        tag=f'frame_{int(t):03d}';frame_receipt=folder/(tag+'.json');node_path=folder/(tag+'_nodes.csv');diag_path=folder/(tag+'_detectors.csv')
        fingerprint=canonical_hash({'feature_signature':signature,'frame':int(t)})
        saved=read_json(frame_receipt) if frame_receipt.exists() else {}
        if saved.get('signature')==fingerprint and saved.get('status')=='completed':
            for relative,digest in saved['artifacts'].items():verified_file(root,relative,digest)
            nodes=pd.read_csv(node_path);diags=pd.read_csv(diag_path)
            emit(root,'features','reused_verified_frame',frame=int(t))
        else:
            begin=time.monotonic();normal,stats=normalize_volume(images[i]);responses=[]
            for sigma in cfg['sigma_um']:
                responses.append(physical_log(normal,spacing,sigma))
                emit(root,'features','scale_response_completed',frame=int(t),sigma_um=sigma)
            multi=np.maximum.reduce(responses);single=responses[cfg['sigma_um'].index(3.)]
            coords,multi_diag=select_candidates(multi,spacing,cfg)
            _,single_diag=select_candidates(single,spacing,cfg)
            nodes=candidate_features(images[i],coords,responses,spacing,cfg['sigma_um'],int(t),offsets,cfg)
            if nodes.empty:raise ValueError('No candidates in a selected frame; inspect images before changing thresholds.')
            diags=pd.DataFrame([{'t':int(t),'recipe':'single_scale_3um',**single_diag},
                                {'t':int(t),'recipe':'multiscale_2_3_4um',**multi_diag}])
            atomic_text(node_path,nodes.to_csv(index=False));atomic_text(diag_path,diags.to_csv(index=False))
            atomic_json(frame_receipt,{'utc':utc(),'status':'completed','signature':fingerprint,
                'artifacts':records(root,[node_path,diag_path]),'frame':int(t),'normalization':stats,
                'elapsed_seconds':round(time.monotonic()-begin,3)})
            emit(root,'features','frame_feature_checkpoint',frame=int(t),candidates=len(nodes),
                elapsed_seconds=round(time.monotonic()-begin,3),completed_frames=i+1,total_frames=4)
        all_nodes.append(nodes);all_detectors.append(diags);checkpoints.extend([node_path,diag_path,frame_receipt])
    pairs=[];motion=[]
    for i in range(3):
        shift,diagnostic=phase_shift(images[i],images[i+1]);drift=shift*spacing
        motion.append({'t_source':int(frames[i]),'t_target':int(frames[i+1]),
            'drift_z_um':float(drift[0]),'drift_y_um':float(drift[1]),'drift_x_um':float(drift[2]),
            'drift_norm_um':float(np.linalg.norm(drift)),
            'mean_absolute_roi_change_raw':float(np.mean(np.abs(images[i+1].astype(float)-images[i].astype(float)))),**diagnostic})
        pairs.append(pair_features(all_nodes[i],all_nodes[i+1],drift,cfg))
        emit(root,'features','pair_feature_checkpoint',frame_pair=[int(frames[i]),int(frames[i+1])],candidate_pairs=len(pairs[-1]))
    node_table=pd.concat(all_nodes,ignore_index=True);pair_table=pd.concat(pairs,ignore_index=True)
    if pair_table.empty:raise ValueError('No adjacent-frame candidate pairs; inspect sample and pair gate before expanding.')
    tables={'node_features.csv':node_table,'pair_features.csv':pair_table,
            'detector_diagnostics.csv':pd.concat(all_detectors,ignore_index=True),'motion_diagnostics.csv':pd.DataFrame(motion)}
    audits=[]
    for kind,table in [('node',node_table),('pair',pair_table)]:
        if np.isinf(table.select_dtypes(include='number').to_numpy()).any():raise ValueError('Infinite feature values.')
        for column in table.select_dtypes(include='number').columns:
            if column in ('node_id','source_id','target_id','t','t_source','t_target'):continue
            audits.append({'table':kind,'feature':column,'missing_fraction':float(table[column].isna().mean()),
                           'unique_values':int(table[column].nunique()),'constant_in_this_pilot':bool(table[column].nunique()<=1)})
    tables['feature_quality.csv']=pd.DataFrame(audits)
    output_paths=[]
    for name,table in tables.items():
        p=root/'outputs/pilot'/name;atomic_text(p,table.to_csv(index=False));output_paths.append(p)
    summary_path=root/'outputs/pilot/feature_summary.json'
    atomic_json(summary_path,{'utc':utc(),'evaluation':'single training-crop unsupervised feature diagnostics',
        'sample_id':cfg['sample_id'],'frames':cfg['frames'],'node_rows':len(node_table),'pair_rows':len(pair_table),
        'node_numeric_feature_count':sum(a['table']=='node' for a in audits),
        'pair_numeric_feature_count':sum(a['table']=='pair' for a in audits),
        'detector_recipes':2,'candidate_cap_frames':int(pd.concat(all_detectors).query("recipe == 'multiscale_2_3_4um'")['candidate_cap_reached'].sum()),
        'training_fits':0,'ground_truth_used':False,'official_metric':None,'score_improvement':None,
        'versions':current_versions,'code_signature':code_signature(root),
        'limitations':['Not a tracking baseline','No annotation-based recall or precision','Not representative validation',
          'Morphology is an intensity-patch proxy, not a segmentation label','Crowding is conditional on capped candidate detections',
          'Image phase correlation is periodic, coarse, and unreliable for nonrigid/weak scenes; no biological velocity claim',
          'Only four central ROIs; not a whole video or whole embryo']})
    return receipt(root,'features',signature,[*output_paths,summary_path,*checkpoints],
        node_rows=len(node_table),pair_rows=len(pair_table),evaluation='feature integrity only; no competitive score',
        feature_families='multiscale, contrast, intensity-shape, candidate crowding, image drift, pair context')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['metadata','sample','features']);args=p.parse_args()
    root=project_root();cfg=read_json(root/'configs/pilot.json');validate_config(cfg)
    start=time.monotonic();emit(root,args.stage,'started',code_signature=code_signature(root))
    try:
        {'metadata':metadata_stage,'sample':sample_stage,'features':feature_stage}[args.stage](root,cfg)
    except BaseException as exc:
        atomic_json(root/f'outputs/pilot/{args.stage}_failure.json',{'utc':utc(),'stage':args.stage,
            'error_type':type(exc).__name__,'message':str(exc)[:800],'elapsed_seconds':round(time.monotonic()-start,3),
            'automatic_retry':False,'training_fits':0})
        emit(root,args.stage,'failed',error_type=type(exc).__name__,elapsed_seconds=round(time.monotonic()-start,3))
        raise
    emit(root,args.stage,'worker_finished',total_elapsed_seconds=round(time.monotonic()-start,3))

if __name__=='__main__':main()
