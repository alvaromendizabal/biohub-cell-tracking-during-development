"""Bounded local CPU worker; stages reuse verified data and never download or fit."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from .feature_io import (atomic,write_json,read_json,utc,digest,source_signature,
    fingerprint,artifacts,verify,checkpoint_valid,check_tests,load_inputs,emit,clean_message)


def _done(root,path,signature,paths,**details):
    value={'status':'completed','utc':utc(),'signature':signature,'artifacts':artifacts(root,paths),**details}
    write_json(path,value);return value


def round_worker(root,stage):
    import numpy as np
    import pandas as pd
    from .feature_audit import audit,planned_ablations
    root=Path(root).resolve();start=time.monotonic();check_tests(root)
    cfg,images,frames,offsets,spacing,nodes,pairs,provenance=load_inputs(root)
    if stage not in ['round1','round2']:raise ValueError('Only round1 or round2 is allowed.')
    if stage=='round2':
        parent=read_json(root/'outputs/features/round1_receipt.json')
        if parent.get('status')!='completed' or parent.get('source_signature')!=source_signature(root):raise ValueError('Run Round 1 successfully with this source before Round 2.')
        if parent.get('input_fingerprint')!=fingerprint(provenance):raise ValueError('Round 1 belongs to different inputs or environment.')
        for p,h in parent['artifacts'].items():verify(root,p,h)
        provenance['round1_signature']=parent['signature']
    signature=fingerprint({'stage':stage,'input':provenance,'configuration':cfg[stage]})
    pointer=root/f'outputs/features/{stage}_receipt.json'
    old=checkpoint_valid(root,pointer,signature)
    if old:
        emit(root,stage,'reused_complete_round',signature=signature,rows=old['rows']);return old
    folder=root/'outputs/features/runs'/signature[:20]/stage;folder.mkdir(parents=True,exist_ok=True)
    all_tables=[];all_divisions=[];parts=[]
    emit(root,stage,'started',signature=signature,network_requests=0,training_fits=0)
    declared=cfg[stage]['families']
    if stage=='round1':
        from .feature_round1 import frame_features
        for i,t in enumerate(frames):
            table_path=folder/f'frame_{int(t):03d}.csv';part=folder/f'frame_{int(t):03d}.json';part_sig=fingerprint([signature,int(t)])
            cached=checkpoint_valid(root,part,part_sig)
            if cached:
                table=pd.read_csv(table_path);event='reused_frame'
            else:
                selected=nodes.loc[nodes.t==int(t)].copy()
                table=frame_features(images[i],selected,spacing,offsets,declared,
                      emit=lambda **kw:emit(root,stage,kw.pop('event'),frame=int(t),**kw))
                atomic(table_path,table.to_csv(index=False));_done(root,part,part_sig,[table_path],frame=int(t));event='completed_frame'
            all_tables.append(table);parts.extend([table_path,part])
            emit(root,stage,event,completed=i+1,total=len(frames),rows=len(table),elapsed_seconds=round(time.monotonic()-start,3))
        table=pd.concat(all_tables,ignore_index=True)
        if table.node_id.duplicated().any() or set(table.node_id)!=set(nodes.node_id):raise ValueError('Round 1 changed the fixed candidate set.')
        path=folder/'node_features.csv';prefix='r1_';division_path=None
    else:
        from .feature_round2 import make_context,pair_frame_features
        r1=pd.read_csv(root/parent['table'])
        context=make_context(images,frames,offsets,spacing,nodes,r1)
        frame_map={int(t):nodes.loc[nodes.t==int(t)].copy() for t in frames}
        for i,t in enumerate(frames[:-1]):
            table_path=folder/f'pair_{int(t):03d}.csv';div_path=folder/f'division_{int(t):03d}.csv';part=folder/f'pair_{int(t):03d}.json';part_sig=fingerprint([signature,int(t)])
            cached=checkpoint_valid(root,part,part_sig)
            if cached:
                table=pd.read_csv(table_path);division=pd.read_csv(div_path);event='reused_frame_pair'
            else:
                selected=pairs.loc[pairs.t_source==int(t)].copy()
                table,division=pair_frame_features(selected,pairs,context,frame_map,declared,cfg[stage]['max_siblings_per_edge'],
                    emit=lambda **kw:emit(root,stage,kw.pop('event'),source_frame=int(t),**kw))
                atomic(table_path,table.to_csv(index=False));atomic(div_path,division.to_csv(index=False));_done(root,part,part_sig,[table_path,div_path],source_frame=int(t));event='completed_frame_pair'
            all_tables.append(table);all_divisions.append(division);parts.extend([table_path,div_path,part])
            emit(root,stage,event,completed=i+1,total=len(frames)-1,rows=len(table),elapsed_seconds=round(time.monotonic()-start,3))
        table=pd.concat(all_tables,ignore_index=True)
        if table.duplicated(['source_id','target_id']).any() or set(zip(table.source_id,table.target_id))!=set(zip(pairs.source_id,pairs.target_id)):raise ValueError('Round 2 changed the fixed candidate-pair set.')
        path=folder/'pair_features.csv';prefix='r2_';division_path=folder/'division_hypotheses.csv'
        atomic(division_path,pd.concat(all_divisions,ignore_index=True).drop_duplicates(['source_id','daughter_a','daughter_b']).to_csv(index=False));parts.append(division_path)
    quality,redundancy=audit(table,prefix)
    audit_path=folder/'quality.csv';redundancy_path=folder/'redundancy.csv';plan=folder/'ablation_plan.json'
    atomic(path,table.to_csv(index=False));atomic(audit_path,quality.to_csv(index=False));atomic(redundancy_path,redundancy.to_csv(index=False))
    write_json(plan,{'status':'blocked_pending_labeled_embryo_disjoint_baseline_and_official_scorer','training_fits':0,'experiments':planned_ablations(declared)})
    parts.extend([path,audit_path,redundancy_path,plan])
    result=_done(root,pointer,signature,parts,source_signature=source_signature(root),
        input_fingerprint=fingerprint({k:v for k,v in provenance.items() if k!='round1_signature'}),
        stage=stage,rows=len(table),descriptor_columns=128,family_count=8,table=path.relative_to(root).as_posix(),
        quality=audit_path.relative_to(root).as_posix(),redundancy=redundancy_path.relative_to(root).as_posix(),
        division_table=division_path.relative_to(root).as_posix() if division_path else None,
        training_fits=0,network_requests=0,official_score=None,score_delta=None,leaderboard_score=None,
        evaluation='single training-crop feature-functionality diagnostics; no labels',elapsed_seconds=round(time.monotonic()-start,3))
    emit(root,stage,'completed',rows=len(table),descriptor_columns=128,elapsed_seconds=result['elapsed_seconds'])
    return result


def _terminate(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=3)


def run_round(stage,root):
    import psutil
    root=Path(root).resolve();check_tests(root)
    cfg=read_json(root/'configs/feature_rounds.json')
    if stage not in ['round1','round2']:raise ValueError('Only round1 or round2 is allowed.')
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    folder=root/'outputs/features/processes'/run_id;folder.mkdir(parents=True)
    stdout=folder/'stdout.jsonl';stderr=folder/'stderr_private.txt';proc=None;position=0;start=time.monotonic();peak=0
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    command=[sys.executable,'-m','biohub_tracking.feature_runner','--root',str(root),'--stage',stage,'--worker']
    try:
        with stdout.open('w') as out,stderr.open('w') as err:
            proc=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
            while proc.poll() is None:
                time.sleep(.25)
                with stdout.open() as f:f.seek(position);text=f.read(32000);position=f.tell()
                if text:print(text,end='',flush=True)
                if time.monotonic()-start>cfg[stage]['wall_seconds']:raise TimeoutError(f'{stage} exceeded its {cfg[stage]["wall_seconds"]}-second wall limit.')
                try:
                    parent=psutil.Process(proc.pid);processes=[parent,*parent.children(recursive=True)]
                    used=sum(p.memory_info().rss for p in processes if p.is_running());peak=max(peak,used)
                    if used>cfg['memory_GiB']*1024**3:raise MemoryError('Worker process-tree memory limit exceeded.')
                except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
        with stdout.open() as f:f.seek(position);print(f.read(32000),end='',flush=True)
        if proc.returncode!=0:
            failure=root/f'outputs/features/{stage}_failure.json'
            detail=read_json(failure).get('message','') if failure.exists() else ''
            raise RuntimeError(f'{stage} worker exited {proc.returncode}. {detail}')
        receipt=read_json(root/f'outputs/features/{stage}_receipt.json')
        write_json(folder/'supervisor.json',{'utc':utc(),'status':'completed','stage':stage,'wall_seconds':round(time.monotonic()-start,3),'peak_process_tree_GiB':round(peak/1024**3,3)})
        return receipt
    except BaseException as exc:
        if proc is not None:_terminate(proc)
        message=clean_message(exc)
        write_json(folder/'supervisor.json',{'utc':utc(),'status':'failed','stage':stage,'message':message,'wall_seconds':round(time.monotonic()-start,3)})
        print('FEATURE_ROUND_STOPPED: '+message,flush=True)
        print('Run: python scripts/build_feature_return_package.py',flush=True)
        raise
    finally:
        if proc is not None and proc.poll() is None:_terminate(proc)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',choices=['round1','round2'],required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    try:
        result=round_worker(args.root,args.stage) if args.worker else run_round(args.stage,args.root)
        print(args.stage.upper()+'_COMPLETE',flush=True)
    except BaseException as exc:
        write_json(args.root/f'outputs/features/{args.stage}_failure.json',{'utc':utc(),'stage':args.stage,'message':clean_message(exc),'error_type':type(exc).__name__})
        raise

if __name__=='__main__':main()
