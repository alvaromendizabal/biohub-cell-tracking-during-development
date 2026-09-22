"""AWS-only five-movie cross-embryo appearance/geometry association ablation."""
from __future__ import annotations
import os,sys,json,gzip,hashlib,time,traceback,gc,csv,pickle,platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import scipy,sklearn
import appearance_edges as ae
import error_worker as ew

ROOT=Path(os.environ.get('BIOHUB_PROJECT_ROOT','/home/sagemaker-user/biohub-cell-tracking-during-development')).resolve()
OUT=Path(os.environ.get('BIOHUB_ASSOCIATION_OUT',str(ROOT/'outputs/competitive_gap_closure/appearance_association'))).resolve()
HERE=Path(__file__).resolve().parent
OLD=ROOT/'outputs/competitive_gap_closure/temporal_reassignment'
PRIOR=ROOT/'outputs/competitive_gap_closure/competing_parent_events'
CACHE=ROOT/'data/biohub_validation_cache'
COHORT=['44b6_12dfb391','6bba_062c8d37','44b6_267148e4','6bba_07e24132','44b6_2a2eff9f']
ARMS=('geometry','geometry_appearance')
read,save,sha,log=ew.read,ew.save,ew.sha,ew.log

def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def np_save(path,**kw):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def reused(folder,contract,names):
 done=folder/'done.json'
 if not done.is_file():return False
 d=read(done);ae.require(d['contract']==contract,'CACHE_CONTRACT_CHANGED '+str(folder))
 for n in names:ae.require((folder/n).is_file() and sha(folder/n)==d['files'].get(n),'CACHE_CORRUPT '+str(folder/n))
 return True
def seal(folder,contract,names):save(folder/'done.json',{'status':'complete','contract':contract,'files':{n:sha(folder/n) for n in names}})
def graph_write(path,nodes,edges):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with gzip.open(tmp,'wt') as f:json.dump({'nodes':nodes,'edges':edges},f,separators=(',',':'),allow_nan=False)
 os.replace(tmp,path)
def progress(phase,**kw):save(OUT/'progress_0.json',dict(phase=phase,**kw));log(phase+' '+json.dumps(kw))
def summary(rows):
 w=sum(r['edge_tp']+r['edge_fp']+r['edge_fn'] for r in rows)
 c={k:sum(int(r[k]) for r in rows) for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')}
 adj=sum(r['adj_edge_jaccard']*(r['edge_tp']+r['edge_fp']+r['edge_fn']) for r in rows)/max(w,1)
 div=c['division_tp']/max(c['division_tp']+c['division_fp']+c['division_fn'],1)
 return dict(c,adj_edge_jaccard=adj,division_jaccard=div,score=adj+.1*div,n=len(rows))
def evaluate(nodes,edges,stem,gt):
 from pinned_metric import metrics
 g,to_orig=ew.metric_graph(nodes,edges);m=metrics.evaluate(g,gt,scale=ew.SCALE,max_distance=7.)
 row=metrics.per_sample_metrics(m,ew.estimated_nodes(CACHE/'train'/(stem+'.geff')),metrics.node_recall(g,gt))
 row.update(stem=stem,embryo=stem.split('_')[0]);return row,g,to_orig

def image_contract(state,stem):
 prefix=f'train/{stem}.zarr';arraykey=prefix+'/0';plan=state['plans'][stem].get(arraykey)
 ae.require(plan is not None and len(plan['shape'])==4,'MISSING_TZYX_IMAGE_PLAN')
 keys={prefix+'/zarr.json',arraykey+'/zarr.json',*(arraykey+'/'+k for k in plan['keys'])}
 for k in keys:
  ae.require(not Path(k).is_absolute() and '..' not in Path(k).parts,'IMAGE_PATH_ESCAPE')
  p=CACHE/k;e=state['entries'].get(k);ae.require(e is not None,'MISSING_IMAGE_RECEIPT '+k)
  if e.get('status')=='complete':
   ae.require(p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'IMAGE_CACHE_INVALID '+k)
  else:ae.require(e.get('meaning')=='zarr_chunk_absent_fill_value' and not p.exists(),'UNVERIFIED_IMAGE_CHUNK '+k)
 return signature({k:state['entries'][k] for k in sorted(keys)}),plan

def extract_profiles(stem,nodes,pairs,contract,shape):
 folder=OUT/'appearance'/stem;names=['descriptors.npz','receipt.json']
 if reused(folder,contract,names):return folder/'descriptors.npz',True
 import zarr
 array=zarr.open_array(str(CACHE/'train'/(stem+'.zarr')/'0'),mode='r')
 ae.require(tuple(array.shape)==tuple(shape) and len(array.shape)==4,'ZARR_SHAPE_CONTRACT')
 needed=sorted(set(pairs[:,:2].reshape(-1).tolist()));frames={}
 for i in needed:
  t=int(nodes[i]['t']);ae.require(0<=t<shape[0],'NODE_FRAME_OUT_OF_RANGE');frames.setdefault(t,[]).append(i)
 all_ids=[];all_profiles=[];all_stats=[];newframes=0;reuseframes=0
 for j,(t,ids) in enumerate(sorted(frames.items())):
  fd=folder/'frames'/str(t);fc=signature({'input':contract,'t':t,'ids':ids})
  if reused(fd,fc,['frame.npz']):reuseframes+=1
  else:
   progress('image_profile_frame',current_movie=stem,frame=t,completed_frames=j,total_frames=len(frames))
   frame=np.asarray(array[t]);coords=np.asarray([[nodes[i][k] for k in ('z','y','x')] for i in ids],float)
   ae.require(np.isfinite(coords).all() and (coords>=-.5).all() and (coords<=np.asarray(shape[1:])-.5).all(),'IMAGE_COORDINATE_BOUNDS')
   profiles,stats=ae.frame_profiles(frame,coords)
   np_save(fd/'frame.npz',ids=np.asarray(ids,np.int64),profiles=profiles,stats=stats);seal(fd,fc,['frame.npz']);newframes+=1
   del frame,coords
  with np.load(fd/'frame.npz',allow_pickle=False) as z:
   ae.require(np.array_equal(z['ids'],np.asarray(ids)),'FRAME_NODE_IDS_CHANGED')
   all_ids.append(z['ids']);all_profiles.append(z['profiles']);all_stats.append(z['stats'])
 ae.require(all_ids,'NO_ELIGIBLE_IMAGE_NODES')
 np_save(folder/'descriptors.npz',ids=np.concatenate(all_ids),profiles=np.concatenate(all_profiles),stats=np.concatenate(all_stats))
 save(folder/'receipt.json',{'new_frames':newframes,'reused_frames':reuseframes,'nodes':len(needed),'frames':len(frames),'image_shape':shape,
   'source':'immutable AWS Zarr cache','label_free':True,'physical_offsets_um':[-3,-1.5,0,1.5,3]})
 seal(folder,contract,names);return folder/'descriptors.npz',False

def previous_result_summary():
 r=read(PRIOR/'result.json')
 ae.require(r['status']=='complete' and r['decision']=='reject_fixed_competing_parent_event_model','PRIOR_EVENT_RESULT_NOT_REJECTED')
 ae.require(r['model_fits']==2 and r['accepted_new_forks']==108 and abs(r['score_delta'])<1e-12,'PRIOR_RESULT_CHANGED')
 return {'decision':r['decision'],'forks_added':108,'reassigned_children':r['reassigned_children'],'score_delta':0.,'prior_fits_not_repeated':2}

def main():
 ew.offline_guard();OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 ew.ROOT=ROOT;ew.OLD=OLD;ew.CACHE=CACHE
 from pinned_metric import metrics
 from tracksdata.options import set_options
 set_options(show_progress=False)
 import zarr,tracksdata,polars
 code={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}
 codeid=signature(code);expected=read(HERE/'expected_baselines.json');prior=previous_result_summary()
 environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
              'zarr':zarr.__version__,'polars':polars.__version__}
 save(OUT/'environment.json',environment)
 save(OUT/'experiment_contract.json',{'config':asdict(ae.CONFIG),'geometry_features':list(ae.GEO_NAMES),'appearance_features':list(ae.APP_NAMES),
      'cohort':COHORT,'arms':list(ARMS),'max_model_fits':4,'training':'leave one embryo out; no random early-stopping split; fixed configuration',
      'prior_result':prior,'existing_models_reused':'frozen detector predictions and all five baseline graphs; rejected event model not used',
      'new_method':'learned nonlinear continuation association with matched image-ablation and balanced global assignment',
      'not_claimed':'not a Trackastra, HOCT, or 0.974 winning-system reproduction',
      'source_sha256':codeid,'no_downloads':True,'no_gpu_inference':True,'approved_for_submission':False})
 state=read(ROOT/'outputs/competitive_gap_closure/division_retention_aws/data_object_manifest.json')
 graphs={};data={};inputs={};feature_reuse=0;descriptor_reuse=0
 # Candidate and appearance generation is completed for all videos before annotations.
 for i,stem in enumerate(COHORT):
  progress('feature_inputs',current_movie=stem,completed_movies=i,total_movies=5)
  src=OLD/'videos'/stem/'frozen'/'graph.json.gz';rr=read(src.with_name('receipt.json'))
  ae.require(rr['status']=='complete' and rr['parity_passed'] is True,'UNVERIFIED_BASELINE')
  ae.require(sha(src)==rr['graph_sha256']==expected[stem]['graph_sha256'],'FROZEN_GRAPH_CHANGED')
  nodes,edges=ew.graph_load(src);graphs[stem]=(nodes,edges)
  image_hash,plan=image_contract(state,stem)
  contract=signature({'baseline':sha(src),'image':image_hash,'code':codeid,'environment':environment,'config':asdict(ae.CONFIG)})
  inputs[stem]=contract;folder=OUT/'features'/stem
  if reused(folder,contract,['pairs.npz','coverage.json']):feature_reuse+=1
  else:
   pairs,geo,inc,counts=ae.candidates(nodes,edges,heartbeat=lambda t,n:progress('label_free_candidate_pairs',current_movie=stem,frame=t,pairs=n))
   ae.require(len(pairs)>0,'EMPTY_ELIGIBLE_CANDIDATES')
   desc,wasreused=extract_profiles(stem,nodes,pairs,contract,plan['shape']);descriptor_reuse+=int(wasreused)
   with np.load(desc,allow_pickle=False) as z:app=ae.appearance_features(pairs,z['ids'],z['profiles'],z['stats'])
   np_save(folder/'pairs.npz',pairs=pairs,geometry=geo,appearance=app,incumbent=inc)
   save(folder/'coverage.json',dict(counts,stem=stem,geometry_count=len(ae.GEO_NAMES),appearance_count=len(ae.APP_NAMES),labels_read=False))
   seal(folder,contract,['pairs.npz','coverage.json'])
  with np.load(folder/'pairs.npz',allow_pickle=False) as z:data[stem]={k:z[k] for k in z.files}
 save(OUT/'all_features_frozen_before_labels.json',{'files':{s:sha(OUT/'features'/s/'pairs.npz') for s in COHORT},'labels_read':False})
 baseline_rows=[];coverage=[];gt_hashes={};matched_maps={};gt_pairs={}
 for i,stem in enumerate(COHORT):
  progress('baseline_and_training_labels',current_movie=stem,completed_movies=i,total_movies=5)
  gt_hashes[stem]=ew.verify_gt(state,stem);gt=ew.load_geff(CACHE/'train'/(stem+'.geff'))
  nodes,edges=graphs[stem];row,g,orig=evaluate(nodes,edges,stem,gt);ew.verify_metric(row,expected[stem]['metric']);baseline_rows.append(row)
  matched=ew.matching_map(g,orig);matched_maps[stem]=matched
  _,ge=ew.nodes_edges(gt);gep={(e['source_id'],e['target_id']) for e in ge};gt_pairs[stem]=gep
  d=data[stem];y=ae.sparse_labels(d['pairs'],matched,gep);d['labels']=y
  # Exact sparse-label parity on EVERY generated incumbent edge.
  K=tracksdata.DEFAULT_ATTR_KEYS
  ev={(orig[int(e[K.EDGE_SOURCE])],orig[int(e[K.EDGE_TARGET])]):1 if e[K.MATCHED_EDGE_MASK] else (0 if e['pred_valid'] else -1)
          for e in metrics._evaluate_matched_graph(g,gt).iter_rows(named=True)}
  for k in np.flatnonzero(d['incumbent']):
   a,b,_=map(int,d['pairs'][k]);ae.require(int(y[k])==ev[(a,b)],'SPARSE_LABEL_PARITY_FAILED')
  gt_mapped={(matched[int(a)],matched[int(b)]) for (a,b,t),v in zip(d['pairs'],y) if v==1}
  cov=dict(read(OUT/'features'/stem/'coverage.json'),embryo=stem.split('_')[0],positive_pairs=int(np.sum(y==1)),
       negative_pairs=int(np.sum(y==0)),ignored_pairs=int(np.sum(y<0)),distinct_positive_gt_edges=len(gt_mapped),
       eligible_incumbent_tp=int(np.sum(d['incumbent']&(y==1))),eligible_incumbent_fp=int(np.sum(d['incumbent']&(y==0))))
  coverage.append(cov);np_save(OUT/'labels'/(stem+'.npz'),labels=y)
  del gt,g;gc.collect()
 save(OUT/'coverage.json',coverage)
 readiness=[]
 for heldout in ('44b6','6bba'):
  training=[s for s in COHORT if not s.startswith(heldout+'_')];testing=[s for s in COHORT if s.startswith(heldout+'_')]
  ae.require(not set(training)&set(testing),'EMBRYO_LEAKAGE')
  positive=sum(c['positive_pairs'] for c in coverage if c['stem'] in training);negative=sum(c['negative_pairs'] for c in coverage if c['stem'] in training)
  readiness.append({'heldout_embryo':heldout,'train_movies':training,'test_movies':testing,'positive_pairs':positive,'negative_pairs':negative,
       'passed':positive>=ae.CONFIG.minimum_training_positive and negative>=ae.CONFIG.minimum_training_negative})
 save(OUT/'training_readiness.json',readiness)
 save(OUT/'fit_progress.json',{'new_fits':0,'reused_fits':0})
 if not all(x['passed'] for x in readiness):
  save(OUT/'result.json',{'status':'complete','decision':'stop_insufficient_cross_embryo_edge_labels','baseline':summary(baseline_rows),
       'score_delta':None,'coverage':coverage,'new_fits':0,'reused_fits':0,'measured_movies':COHORT,'approved_for_submission':False,
       'new_prediction_graphs':0,'feature_reuse':feature_reuse,'models':[],'arms':{},'prior_result':prior});return
 models={};new_fits=0;reused_fits=0;model_audit=[]
 for fold in readiness:
  heldout=fold['heldout_embryo'];train=fold['train_movies'];test=fold['test_movies']
  for arm in ARMS:
   folder=OUT/'models'/heldout/arm
   contract=signature({'training_inputs':{s:inputs[s] for s in train},'labels':{s:sha(OUT/'labels'/(s+'.npz')) for s in train},'arm':arm,'fold':fold})
   if reused(folder,contract,['model.pkl','audit.json']):
    with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
    audit=read(folder/'audit.json');reused_fits+=1
   else:
    progress('fit_training_embryo_only',heldout_embryo=heldout,arm=arm,completed_fits=new_fits+reused_fits,training_movies=train)
    def xx(s):return data[s]['geometry'] if arm=='geometry' else np.column_stack([data[s]['geometry'],data[s]['appearance']])
    x=np.concatenate([xx(s) for s in train]);y=np.concatenate([data[s]['labels'] for s in train])
    model,audit=ae.train(x,y);audit.update(heldout_embryo=heldout,training_movies=train,test_movies=test,arm=arm,model_class=type(model).__name__)
    folder.mkdir(parents=True,exist_ok=True);tmp=folder/'model.pkl.partial'
    with tmp.open('wb') as f:pickle.dump(model,f,protocol=5)
    os.replace(tmp,folder/'model.pkl');save(folder/'audit.json',audit);seal(folder,contract,['model.pkl','audit.json']);new_fits+=1
    save(OUT/'fit_progress.json',{'new_fits':new_fits,'reused_fits':reused_fits});del x,y
   models[(heldout,arm)]=model;model_audit.append(audit)
 save(OUT/'model_audit.json',model_audit);save(OUT/'fit_progress.json',{'new_fits':new_fits,'reused_fits':reused_fits})
 transactions={};newgraphs=0;reusegraphs=0
 for stem in COHORT:
  d=data[stem];nodes,edges=graphs[stem];heldout=stem.split('_')[0]
  for arm in ARMS:
   folder=OUT/'videos'/stem/arm;modelpath=OUT/'models'/heldout/arm/'model.pkl';contract=signature({'input':inputs[stem],'model':sha(modelpath)})
   if reused(folder,contract,['graph.json.gz','scores.npz','transaction.json']):reusegraphs+=1
   else:
    progress('heldout_assignment_no_labels',current_movie=stem,arm=arm)
    x=d['geometry'] if arm=='geometry' else np.column_stack([d['geometry'],d['appearance']])
    scores=models[(heldout,arm)].predict_proba(x)[:,1]
    candidate,transaction=ae.decode(nodes,edges,d['pairs'],scores,d['incumbent'])
    graph_write(folder/'graph.json.gz',nodes,candidate);np_save(folder/'scores.npz',scores=scores)
    save(folder/'transaction.json',dict(transaction,stem=stem,arm=arm,trained_on=[s for s in COHORT if not s.startswith(heldout+'_')]))
    seal(folder,contract,['graph.json.gz','scores.npz','transaction.json']);newgraphs+=1
   transactions[(stem,arm)]=read(folder/'transaction.json')
 save(OUT/'all_candidate_graphs_frozen_before_scoring.json',{'graphs':{s+'/'+a:sha(OUT/'videos'/s/a/'graph.json.gz') for s in COHORT for a in ARMS}})
 rows={a:[] for a in ARMS};modification_rows=[];metric_reuse=0
 for stem in COHORT:
  for arm in ARMS:
   folder=OUT/'videos'/stem/arm;mp=folder/'metric.json';done=folder/'metric_done.json'
   contract=signature({'graph':sha(folder/'graph.json.gz'),'gt':gt_hashes[stem],'code':codeid})
   if done.is_file():
    dr=read(done);ae.require(dr['contract']==contract and sha(mp)==dr['sha256'],'METRIC_CACHE_CORRUPT');row=read(mp);metric_reuse+=1
   else:
    progress('exact_heldout_metric',current_movie=stem,arm=arm)
    nodes,edges=ew.graph_load(folder/'graph.json.gz');gt=ew.load_geff(CACHE/'train'/(stem+'.geff'));row,g,_=evaluate(nodes,edges,stem,gt)
    save(mp,row);save(done,{'contract':contract,'sha256':sha(mp)});del g,gt;gc.collect()
   rows[arm].append(row)
   tx=transactions[(stem,arm)]
   changes=tx['changes'];oldp=np.asarray([(c['source_id'],c['old_target_id'],0) for c in changes],np.int64).reshape(-1,3)
   newp=np.asarray([(c['source_id'],c['new_target_id'],0) for c in changes],np.int64).reshape(-1,3)
   oldy=ae.sparse_labels(oldp,matched_maps[stem],gt_pairs[stem]);newy=ae.sparse_labels(newp,matched_maps[stem],gt_pairs[stem])
   modification_rows.append({'stem':stem,'arm':arm,'changed_edges':len(changes),'cycles':tx['accepted_cycles'],
       'new_true_positive_edges':int(np.sum((oldy!=1)&(newy==1))),'lost_true_positive_edges':int(np.sum((oldy==1)&(newy!=1))),
       'old_false_positive_edges':int(np.sum(oldy==0)),'new_false_positive_edges':int(np.sum(newy==0)),
       'both_unassessable':int(np.sum((oldy<0)&(newy<0)))})
 save(OUT/'changed_edge_audit.json',modification_rows)
 base=summary(baseline_rows);arms={};embryos={}
 for arm in ARMS:
  stat=summary(rows[arm]);delta=stat['score']-base['score'];sub=[]
  for e in ('44b6','6bba'):
   b=summary([r for r in baseline_rows if r['embryo']==e]);c=summary([r for r in rows[arm] if r['embryo']==e]);sub.append({'embryo':e,'baseline':b,'candidate':c,'delta':c['score']-b['score']})
  changed=sum(x['changed_edges'] for x in modification_rows if x['arm']==arm)
  promote=changed>0 and delta>=.002 and stat['adj_edge_jaccard']>=base['adj_edge_jaccard']-.0005 and all(r['delta']>=-1e-10 for r in sub)
  promote=promote and stat['division_tp']>=base['division_tp'] and stat['division_fp']<=base['division_fp']
  arms[arm]={'metric':stat,'score_delta':delta,'changed_edges':changed,'screen_passed':bool(promote),'embryos':sub,'rows':rows[arm]}
 gain=arms['geometry_appearance']['metric']['score']-arms['geometry']['metric']['score']
 passes=[a for a in ARMS if arms[a]['screen_passed']]
 decision='independent_confirmation_required' if passes else 'reject_fixed_appearance_association_screen'
 result={'status':'complete','decision':decision,'baseline':base,'baseline_rows':baseline_rows,'arms':arms,'appearance_increment':gain,
       'score_delta':arms['geometry_appearance']['score_delta'],'new_fits':new_fits,'reused_fits':reused_fits,'models':model_audit,
       'new_prediction_graphs':newgraphs,'reused_prediction_graphs':reusegraphs,'reused_metrics':metric_reuse,'feature_reuse':feature_reuse,
       'coverage':coverage,'modifications':modification_rows,'approved_for_submission':False,'measured_movies':COHORT,'prior_result':prior,
       'official_score_snapshot':.947,'leader_snapshot':.974,'fresh_scoring_query':False,'network_requests':0,'new_downloads':0,
       'model_fits':new_fits,'gpu_inference_calls':0,'main_network_inference_calls':0,'elapsed_seconds':time.monotonic()-start,
       'limitations':['Five repeatedly inspected development movies, two embryo groups; pretrained data independence unverified.',
         'Association-only intervention protects detected divisions; it cannot add missing divisions or missing detections.',
         'Hand-designed appearance profiles plus nonlinear edge scoring; not a learned image backbone or winning-system reproduction.',
         'Model scores are not externally calibrated. No tuning is performed on this held-out result.']}
 save(OUT/'result.json',result)
 with (OUT/'comparison.csv').open('w',newline='') as f:
  keys=['stem','embryo','arm','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','adj_edge_jaccard']
  writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
  for arm,rr in [('frozen',baseline_rows),*rows.items()]:
   for r in rr:writer.writerow(dict(r,arm=arm))
 progress('complete',decision=decision,score_delta=result['score_delta'],completed_movies=5,total_movies=5)

if __name__=='__main__':
 try:main()
 except BaseException as e:save(OUT/'worker_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
