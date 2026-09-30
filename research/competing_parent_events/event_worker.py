"""Two embryo-disjoint fits, offline candidate generation, pinned metric validation.

The annotation-assisted error audit is used as TRAINING LABELS and evaluation only.
Candidate features and hypotheses are completed before any annotations are opened.
"""
from __future__ import annotations
import os,sys,json,gzip,hashlib,time,traceback,gc,csv
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
import division_events as ev
import error_worker as ew
from diagnostics import scalar_score

ROOT=Path(os.environ['BIOHUB_PROJECT_ROOT']).resolve()
OUT=Path(os.environ['BIOHUB_EVENT_OUT']).resolve()
HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'outputs/competitive_gap_closure/exact_error_budget'
OLD=ROOT/'outputs/competitive_gap_closure/temporal_reassignment'
CACHE=ROOT/'data/biohub_validation_cache'
COHORT=['44b6_12dfb391','6bba_062c8d37','44b6_267148e4','6bba_07e24132','44b6_2a2eff9f']
read,save,sha,log=ew.read,ew.save,ew.sha,ew.log

def file_contract(inputs):
 return hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def np_save(path,**kw):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
 with temp.open('wb') as f:np.savez_compressed(f,**kw)
 os.replace(temp,path)
def completed(folder,contract,files):
 p=folder/'done.json'
 if not p.is_file():return False
 d=read(p);ev.require(d['contract']==contract,'COMPLETED_CONTRACT_CHANGED '+str(folder))
 for n in files:ev.require((folder/n).is_file() and sha(folder/n)==d['files'].get(n),'CORRUPT_COMPLETED '+str(folder/n))
 return True
def done(folder,contract,files):save(folder/'done.json',{'contract':contract,'files':{n:sha(folder/n) for n in files},'status':'complete'})
def graph_write(path,nodes,edges):
 temp=path.with_name(path.name+'.partial');path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(temp,'wt') as f:json.dump({'nodes':nodes,'edges':edges},f,separators=(',',':'),allow_nan=False)
 os.replace(temp,path)
def progress(phase,**kw):save(OUT/'progress_0.json',{'phase':phase,**kw});log(phase+' '+json.dumps(kw))
def summary(rows):
 w=sum(r['edge_tp']+r['edge_fp']+r['edge_fn'] for r in rows)
 c={k:sum(int(r[k]) for r in rows) for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')}
 adj=sum(r['adj_edge_jaccard']*(r['edge_tp']+r['edge_fp']+r['edge_fn']) for r in rows)/max(w,1)
 div=c['division_tp']/max(c['division_tp']+c['division_fp']+c['division_fn'],1)
 return dict(c,adj_edge_jaccard=adj,division_jaccard=div,score=adj+.1*div,n=len(rows))
def evaluation(nodes,edges,stem,gt):
 from pinned_metric import metrics
 g,to_orig=ew.metric_graph(nodes,edges)
 m=metrics.evaluate(g,gt,scale=ew.SCALE,max_distance=7.)
 row=metrics.per_sample_metrics(m,ew.estimated_nodes(CACHE/'train'/(stem+'.geff')),metrics.node_recall(g,gt))
 row.update(stem=stem,embryo=stem.split('_')[0]);return row,g,to_orig

def main():
 ew.offline_guard();OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 save(OUT/'fit_progress.json',{'new_fits':0,'reused_fits':0})
 ew.ROOT=ROOT;ew.CACHE=CACHE;ew.OLD=OLD
 expected=read(HERE/'expected_baselines.json')
 old=read(OLD/'result.json');ev.require(old['status']=='complete' and old['measured_movies']==COHORT,'FIVE_MOVIE_BASELINE_CONTRACT_CHANGED')
 code={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}
 save(OUT/'experiment_contract.json',{'cohort':COHORT,'config':asdict(ev.CONFIG),'features':list(ev.FEATURES),
   'training_split':'leave one embryo out; all learning and threshold selection confined to training embryo',
   'labels':'sparse event labels; unassessable events ignored, never blanket negative',
   'proposal_semantics':'generate unary-to-division alternatives, including replacement of an incumbent nondivision parent; existing division edges protected',
   'network_requests':0,'new_downloads':0,'max_fits':2,'approved_for_submission':False,
   'provenance_note':'New lightweight division-event model. NOT the HOCT architecture or a winning-pipeline reproduction.',
   'limits':'tiny six-event development cohort, repeatedly inspected; not an independent final assessment',
   'code_sha256':file_contract(code)})
 contracts={};proposal_data={};graphs={};proposal_counts={};proposal_reuse=0
 # Do not open ground truth or the diagnostic audit until ALL proposals are frozen.
 for i,stem in enumerate(COHORT):
  src=OLD/'videos'/stem/'frozen'/'graph.json.gz';receipt=read(src.with_name('receipt.json'))
  ev.require(receipt['status']=='complete' and receipt.get('parity_passed') is True,'BASELINE_NOT_VERIFIED')
  ev.require(src.is_file() and sha(src)==expected[stem]['graph_sha256']==receipt['graph_sha256'],'BASELINE_HASH_CHANGED')
  nodes,edges=ew.graph_load(src);ev.structure(nodes,edges);graphs[stem]=(nodes,edges)
  contract=file_contract({'code':code,'baseline_sha256':sha(src),'config':asdict(ev.CONFIG)})
  contracts[stem]=contract;folder=OUT/'proposals'/stem
  if completed(folder,contract,['events.npz','counts.json']):
   proposal_reuse+=1;progress('proposal_reuse',current_movie=stem,completed_movies=i,total_movies=5)
  else:
   progress('label_free_proposals',current_movie=stem,completed_movies=i,total_movies=5)
   last=[0.]
   def pulse(t,count):
    if time.monotonic()-last[0]>10:
     last[0]=time.monotonic();progress('label_free_proposals',current_movie=stem,frame=t,candidate_events=count,completed_movies=i,total_movies=5)
   ids,x,counts=ev.propose(nodes,edges,heartbeat=pulse)
   np_save(folder/'events.npz',ids=ids,features=x)
   save(folder/'counts.json',dict(counts,stem=stem,rows=len(ids),feature_count=x.shape[1],labels_read=False))
   done(folder,contract,['events.npz','counts.json'])
  with np.load(folder/'events.npz',allow_pickle=False) as z:proposal_data[stem]=(z['ids'],z['features'])
  proposal_counts[stem]=read(folder/'counts.json')
 save(OUT/'all_proposals_frozen_before_annotations.json',{'status':'complete','proposals':{s:sha(OUT/'proposals'/s/'events.npz') for s in COHORT},'label_free':True})
 progress('sparse_training_label_construction',completed_movies=0,total_movies=5)
 audit=read(AUDIT/'result.json');ev.require(audit['status']=='complete' and audit['measured_movies']==COHORT,'EXACT_ERROR_AUDIT_REQUIRED')
 ev.require(audit['counts']=={'division_fn':5,'division_fp':1,'division_tp':1,'edge_fn':112,'edge_fp':111,'edge_tp':2391},'AUDITED_BASELINE_COUNTS_CHANGED')
 source_audits={r['stem']:r for r in audit['per_movie']}
 state=read(ROOT/'outputs/competitive_gap_closure/division_retention_aws/data_object_manifest.json')
 datasets={};coverage=[];baseline_rows=[]
 for i,stem in enumerate(COHORT):
  gtsha=ew.verify_gt(state,stem);gt=ew.load_geff(CACHE/'train'/(stem+'.geff'))
  nodes,edges=graphs[stem];row,g,rev=evaluation(nodes,edges,stem,gt)
  ew.verify_metric(row,expected[stem]['metric']);baseline_rows.append(row)
  global_match=ew.matching_map(g,rev)
  from pinned_metric import division_metrics as dm
  components=dm._gt_weak_component_ids(gt)
  gt_out={int(j):[int(v) for v in gt.successors(j)] for j in gt.node_ids()}
  inc,out=ev.structure(nodes,edges);ids,x=proposal_data[stem]
  windows=source_audits[stem]['divisions']['events']
  labels,event_ids=ev.labels_for_events(ids,inc,out,windows,global_match,gt_out,components)
  # Check at least one positive proposal per recovered existing TP when represented.
  for p in source_audits[stem]['divisions']['true_positive_fork_ids']:
   indices=np.flatnonzero((ids[:,0]==p)&(ids[:,4]==1))
   ev.require(len(indices)==1 and labels[indices[0]]==1,'EXISTING_TP_EVENT_LABEL_PARITY_FAILED')
  base_offset=i*1000
  event_map={int(w['gt_divider_id']):base_offset+j for j,w in enumerate(windows)}
  group_events=np.array([event_map.get(int(v),-1) for v in event_ids],dtype=np.int64)
  np_save(OUT/'training_labels'/f'{stem}.npz',labels=labels,event_ids=group_events)
  reached={int(v) for v in event_ids[(labels==1)&(ids[:,4]==0)]}
  coverage.append({'stem':stem,'embryo':stem.split('_')[0],'annotated_divisions':len(windows),
    'baseline_recovered':row['division_tp'],'missed_divisions_with_positive_new_proposal':sum(not w['recovered'] and int(w['gt_divider_id']) in reached for w in windows),
    'positive_proposals':int(np.sum(labels==1)),'known_negative_proposals':int(np.sum(labels==0)),
    'ignored_unassessable_proposals':int(np.sum(labels<0)),**proposal_counts[stem]})
  datasets[stem]={'ids':ids,'x':x,'y':labels,'event':group_events,'gtsha':gtsha}
  progress('sparse_training_label_construction',current_movie=stem,completed_movies=i+1,total_movies=5)
  del gt,g;gc.collect()
 save(OUT/'coverage.json',coverage)
 models={};newfits=0;reusedfits=0;gate=[]
 for heldout in ('44b6','6bba'):
  train=[s for s in COHORT if not s.startswith(heldout+'_')];test=[s for s in COHORT if s.startswith(heldout+'_')]
  ev.require(set(train).isdisjoint(test) and len({s.split('_')[0] for s in train})==1,'EMBRYO_LEAKAGE')
  x=np.concatenate([datasets[s]['x'] for s in train]);y=np.concatenate([datasets[s]['y'] for s in train]);events=np.concatenate([datasets[s]['event'] for s in train])
  npos=len(set(int(v) for v in events[y==1]));nneg=int(np.sum(y==0))
  gate.append({'heldout_embryo':heldout,'training_movies':train,'test_movies':test,'distinct_training_divisions':npos,'known_negative_proposals':nneg,
      'passed':npos>=ev.CONFIG.minimum_positive_events_per_training_embryo and nneg>=20})
  del x,y,events
 save(OUT/'training_readiness.json',gate)
 if not all(r['passed'] for r in gate):
  result={'status':'complete','decision':'stop_sparse_event_model_insufficient_distinct_training_divisions','model_fits':0,'new_fits':0,'reused_fits':0,
      'baseline':summary(baseline_rows),'candidate':None,'score_delta':None,'coverage':coverage,'readiness':gate,'measured_movies':COHORT,
      'approved_for_submission':False,'new_prediction_graphs':0,'proposal_reuse':proposal_reuse,
      'note':'Proposal population and label coverage saved. No scoring gain or model validation is claimed.'}
  save(OUT/'result.json',result);progress('complete',decision=result['decision'],completed_movies=5);return
 for fold in gate:
  heldout=fold['heldout_embryo'];train=fold['training_movies'];folder=OUT/'models'/heldout
  contract=file_contract({'training_inputs':{s:contracts[s] for s in train},'training_label_sha256':{s:sha(OUT/'training_labels'/f'{s}.npz') for s in train},'split':fold})
  if completed(folder,contract,['model.json']):models[heldout]=read(folder/'model.json');reusedfits+=1
  else:
   progress('fit_training_embryo_only',heldout_embryo=heldout,training_movies=train,completed_fits=newfits+reusedfits)
   x=np.concatenate([datasets[s]['x'] for s in train]);y=np.concatenate([datasets[s]['y'] for s in train]);events=np.concatenate([datasets[s]['event'] for s in train])
   model=ev.fit_model(x,y,events);model.update(heldout_embryo=heldout,training_movies=train,test_movies=fold['test_movies'],contract_sha256=contract)
   save(folder/'model.json',model);done(folder,contract,['model.json']);models[heldout]=model;newfits+=1;save(OUT/'fit_progress.json',{'new_fits':newfits,'reused_fits':reusedfits});del x,y,events
 model_audit=[{k:m[k] for k in ('heldout_embryo','training_movies','distinct_training_divisions','known_training_events','threshold','training_tp_above_threshold','training_fp_above_threshold','iterations')} for m in models.values()]
 save(OUT/'model_audit.json',model_audit)
 save(OUT/'fit_progress.json',{'new_fits':newfits,'reused_fits':reusedfits})
 # Produce EVERY held-out candidate graph before scoring any held-out candidate.
 transactions={};candidate_reuse=0
 for stem in COHORT:
  heldout=stem.split('_')[0];model=models[heldout];nodes,edges=graphs[stem];ids,x=proposal_data[stem];folder=OUT/'videos'/stem
  contract=file_contract({'proposal':contracts[stem],'model':sha(OUT/'models'/heldout/'model.json')})
  if completed(folder,contract,['graph.json.gz','transaction.json','scores.npz']):candidate_reuse+=1
  else:
   progress('heldout_inference_no_labels',current_movie=stem,model_trained_on=model['training_movies'])
   scores=ev.predict(model,x);candidate,transaction=ev.apply_events(nodes,edges,ids,scores,model['threshold'])
   graph_write(folder/'graph.json.gz',nodes,candidate);np_save(folder/'scores.npz',scores=scores)
   save(folder/'transaction.json',dict(transaction,contract_sha256=contract,heldout_embryo=heldout,training_movies=model['training_movies'],labels_consulted=False))
   done(folder,contract,['graph.json.gz','transaction.json','scores.npz'])
  transactions[stem]=read(folder/'transaction.json')
 save(OUT/'all_heldout_predictions_frozen_before_scoring.json',{'graphs':{s:sha(OUT/'videos'/s/'graph.json.gz') for s in COHORT},'cross_embryo_models':True})
 candidate_rows=[];metric_reuse=0
 for i,stem in enumerate(COHORT):
  folder=OUT/'videos'/stem;path=folder/'metric.json';donepath=folder/'metric_done.json'
  metric_contract=file_contract({'graph':sha(folder/'graph.json.gz'),'gt':datasets[stem]['gtsha'],'code':file_contract(code)})
  if donepath.is_file():
   d=read(donepath);ev.require(d['contract']==metric_contract and sha(path)==d['sha256'],'CANDIDATE_METRIC_CACHE_INVALID');row=read(path);metric_reuse+=1
  else:
   progress('exact_heldout_scoring',current_movie=stem,completed_movies=i,total_movies=5)
   nodes,edges=ew.graph_load(folder/'graph.json.gz');gt=ew.load_geff(CACHE/'train'/(stem+'.geff'))
   row,g,_=evaluation(nodes,edges,stem,gt);save(path,row);save(donepath,{'contract':metric_contract,'sha256':sha(path)});del gt,g;gc.collect()
  candidate_rows.append(row)
 base=summary(baseline_rows);candidate=summary(candidate_rows)
 embryos=[]
 for e in ('44b6','6bba'):
  b=summary([r for r in baseline_rows if r['embryo']==e]);c=summary([r for r in candidate_rows if r['embryo']==e])
  embryos.append({'embryo':e,'baseline':b,'candidate':c,'delta':c['score']-b['score']})
 interventions=sum(r['accepted_new_forks'] for r in transactions.values());delta=candidate['score']-base['score']
 promote=bool(interventions and delta>=.002 and candidate['adj_edge_jaccard']>=base['adj_edge_jaccard']-.0005 and all(e['delta']>=-1e-10 for e in embryos)
      and candidate['division_tp']>=base['division_tp'] and candidate['division_fp']<=base['division_fp'])
 result={'status':'complete','decision':'independent_confirmation_required' if promote else 'reject_fixed_competing_parent_event_model',
  'baseline':base,'candidate':candidate,'score_delta':delta,'embryos':embryos,'baseline_rows':baseline_rows,'candidate_rows':candidate_rows,
  'accepted_new_forks':interventions,'reassigned_children':sum(r['reassigned_children'] for r in transactions.values()),'transactions':transactions,
  'coverage':coverage,'models':model_audit,'new_fits':newfits,'reused_fits':reusedfits,'model_fits':newfits,'total_models':2,
  'new_prediction_graphs':5-candidate_reuse,'reused_prediction_graphs':candidate_reuse,'reused_metrics':metric_reuse,'proposal_reuse':proposal_reuse,
  'approved_for_submission':False,'measured_movies':COHORT,'elapsed_seconds':time.monotonic()-start,
  'limitations':['Only six annotated division events in two repeatedly used development embryos.','Embryo-disjoint fits do not undo prior researcher exposure or unknown pretrained overlap.',
     'Event scores are regularized ranking/classification scores, not calibrated probabilities.','New lightweight graph-feature method, not HOCT reproduction.'],
  'network_requests':0,'downloads':0,'gpu_inference_calls':0,'official_public_score_snapshot':.947,'leader_snapshot':.974,'fresh_scoring_query':False}
 save(OUT/'result.json',result)
 with (OUT/'comparison.csv').open('w',newline='') as f:
  fields=['stem','embryo','arm','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','adj_edge_jaccard']
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for arm,rr in [('frozen',baseline_rows),('learned_events',candidate_rows)]:
   for row in rr:writer.writerow(dict(row,arm=arm))
 progress('complete',decision=result['decision'],score_delta=delta,completed_movies=5,total_movies=5)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(OUT/'worker_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
