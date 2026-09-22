"""Five frozen-graph error budget: CPU-only, cached data, no model execution."""
from __future__ import annotations
from collections import Counter
import csv,gc,gzip,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
from diagnostics import require, partition_edges,division_reason,sensitivity_table,scalar_score

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('BIOHUB_PROJECT_ROOT','/home/sagemaker-user/biohub-cell-tracking-during-development')).resolve()
OUT=Path(os.environ.get('BIOHUB_ERROR_OUT',str(ROOT/'outputs/competitive_gap_closure/exact_error_budget'))).resolve()
OLD=ROOT/'outputs/competitive_gap_closure/temporal_reassignment'
CACHE=ROOT/'data/biohub_validation_cache'
SCALE=(1.625,.40625,.40625)

def log(s):print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),s,flush=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial')
    t.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def offline_guard():
    def block(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
            raise RuntimeError('OFFLINE_AUDIT_FORBIDS '+event)
    sys.addaudithook(block)

def graph_load(p):
    with gzip.open(p,'rt') as f:v=json.load(f)
    nodes={int(k):d for k,d in v['nodes'].items()};return nodes,v['edges']

def metric_graph(nodes,edges):
    import tracksdata as td
    import polars as pl
    g=td.graph.InMemoryGraph()
    for k in ('z','y','x'):g.add_node_attr_key(k,pl.Float64,0.)
    orig_to_id={}
    for i,n in sorted(nodes.items()):
        orig_to_id[i]=g.add_node(attrs={'t':int(n['t']),**{k:float(round(n[k])) for k in ('z','y','x')}})
    for e in edges:g.add_edge(orig_to_id[int(e['source_id'])],orig_to_id[int(e['target_id'])],{})
    return g,{int(v):int(k) for k,v in orig_to_id.items()}

def load_geff(p):
    import tracksdata as td
    v=td.graph.IndexedRXGraph.from_geff(p)
    return v[0] if isinstance(v,tuple) else v

def nodes_edges(g):
    nodes={int(r['node_id']):{'node_id':int(r['node_id']),'t':int(r['t']),**{k:float(r[k]) for k in ('z','y','x')}} for r in g.node_attrs().iter_rows(named=True)}
    edges=[{'source_id':int(e['source_id']),'target_id':int(e['target_id'])} for e in g.edge_attrs().iter_rows(named=True)]
    return nodes,edges

def matching_map(g,to_orig):
    import tracksdata as td
    K=td.DEFAULT_ATTR_KEYS
    return {to_orig[int(r[K.NODE_ID])]:int(r[K.MATCHED_NODE_ID]) for r in g.node_attrs(attr_keys=[K.NODE_ID,K.MATCHED_NODE_ID]).iter_rows(named=True)
            if r[K.MATCHED_NODE_ID] is not None and int(r[K.MATCHED_NODE_ID])>=0}

def find_value(v,key):
    if isinstance(v,dict):
        if key in v:return v[key]
        for x in v.values():
            q=find_value(x,key)
            if q is not None:return q
    if isinstance(v,list):
        for x in v:
            q=find_value(x,key)
            if q is not None:return q
    return None

def estimated_nodes(p):
    for name in ('zarr.json','.zattrs'):
        if (p/name).is_file():
            x=find_value(read(p/name),'estimated_number_of_nodes')
            if x is not None:
                x=float(x);require(math.isfinite(x) and x>0,'INVALID_NODE_ESTIMATE');return x
    raise RuntimeError('MISSING_NODE_ESTIMATE '+str(p))

def tree_fingerprint(p):
    rows=[]
    for f in sorted(p.rglob('*')):
        if f.is_file():
            require(not f.is_symlink(),'UNEXPECTED_SYMLINK');rows.append([str(f.relative_to(p)),f.stat().st_size,sha(f)])
    require(rows,'EMPTY_GRAPH_STORE')
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()

def verify_gt(state,stem):
    prefix=f'train/{stem}.geff';keys={k for k in state['entries'] if k.startswith(prefix+'/')}
    for p,a in state.get('plans',{}).get(stem,{}).items():
        if p.startswith(prefix+'/'):
            keys.add(p+'/zarr.json');keys.update(p+'/'+k for k in a['keys'])
    require(keys,'GT_MANIFEST_EMPTY')
    for key in sorted(keys):
        require(not Path(key).is_absolute() and '..' not in Path(key).parts,'UNSAFE_OBJECT_KEY')
        p=CACHE/key;require(p.resolve().is_relative_to(CACHE),'CACHE_PATH_ESCAPE')
        e=state['entries'].get(key);require(e is not None,'GT_OBJECT_RECEIPT_MISSING '+key)
        if e.get('status')=='complete':
            require(p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'GT_CACHE_INTEGRITY '+key)
        else:require(e.get('meaning')=='zarr_chunk_absent_fill_value' and not p.exists(),'GT_CHUNK_UNVERIFIED')
    return hashlib.sha256(json.dumps({k:state['entries'][k] for k in sorted(keys)},sort_keys=True).encode()).hexdigest()

def verify_metric(row,expected):
    for k in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'):
        require(int(row[k])==int(expected[k]),'CONTROL_METRIC_COUNT_MISMATCH '+k)
    for k in ('adj_edge_jaccard','edge_jaccard','node_recall','total_node_ratio'):
        require(math.isclose(row[k],expected[k],abs_tol=1e-10,rel_tol=1e-10),'CONTROL_METRIC_FLOAT_MISMATCH '+k)

def detailed_divisions(g,gt,to_orig,expected):
    from pinned_metric import division_metrics as dm
    import tracksdata as td
    K=td.DEFAULT_ATTR_KEYS
    scored=dm.score_divisions(g,gt,SCALE,7.)
    require(sum(scored.scores.values())==expected['division_tp'],'DIVISION_TP_MISMATCH')
    require(len(scored.scores)-sum(scored.scores.values())==expected['division_fn'],'DIVISION_FN_MISMATCH')
    require(len(scored.fp_forks)==expected['division_fp'],'DIVISION_FP_MISMATCH')
    windows=dm.match_divisions(g,gt,SCALE,7.);subgraphs=dm.extract_divisions(gt)
    _,cross,malformed=dm._pred_division_fork_sets(g,gt,SCALE,7.)
    invalid=cross|malformed;events=[]
    for divider,recovered in sorted(scored.scores.items()):
        pred=windows[divider];sub=subgraphs[divider]
        m={int(r[K.NODE_ID]):int(r[K.MATCHED_NODE_ID]) for r in dm._matched_node_attrs(pred).iter_rows(named=True)}
        parents={int(divider),*[int(x) for x in sub.predecessors(divider)]}
        parent_pred={p for p,q in m.items() if q in parents}
        daughter_sets=[]
        for child in sub.successors(divider):
            lineage={int(child),*[int(x) for x in sub.successors(child)]}
            daughter_sets.append({p for p,q in m.items() if q in lineage})
        local=parent_pred|{int(j) for p in parent_pred for j in pred.successors(p)}
        forks={p for p in local if pred.out_degree(p)>=2}
        valid={p for p in forks-invalid if dm._is_strongly_connected_division(pred,p,parent_pred,daughter_sets)}
        category=division_reason(recovered=bool(recovered),parent_support=parent_pred,daughter_support=daughter_sets,
                                 local_forks=forks,valid_topology_forks=valid,invalid_forks=invalid)
        events.append({'gt_divider_id':int(divider),'recovered':bool(recovered),'category':category,
            'parent_side_prediction_ids':sorted(to_orig[p] for p in parent_pred),
            'daughter_side_prediction_ids':[sorted(to_orig[p] for p in s) for s in daughter_sets],
            'local_fork_prediction_ids':sorted(to_orig[p] for p in forks),
            'locally_valid_fork_prediction_ids':sorted(to_orig[p] for p in valid),
            'note':'Event identity uses exact window-specific matching; do not replace with global edge matching.'})
    return {'events':events,'false_positive_fork_ids':sorted(to_orig[int(x)] for x in scored.fp_forks),
            'true_positive_fork_ids':sorted(to_orig[int(x)] for x in scored.tp_forks),
            'miss_categories':dict(Counter(e['category'] for e in events if not e['recovered']))}

def analyze_movie(stem,expected,raw_path):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from pinned_metric import metrics
    K=td.DEFAULT_ATTR_KEYS
    source=OLD/'videos'/stem/'frozen'/'graph.json.gz';nodes,edges=graph_load(source)
    g,to_orig=metric_graph(nodes,edges);gt=load_geff(CACHE/'train'/(stem+'.geff'));gn,ge=nodes_edges(gt)
    er=metrics.evaluate(g,gt,scale=SCALE,max_distance=7.)
    row=metrics.per_sample_metrics(er,estimated_nodes(CACHE/'train'/(stem+'.geff')),metrics.node_recall(g,gt))
    verify_metric(row,expected);row.update(stem=stem,embryo=stem.split('_')[0])
    evaluation=[{'source_id':to_orig[int(e[K.EDGE_SOURCE])],'target_id':to_orig[int(e[K.EDGE_TARGET])],
                 'matched':bool(e[K.MATCHED_EDGE_MASK]),'valid':bool(e['pred_valid'])}
                for e in metrics._evaluate_matched_graph(g,gt).iter_rows(named=True)]
    m=matching_map(g,to_orig)
    raw=load_geff(raw_path);rn,re=nodes_edges(raw);rg,ri=metric_graph(rn,re)
    from tracksdata.options import set_options
    set_options(show_progress=False)
    import warnings
    from scipy.sparse import SparseEfficiencyWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=SparseEfficiencyWarning)
        rg.match(gt,matching=DistanceMatching(max_distance=7.,scale=SCALE))
    rm=matching_map(rg,ri)
    attribution=partition_edges(nodes,edges,gn,ge,m,evaluation,row,rm,rn)
    require(attribution['matched_gt_nodes']/len(gn)==row['node_recall'],'NODE_RECALL_ATTRIBUTION_MISMATCH')
    divisions=detailed_divisions(g,gt,to_orig,row)
    # This is an analysis result, not a new prediction; no graph is written.
    result={'status':'complete','stem':stem,'metric':row,'edges':attribution,'divisions':divisions,
            'raw_final_identity':{'shared_prediction_ids':len(set(nodes)&set(rn)),
              'shared_ids_time_mismatch':sum(int(nodes[i]['t'])!=int(rn[i]['t']) for i in set(nodes)&set(rn)),
              'raw_nodes_removed':len(set(rn)-set(nodes)),'synthetic_final_nodes':len(set(nodes)-set(rn))},
            'labels_used':'diagnostics_only','prediction_changes':0,'approved_for_submission':False}
    require(result['raw_final_identity']['shared_ids_time_mismatch']==0,'RAW_FINAL_IDENTITY_MISMATCH')
    return result

def main():
    offline_guard();os.environ['CUDA_VISIBLE_DEVICES']='';OUT.mkdir(parents=True,exist_ok=True);beg=time.monotonic()
    expected=read(HERE/'expected_baselines.json');prior=read(OLD/'result.json')
    require(prior['status']=='complete' and prior['accepted_swaps']==0,'PRIOR_DECISION_MISSING')
    require(prior['decision']=='stop_configuration_no_qualified_identity_exchanges','PRIOR_DECISION_CHANGED')
    require(set(prior['measured_movies'])==set(expected),'COHORT_CHANGED')
    state=read(ROOT/'outputs/competitive_gap_closure/division_retention_aws/data_object_manifest.json')
    support=read(ROOT/'outputs/competitive_gap_closure/division_retention_cached/result.json')
    paths={x['stem']:x for x in support['proposal_audit']}
    codehash=hashlib.sha256(''.join(sha(HERE/n) for n in ['diagnostics.py','error_worker.py','expected_baselines.json','pinned_metric/metrics.py','pinned_metric/division_metrics.py']).encode()).hexdigest()
    records=[];reused=0
    for i,stem in enumerate(prior['measured_movies']):
        gp=OLD/'videos'/stem/'frozen'/'graph.json.gz';rp=gp.with_name('receipt.json')
        receipt=read(rp)
        require(receipt['status']=='complete' and receipt.get('parity_passed') is True,'BASELINE_INCOMPLETE')
        require(sha(gp)==expected[stem]['graph_sha256']==receipt['graph_sha256'],'BASELINE_GRAPH_CHANGED')
        gtsha=verify_gt(state,stem);raw_path=Path(paths[stem]['raw_graph_path']).resolve()
        parent=ROOT/'outputs/public_frontier_reproduction/exact_0947/our_exact_output'
        require(raw_path.is_relative_to(parent),'RAW_OUTSIDE_REFERENCE')
        rawsha=tree_fingerprint(raw_path);require(rawsha==paths[stem]['raw_graph_tree_sha256'],'RAW_GRAPH_CHANGED')
        contract=hashlib.sha256(json.dumps([codehash,expected[stem],gtsha,rawsha],sort_keys=True).encode()).hexdigest()
        dest=OUT/'movies'/stem;done=dest/'done.json';result_path=dest/'diagnostic_only.json'
        save(OUT/'progress_0.json',{'phase':'exact_error_attribution','completed_movies':len(records),'total_movies':5,'current_movie':stem})
        if done.is_file():
            d=read(done);require(d['contract']==contract,'DIAGNOSTIC_CACHE_CONTRACT_CHANGED')
            require(result_path.is_file() and sha(result_path)==d['sha256'],'DIAGNOSTIC_CACHE_CORRUPT')
            result=read(result_path);reused+=1;log('AUDIT_REUSED '+stem)
        else:
            log(f'AUDIT_START {i+1}/5 {stem}; CPU graph matching only')
            result=analyze_movie(stem,expected[stem]['metric'],raw_path);save(result_path,result)
            save(done,{'status':'complete','contract':contract,'sha256':sha(result_path)})
            log(f'AUDIT_COMPLETE {stem} FN={result["edges"]["edge_fn"]} FP={result["edges"]["edge_fp"]}')
        require(result['status']=='complete' and result['stem']==stem,'DIAGNOSTIC_IDENTITY_MISMATCH');records.append(result);gc.collect()
    from pinned_metric import metrics
    summary=metrics.summarise([x['metric'] for x in records]);require(summary['n']==summary['n_adj']==5,'METRIC_DROPPED_ROWS')
    require(math.isclose(summary['score'],prior['summaries']['frozen']['score'],abs_tol=1e-10),'AGGREGATE_CONTROL_MISMATCH')
    sensitivity=sensitivity_table(records)
    combined={k:dict(sum((Counter(m['edges'][k]) for m in records),Counter())) for k in ['fn_categories','fp_categories','unmatched_gt_node_categories']}
    divcats=dict(sum((Counter(m['divisions']['miss_categories']) for m in records),Counter()))
    ranking=[{'diagnostic':x['scenario'],'local_count_sensitivity':x['delta_from_local_baseline'],'events':x['event_count'],
              'next_action':'design_label_free_method_and_independent_validation','achieved_gain':False} for x in sensitivity]
    result={'status':'complete','measured_movies':[x['stem'] for x in records],'baseline_summary':summary,
            'counts':{k:sum(m['metric'][k] for m in records) for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']},
            **combined,'missed_division_categories':divcats,'count_sensitivities':sensitivity,'ranked_research_targets':ranking,
            'per_movie':records,'reused_movies':reused,'new_movie_audits':5-reused,'elapsed_seconds':time.monotonic()-beg,
            'decision':'design_next_method_from_exact_error_attribution_not_another_blind_threshold_change',
            'prediction_changes':0,'model_fits':0,'network_requests':0,'new_downloads':0,'gpu_inference_calls':0,
            'kaggle_actions':0,'approved_for_submission':False,'official_public_score_snapshot':.947,'leader_snapshot':.974,
            'fresh_leaderboard_query':False,
            'limitations':['Previously used five-movie development cohort, two embryos; pretrained independence unverified.',
                'Sensitivity scores use annotation-dependent hypothetical count changes, not predicted or achieved gains.',
                'Unmatched prediction nodes are not ground-truth false positives.','Only graph and metadata files are read; no microscopy images loaded.']}
    save(OUT/'result.json',result)
    for filename,field in [('diagnostic_false_negatives.csv','false_negative_edges'),('diagnostic_false_positives.csv','false_positive_edges'),('diagnostic_unmatched_gt_nodes.csv','unmatched_gt_nodes')]:
        rows=[dict(stem=m['stem'],**r) for m in records for r in m['edges'][field]]
        if rows:
            with (OUT/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'progress_0.json',{'phase':'complete','completed_movies':5,'total_movies':5,'decision':result['decision']})
    log('EXACT_ERROR_BUDGET_COMPLETE '+json.dumps(result['counts']))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'worker_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
