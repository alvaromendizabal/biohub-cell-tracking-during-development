"""Fixed-node continuation reranking: temporal geometry, image profiles, assignment.

No label-derived identifiers or absolute location/time enter model features.
Original divisions and their two-hop neighborhoods are locked. A balanced assignment
permutes only the targets of existing unary links, preserving all node degrees.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from collections import defaultdict
import math,time,signal
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from division_events import structure,motion,require,SCALE

@dataclass(frozen=True)
class Config:
    radius_um:float=12.0
    alternatives:int=6
    context_steps:int=3
    division_protect_hops:int=2
    incumbent_bonus:float=math.log(2.0)
    minimum_new_score:float=0.5
    max_pairs_per_movie:int=400000
    max_frame_links:int=1800
    max_fit_seconds:int=90
    max_trees:int=100
    minimum_training_positive:int=100
    minimum_training_negative:int=200
CONFIG=Config()
GEO_NAMES=('distance_um','abs_dz_um','abs_dy_um','abs_dx_um','source_speed','target_speed',
 'forward_error','backward_error','velocity_disagreement','source_history','target_future',
 'source_motion_noise','target_motion_noise','source_nearest_neighbor','target_nearest_neighbor',
 'source_neighbors_10um','target_neighbors_10um','forward_alignment','backward_alignment','velocity_alignment')
APP_NAMES=('patch_correlation','patch_absolute_difference','patch_squared_difference',
 'local_intensity_difference','peak_intensity_difference','contrast_log_ratio',
 'radial_size_difference','axial_size_difference','boundary_support_min')
OFFSETS=np.stack(np.meshgrid(*[np.linspace(-3.,3.,5)]*3,indexing='ij'),axis=-1).reshape(-1,3)

def cosine(a,b):return float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-9))
def protected_nodes(nodes,edges,hops=2):
    inc,out=structure(nodes,edges);locked={p for p,v in out.items() if len(v)==2};front=set(locked)
    for _ in range(hops):
        nxt=set()
        for p in front:
            nxt.update(out.get(p,[]))
            if p in inc:nxt.add(inc[p])
        front=nxt-locked;locked.update(nxt)
    return locked

def candidates(nodes,edges,config=CONFIG,heartbeat=None):
    inc,out=structure(nodes,edges);locked=protected_nodes(nodes,edges,config.division_protect_hops)
    points={i:np.asarray([n[k] for k in ('z','y','x')],float)*SCALE for i,n in nodes.items()}
    back=motion(points,nodes,inc,out,False,config.context_steps);future=motion(points,nodes,inc,out,True,config.context_steps)
    frames=defaultdict(list)
    for i,n in nodes.items():frames[int(n['t'])].append(i)
    density={}
    for t,ids in frames.items():
        ids=sorted(ids);xyz=np.asarray([points[i] for i in ids]);tree=cKDTree(xyz)
        near=tree.query(xyz,k=min(2,len(ids)))[0]
        for j,i in enumerate(ids):
            nn=float(near[j,1]) if len(ids)>1 else 0.
            density[i]=(nn,len(tree.query_ball_point(xyz[j],10.))-1)
    ids_out=[];features=[];incumbent=[];eligible=0
    for t in sorted(frames):
        sources=sorted(p for p in frames[t] if p not in locked and len(out.get(p,[]))==1 and out[p][0] not in locked)
        targets=sorted(out[p][0] for p in sources)
        if not sources:continue
        require(len(sources)==len(set(targets)) and len(sources)<=config.max_frame_links,'FRAME_ASSIGNMENT_BOUND_OR_MERGE')
        eligible+=len(sources);tree=cKDTree(np.asarray([points[b] for b in targets]))
        for p in sources:
            distances,indices=tree.query(points[p],k=min(len(targets),config.alternatives),distance_upper_bound=config.radius_um)
            chosen={out[p][0]}
            chosen.update(targets[int(j)] for d,j in zip(np.atleast_1d(distances),np.atleast_1d(indices)) if np.isfinite(d) and int(j)<len(targets))
            for b in sorted(chosen):
                delta=points[b]-points[p];vp,hp,npast=back[p];vb,hb,nfuture=future[b]
                row=[float(np.linalg.norm(delta)),*np.abs(delta).tolist(),float(np.linalg.norm(vp)),float(np.linalg.norm(vb)),
                    float(np.linalg.norm(delta-vp)),float(np.linalg.norm(delta-vb)),float(np.linalg.norm(vp-vb)),hp,hb,npast,nfuture,
                    *density[p],*density[b],cosine(delta,vp),cosine(delta,vb),cosine(vp,vb)]
                # density was interleaved above; reorder to the declared column schema.
                row[13:17]=[density[p][0],density[b][0],density[p][1],density[b][1]]
                require(len(row)==len(GEO_NAMES) and np.isfinite(row).all(),'GEOMETRY_FEATURE_INVALID')
                ids_out.append((p,b,t));features.append(row);incumbent.append(b==out[p][0])
        require(len(ids_out)<=config.max_pairs_per_movie,'CANDIDATE_PAIR_LIMIT')
        if heartbeat:heartbeat(t,len(ids_out))
    return (np.asarray(ids_out,np.int64).reshape(-1,3),np.asarray(features,np.float32).reshape(-1,len(GEO_NAMES)),
            np.asarray(incumbent,bool),{'eligible_existing_links':eligible,'candidate_pairs':len(ids_out),'locked_nodes':len(locked)})

def frame_profiles(frame,coords,scale=SCALE):
    """125-point, fixed physical stencil; no full-volume resampling or fit.

    Frame intensity normalization is inference-time, label-free, per-image.
    Points beyond the field of view are clipped and explicitly flagged.
    """
    frame=np.asarray(frame);coords=np.asarray(coords,float).reshape(-1,3);scale=np.asarray(scale,float)
    require(frame.ndim==3 and frame.size and frame.nbytes<=256*(1<<20),'IMAGE_FRAME_BOUND')
    require(scale.shape==(3,) and np.isfinite(scale).all() and (scale>0).all(),'INVALID_VOXEL_SCALE')
    require(np.isfinite(coords).all(),'INVALID_SAMPLE_COORDINATE')
    require(np.isfinite(frame).all(),'NONFINITE_IMAGE')
    sampled=frame.reshape(-1)[::max(1,frame.size//65536)].astype(float)
    qlo,qhi=np.percentile(sampled,[1,99]);den=max(float(qhi-qlo),1.)
    positions=coords[:,None,:]+OFFSETS[None,:,:]/scale
    valid=((positions>=0)&(positions<=np.asarray(frame.shape)-1)).all(axis=2).mean(axis=1)
    positions=np.clip(positions,0,np.asarray(frame.shape)-1)
    values=map_coordinates(frame,positions.reshape(-1,3).T,order=1,mode='nearest',prefilter=False,output=np.float32).reshape(len(coords),len(OFFSETS))
    values=np.clip((values-float(qlo))/den,-5,20)
    center=values.mean(axis=1);contrast=values.std(axis=1);peak=values.max(axis=1)
    centered=values-center[:,None];norm=np.linalg.norm(centered,axis=1)
    profiles=centered/np.maximum(norm[:,None],1e-6)
    mass=np.maximum(values-np.quantile(values,.2,axis=1)[:,None],0.)
    sizes=np.sqrt((mass@np.square(OFFSETS))/np.maximum(mass.sum(axis=1)[:,None],1e-6))
    stats=np.column_stack([center,peak,contrast,np.linalg.norm(sizes,axis=1),sizes[:,0],valid]).astype(np.float32)
    require(np.isfinite(profiles).all() and np.isfinite(stats).all(),'NONFINITE_IMAGE_DESCRIPTOR')
    return profiles.astype(np.float32),stats

def appearance_features(pairs,node_ids,profiles,stats):
    node_ids=np.asarray(node_ids);require(len(node_ids)==len(set(node_ids.tolist())),'DUPLICATE_APPEARANCE_ID')
    order=np.argsort(node_ids);node_ids=node_ids[order];profiles=profiles[order];stats=stats[order]
    a=np.searchsorted(node_ids,pairs[:,0]);b=np.searchsorted(node_ids,pairs[:,1])
    require(np.all(a<len(node_ids)) and np.all(b<len(node_ids)),'APPEARANCE_ENDPOINT_MISSING')
    require(np.array_equal(node_ids[a],pairs[:,0]) and np.array_equal(node_ids[b],pairs[:,1]),'APPEARANCE_ID_MISMATCH')
    pa,pb=profiles[a],profiles[b];sa,sb=stats[a],stats[b]
    out=np.column_stack([(pa*pb).sum(axis=1),np.abs(pa-pb).mean(axis=1),np.square(pa-pb).sum(axis=1),
      np.abs(sa[:,0]-sb[:,0]),np.abs(sa[:,1]-sb[:,1]),np.abs(np.log((sa[:,2]+.01)/(sb[:,2]+.01))),
      np.abs(sa[:,3]-sb[:,3]),np.abs(sa[:,4]-sb[:,4]),np.minimum(sa[:,5],sb[:,5])]).astype(np.float32)
    require(out.shape==(len(pairs),len(APP_NAMES)) and np.isfinite(out).all(),'BAD_APPEARANCE_FEATURES')
    return out

def sparse_labels(pairs,matched,gt_edges):
    gt_edges=set(tuple(map(int,e)) for e in gt_edges);active_out={s for s,t in gt_edges};active_in={t for s,t in gt_edges}
    y=np.full(len(pairs),-1,np.int8)
    for i,(a,b,*_) in enumerate(pairs):
        s=matched.get(int(a));t=matched.get(int(b))
        if s is not None and t is not None and (s,t) in gt_edges:y[i]=1
        elif s in active_out or t in active_in:y[i]=0
    return y

def train(x,y,config=CONFIG):
    """No random held-out split and no class balancing that turns scores into priors."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    x=np.asarray(x,np.float32);y=np.asarray(y);known=y>=0;x=x[known];y=y[known]
    require(len(x) and np.isfinite(x).all(),'INVALID_TRAINING_FEATURES')
    require(np.sum(y==1)>=config.minimum_training_positive and np.sum(y==0)>=config.minimum_training_negative,'INSUFFICIENT_EDGE_SUPERVISION')
    model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.05,max_iter=config.max_trees,max_leaf_nodes=7,
            max_depth=3,min_samples_leaf=30,l2_regularization=1.,max_bins=63,early_stopping=False,random_state=20260922)
    start=time.monotonic()
    previous=signal.getsignal(signal.SIGALRM)
    def expired(sig,frame):raise TimeoutError('MODEL_FIT_EXCEEDED_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,config.max_fit_seconds)
    try:model.fit(x,y)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    require(np.isfinite(model.predict_proba(x[:min(1024,len(x))])).all(),'NONFINITE_FIT')
    return model,{'positive_edges':int(np.sum(y==1)),'negative_edges':int(np.sum(y==0)),
             'boosting_iterations':int(model.n_iter_),'seconds':time.monotonic()-start,
             'early_stopping':False,'class_weight':None,'feature_count':x.shape[1],
             'scores_are_not_calibrated':True,'fit_definition':'one fixed model and one fit call; 100 boosting iterations; 90-second signal bound'}

def decode(nodes,edges,pairs,scores,incumbent,config=CONFIG):
    """Solve complete frame assignments, then accept only entire safe cycles.

    Alternative links not generated above cannot be selected. Every cycle strictly
    improves aggregate learned log odds after the fixed incumbent bonus.
    """
    inc,out=structure(nodes,edges);locked=protected_nodes(nodes,edges,config.division_protect_hops)
    require(len(scores)==len(pairs)==len(incumbent) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'INVALID_SCORES')
    require(len({tuple(map(int,row[:2])) for row in pairs})==len(pairs),'DUPLICATE_CANDIDATE')
    original={(int(e['source_id']),int(e['target_id'])):e for e in edges};result={k:dict(v) for k,v in original.items()};changes=[];cycles=0
    for t in sorted(set(pairs[:,2].tolist())):
        idx=np.flatnonzero(pairs[:,2]==t);sources=sorted(set(pairs[idx,0].tolist()));targets=sorted(set(pairs[idx[incumbent[idx]],1].tolist()))
        require(len(sources)==len(targets)<=config.max_frame_links,'UNBALANCED_ASSIGNMENT')
        ri={p:i for i,p in enumerate(sources)};ci={b:i for i,b in enumerate(targets)};lookup={}
        weights=np.full((len(sources),len(targets)),-1e6,dtype=float)
        for k in idx:
            p,b,tt=map(int,pairs[k]);require(p not in locked and b not in locked,'PROTECTED_NODE_IN_CANDIDATES')
            require(nodes[p]['t']==tt and nodes[b]['t']==tt+1 and b in ci,'INVALID_CANDIDATE_SCHEMA')
            require(bool(incumbent[k])==((p,b) in original),'INCUMBENT_MASK_MISMATCH')
            l=float(logit(np.clip(scores[k],1e-5,1-1e-5)));weights[ri[p],ci[b]]=l+(config.incumbent_bonus if incumbent[k] else 0.)
            lookup[(p,b)]=(float(scores[k]),l)
        require(all((p,out[p][0]) in lookup for p in sources),'INCUMBENT_MISSING')
        row,col=linear_sum_assignment(weights,maximize=True)
        require(len(row)==len(sources) and all(weights[i,j]>-1e5 for i,j in zip(row,col)),'INCOMPLETE_ASSIGNMENT')
        new={sources[i]:targets[j] for i,j in zip(row,col)}
        changed={p for p in sources if new[p]!=out[p][0]};seen=set()
        for start in sorted(changed):
            if start in seen:continue
            component=[];p=start
            while p not in seen:
                require(p in changed,'ASSIGNMENT_NOT_A_CYCLE');seen.add(p);component.append(p);p=inc[new[p]]
            require(p==start,'INTERSECTING_ASSIGNMENT_CYCLES')
            gain=sum(lookup[(p,new[p])][1]-lookup[(p,out[p][0])][1] for p in component)
            if gain<=len(component)*config.incumbent_bonus+1e-10:continue
            if any(lookup[(p,new[p])][0]<config.minimum_new_score for p in component):continue
            cycles+=1
            for p in component:
                oldb=out[p][0];b=new[p];del result[(p,oldb)];result[(p,b)]={'source_id':p,'target_id':b}
                changes.append({'source_id':p,'old_target_id':oldb,'new_target_id':b,'new_score':lookup[(p,b)][0],
                                'old_score':lookup[(p,oldb)][0],'cycle':cycles,'cycle_gain':gain})
    new=[result[k] for k in sorted(result)];newinc,newout=structure(nodes,new)
    require(len(new)==len(edges),'EDGE_COUNT_CHANGED')
    require(set(newinc)==set(inc) and {p:len(v) for p,v in newout.items()}=={p:len(v) for p,v in out.items()},'NODE_DEGREE_CHANGED')
    for (p,b) in original:
        if p in locked or b in locked:require((p,b) in result,'DIVISION_NEIGHBORHOOD_CHANGED')
    return new,{'accepted_cycles':cycles,'changed_edges':len(changes),'changes':changes,'labels_consulted':False,
                'nodes_changed':0,'coordinates_changed':0,'edge_count_preserved':True,'node_degrees_preserved':True,'protected_nodes':len(locked)}
