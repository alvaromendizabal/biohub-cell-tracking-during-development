"""Round 2: eight association/division-context families, 16 columns each.

Inputs are observed images and fixed predicted candidates. No target labels are
accepted. Temporal history is an uncertain, mutually-nearest candidate at t-1,
not a ground-truth track. Target frame t+1 is already observed for linking.
No frame after the target is used. Division candidates remain hypotheses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .feature_math import (patch, normalize, moments, ratio, logratio, cosine, ncc,
    histogram_similarity, finite_or_nan, points_anisotropy)


def world(n):return np.array([n['z_um'],n['y_um'],n['x_um']],float)


def make_context(images,frames,offsets,spacing,nodes,round1):
    normal={int(t):normalize(images[i])[0] for i,t in enumerate(frames)}
    raw={int(t):images[i] for i,t in enumerate(frames)};result={}
    r1=round1.set_index('node_id',verify_integrity=True)
    for _,row in nodes.iterrows():
        idx=int(row.node_id);t=int(row.t);p=patch(raw[t],normal[t],row[['z','y','x']].to_numpy(int)-offsets,spacing)
        result[idx]={'row':row.to_dict(),'patch':p,'moments':moments(p,5.),'r1':r1.loc[idx].to_dict()}
    return result


def nearest(points, p, ids):
    if len(points)==0:return None,float('nan'),float('nan')
    d=np.linalg.norm(points-p,axis=1);order=np.lexsort((np.asarray(ids),d))
    gap=float(d[order[1]]-d[order[0]]) if len(order)>1 else np.nan
    return int(ids[order[0]]),float(d[order[0]]),gap


def pair_geometry(source,target):
    s,t=source['row'],target['row'];d=world(t)-world(s);norm=float(np.linalg.norm(d))
    cs,ct=source['moments'][2],target['moments'][2];es,et=source['moments'][3],target['moments'][3]
    def maha(c):
        return float(np.sqrt(max(0.,d@np.linalg.solve(c+np.eye(3)*1e-3,d)))) if np.isfinite(c).all() else np.nan
    vals=[*d,norm,np.linalg.norm(d[1:]),ratio(abs(d[0]),norm),*[ratio(v,norm) for v in d],
          maha(cs),maha(ct),ratio(norm,np.sqrt(es.sum())),ratio(norm,np.sqrt(et.sum())),
          s['roi_boundary_distance_um'],t['roi_boundary_distance_um'],t['roi_boundary_distance_um']-s['roi_boundary_distance_um']]
    keys=['dz','dy','dx','distance','xy_distance','abs_z_fraction','unit_z','unit_y','unit_x','source_mahalanobis','target_mahalanobis','source_size_distance','target_size_distance','source_border_margin','target_border_margin','border_change']
    return dict(zip(keys,vals))


def correspondence(source,target):
    a,b=source['patch'],target['patch'];out={}
    for r in [3,5]:
        mask=a.radius<=r;x=a.normal[mask];y=b.normal[mask]
        ga=np.gradient(a.normal,*a.spacing);gb=np.gradient(b.normal,*b.spacing)
        # Gradient components use physical voxel spacing at both times.
        gx=np.stack(ga,axis=-1)[mask].ravel();gy=np.stack(gb,axis=-1)[mask].ravel()
        out.update({f'r{r}_ncc':ncc(x,y),f'r{r}_rmse':float(np.sqrt(np.mean((x-y)**2))),
                    f'r{r}_gradient_ncc':ncc(gx,gy),f'r{r}_cosine':cosine(x,y)})
    sr,tr=source['r1'],target['r1'];profile=lambda row:np.array([row[f'r1_radial__ring{i}_mean'] for i in range(4)])
    js,inter=histogram_similarity(a.normal,b.normal)
    out.update(radial_cosine=cosine(profile(sr),profile(tr)),hist_js=js,hist_intersection=inter,
      log_mean_ratio=logratio(float(b.raw.mean()),float(a.raw.mean())),log_std_ratio=logratio(float(b.raw.std()),float(a.raw.std())),
      robust_mean_change=ratio(float(b.raw.mean()-a.raw.mean()),max(a.noise,b.noise,1.)),
      valid_voxel_fraction=float(np.mean(np.isfinite(a.raw)&np.isfinite(b.raw))),absolute_change_p90=float(np.quantile(abs(a.normal-b.normal),.9)))
    return out


def competition(pair,pairs):
    outgoing=pairs.loc[pairs.source_id==pair.source_id].sort_values(['distance_um','target_id'],kind='stable')
    incoming=pairs.loc[pairs.target_id==pair.target_id].sort_values(['distance_um','source_id'],kind='stable')
    value=float(pair.distance_um); out={}
    for side,table,key,ident in [('out',outgoing,'target_id',pair.target_id),('in',incoming,'source_id',pair.source_id)]:
        values=table.distance_um.to_numpy(float);other=table.loc[table[key]!=ident,'distance_um'].to_numpy(float)
        position=np.flatnonzero(table[key].to_numpy()==ident)
        if len(position)!=1:raise ValueError('Pair identity is missing or duplicated in competitors.')
        out[side+'_rank']=int(position[0]+1)
        out[side+'_other_gap']=float(other.min()-value) if len(other) else np.nan
        out[side+'_zscore']=ratio(value-values.mean(),values.std())
        out[side+'_percentile']=float(np.mean(values<=value))
        for scale in [2,5]:
            weight=np.exp(-(values-values.min())/scale)
            out[f'{side}_prob_{scale}']=float(weight[position[0]]/weight.sum())
        out[side+'_count']=len(values)
    out['mutual_nearest']=float(out['out_rank']==1 and out['in_rank']==1)
    v=outgoing.distance_um.to_numpy(float);out['next_best_gap']=float(v[1]-v[0]) if len(v)>1 else np.nan
    return out


def history(source,target,context,frames,max_distance=14.):
    keys=['available','previous_dz','previous_dy','previous_dx','previous_speed','prediction_residual','turn_cosine','speed_change','speed_ratio','previous_log_intensity_change','current_log_intensity_change','intensity_trend_error','previous_scale_change','current_scale_change','previous_neighbor_gap','previous_mutual']
    out=dict.fromkeys(keys,np.nan);out['available']=0.;out['previous_mutual']=0.
    s,t=source['row'],target['row'];time=int(s['t']);previous=frames.get(time-1)
    if previous is None or len(previous)==0:return out
    candidates=previous[['z_um','y_um','x_um']].to_numpy(float)
    pid,d,gap=nearest(candidates,world(s),previous.node_id.to_numpy())
    if pid is None or d>max_distance:return out
    p=context[pid]['row'];now=frames[time]
    reverse,_,_=nearest(now[['z_um','y_um','x_um']].to_numpy(float),world(p),now.node_id.to_numpy())
    if reverse!=int(s['node_id']):return out
    prev=world(s)-world(p);current=world(t)-world(s);speed=np.linalg.norm(prev);v=np.linalg.norm(current)
    before=logratio(s['core_mean_raw'],p['core_mean_raw']);after=logratio(t['core_mean_raw'],s['core_mean_raw'])
    values=[1.,*prev,speed,np.linalg.norm(current-prev),cosine(prev,current),v-speed,ratio(v,speed),before,after,after-before,
            s['best_sigma_um']-p['best_sigma_um'],t['best_sigma_um']-s['best_sigma_um'],gap,1.]
    return dict(zip(keys,values))


def neighborhood(source,target,frames):
    s,t=source['row'],target['row'];current=frames[int(s['t'])];following=frames[int(t['t'])]
    sv=current.loc[current.node_id!=s['node_id']].copy();tv=following.loc[following.node_id!=t['node_id']].copy()
    sv['_d']=np.linalg.norm(sv[['z_um','y_um','x_um']].to_numpy(float)-world(s),axis=1)
    tv['_d']=np.linalg.norm(tv[['z_um','y_um','x_um']].to_numpy(float)-world(t),axis=1)
    sv=sv.sort_values(['_d','node_id']);tv=tv.sort_values(['_d','node_id']);out={}
    for k in [3,5]:
        sn=sv.head(k);tn=tv.head(k);motion=[];mutual=[]
        for _,n in sn.iterrows():
            tid,d,_=nearest(following[['z_um','y_um','x_um']].to_numpy(float),world(n),following.node_id.to_numpy())
            if tid is None or d>14.:continue
            match=following.loc[following.node_id==tid].iloc[0]
            # Do not use the proposed target as a neighbor's correspondence.
            if int(tid)==int(t['node_id']):continue
            reverse,_,_=nearest(current[['z_um','y_um','x_um']].to_numpy(float),world(match),current.node_id.to_numpy())
            motion.append(world(match)-world(n));mutual.append(reverse==int(n.node_id))
        med=np.median(np.asarray(motion),axis=0) if motion else np.full(3,np.nan)
        moving=world(t)-world(s);count=min(len(sn),len(tn))
        profile=float(np.mean(abs(sn['_d'].to_numpy()[:count]-tn['_d'].to_numpy()[:count]))) if count else np.nan
        spatial_s=sn[['z_um','y_um','x_um']].to_numpy(float)-world(s)
        spatial_t=tn[['z_um','y_um','x_um']].to_numpy(float)-world(t)
        vals=[np.linalg.norm(moving-med),cosine(moving,med),profile,len(tn)-len(sn),
              np.mean(mutual) if mutual else np.nan,ratio(float(tn['_d'].mean()),float(sn['_d'].mean())),
              float(np.median(np.linalg.norm(np.asarray(motion)-med,axis=1))) if motion else np.nan,
              points_anisotropy(spatial_t)-points_anisotropy(spatial_s)]
        for key,value in zip(['motion_residual','motion_alignment','distance_profile_mae','neighbor_count_change','mutual_fraction','stretch','motion_mad','orientation_change'],vals):out[f'k{k}_{key}']=value
    return out


def sibling_hypotheses(pair,source,target,pairs,context,max_siblings=4):
    siblings=pairs.loc[(pairs.source_id==pair.source_id)&(pairs.target_id!=pair.target_id)].sort_values(['distance_um','target_id']).head(max_siblings)
    names=['available','sibling_distance','target_distance','daughter_separation','daughter_angle_cosine','midpoint_offset','mass_conservation_error','daughter_parent_mass_ratio','intensity_asymmetry','size_asymmetry','daughter_mean_contrast','source_elongation','separation_size_ratio','distance_asymmetry','geometry_cost','sibling_options']
    candidates=[];s,t=source['row'],target['row'];ms=source['moments'][0];mt=target['moments'][0]
    for _,edge in siblings.iterrows():
        sibling=context[int(edge.target_id)];u=sibling['row'];mu=sibling['moments'][0]
        v=world(t)-world(s);w=world(u)-world(s);vt=np.linalg.norm(v);vu=np.linalg.norm(w);separation=np.linalg.norm(v-w)
        mid=np.linalg.norm((v+w)/2);daughter_size=np.sqrt(max(float(target['moments'][3].sum()+sibling['moments'][3].sum()),0.))
        massratio=ratio(mt+mu,ms);masserror=abs(massratio-1.)
        cost=float(mid+.25*(vt+vu)+abs(vt-vu)) # fixed geometry heuristic, not calibrated division likelihood
        vals=[1.,vu,vt,separation,cosine(v,w),mid,masserror,massratio,
              ratio(abs(t['core_mean_raw']-u['core_mean_raw']),abs(t['core_mean_raw'])+abs(u['core_mean_raw'])),
              ratio(abs(target['moments'][3].sum()-sibling['moments'][3].sum()),target['moments'][3].sum()+sibling['moments'][3].sum()),
              (t['robust_contrast']+u['robust_contrast'])/2,s['weighted_shape_ratio'],ratio(separation,daughter_size),abs(vt-vu),cost,len(siblings)]
        candidates.append((cost,int(u['node_id']),dict(zip(names,vals))))
    if not candidates:
        out=dict.fromkeys(names,np.nan);out['available']=0.;out['sibling_options']=0.;return out,[]
    candidates.sort(key=lambda item:(item[0],item[1]))
    rows=[]
    for cost,sibling_id,values in candidates:
        if int(pair.target_id)<sibling_id:
            rows.append({'source_id':int(pair.source_id),'daughter_a':int(pair.target_id),'daughter_b':sibling_id,
                         't_source':int(pair.t_source),'t_target':int(pair.t_target),**values})
    return candidates[0][2],rows


def pair_frame_features(current_pairs,all_pairs,context,frames,declared,max_siblings=4,emit=lambda **_:None):
    correspondences={(int(p.source_id),int(p.target_id)):correspondence(context[int(p.source_id)],context[int(p.target_id)]) for _,p in current_pairs.iterrows()}
    rows=[];divisions=[]
    appearance_names=['ring0_mean','ring1_mean','ring2_mean','ring3_mean','mass','eig_max','texture_entropy','coherence']
    appearance_columns=[f'r1_radial__ring{i}_mean' for i in range(4)]+['r1_moments__r5_mass','r1_moments__r5_eig_max','r1_texture__x_entropy','r1_structure__r5_coherence']
    for i,(_,pair) in enumerate(current_pairs.iterrows()):
        source=context[int(pair.source_id)];target=context[int(pair.target_id)];s,t=source['row'],target['row'];sr,tr=source['r1'],target['r1']
        groups={'geometry':pair_geometry(source,target)};appearance={}
        for name,column in zip(appearance_names,appearance_columns):
            a,b=sr[column],tr[column]
            appearance[name+'_difference']=b-a;appearance[name+'_relative_difference']=ratio(b-a,abs(a)+abs(b))
        groups['appearance']=appearance;corr=correspondences[(int(pair.source_id),int(pair.target_id))];groups['correspondence']=corr
        groups['competition']=competition(pair,current_pairs)
        groups['history']=history(source,target,context,frames)
        groups['neighborhood']=neighborhood(source,target,frames)
        es,et=source['moments'][3],target['moments'][3]
        other=[v['r5_ncc'] for (sid,tid),v in correspondences.items() if sid==int(pair.source_id) and tid!=int(pair.target_id) and np.isfinite(v['r5_ncc'])]
        names=['orientation_alignment','eig_min_logratio','eig_mid_logratio','eig_max_logratio','trace_logratio','mass_logratio','contrast_logratio','patch_ncc_margin','fine_hessian_sign_change','scale_change','background_change','normalized_intensity_change','source_response','target_response','minimum_response','response_change']
        vals=[abs(cosine(source['moments'][4],target['moments'][4])),*[logratio(b,a) for a,b in zip(es,et)],logratio(et.sum(),es.sum()),
              logratio(target['moments'][0],source['moments'][0]),logratio(t['robust_contrast'],s['robust_contrast']),corr['r5_ncc']-max(other) if other else np.nan,
              float(np.sign(sr['r1_scale__s1p5_hessian_min'])!=np.sign(tr['r1_scale__s1p5_hessian_min'])),
              tr['r1_scale_shape__best_scale']-sr['r1_scale_shape__best_scale'],target['patch'].background-source['patch'].background,
              tr['r1_radial__ring0_mean']-sr['r1_radial__ring0_mean'],s['log_response'],t['log_response'],min(s['log_response'],t['log_response']),t['log_response']-s['log_response']]
        groups['conservation']=dict(zip(names,vals))
        groups['division'],hypotheses=sibling_hypotheses(pair,source,target,current_pairs,context,max_siblings);divisions.extend(hypotheses)
        result={'source_id':int(pair.source_id),'target_id':int(pair.target_id),'t_source':int(pair.t_source),'t_target':int(pair.t_target)}
        for family,names in declared.items():
            if set(groups[family])!=set(names):raise ValueError(f'Round 2 schema mismatch: {family}.')
            result.update({f'r2_{family}__{key}':value for key,value in finite_or_nan(groups[family]).items()})
        rows.append(result)
        if (i+1)%8==0:emit(event='pair_features',completed=i+1,total=len(current_pairs))
    cols=['source_id','daughter_a','daughter_b','t_source','t_target',*declared['division']]
    return pd.DataFrame(rows),pd.DataFrame(divisions,columns=cols).drop_duplicates(['source_id','daughter_a','daughter_b'])
