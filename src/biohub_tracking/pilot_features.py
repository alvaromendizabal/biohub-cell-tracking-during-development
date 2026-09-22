"""Image-derived candidate/pair descriptors, NOT a trained tracking model or official scorer.

No ground-truth graph, node ID or future annotation is accepted by these functions.
Morphology is a local intensity proxy; candidate crowding is not true cell density.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


def normalize_volume(volume):
    volume=np.asarray(volume)
    if volume.ndim!=3 or not np.all(np.isfinite(volume)):
        raise ValueError('Expected a finite 3-D ROI.')
    v=volume.astype(np.float32,copy=False)
    lo,hi=np.quantile(v,[0.01,0.998])
    if hi<=lo:return np.zeros_like(v),{'q01':float(lo),'q998':float(hi),'flat':True}
    return np.clip((v-lo)/(hi-lo),0,1).astype(np.float32),{'q01':float(lo),'q998':float(hi),'flat':False}


def physical_log(volume,spacing_um,sigma_um):
    """-sigma^2 times the PHYSICAL Laplacian, not an unscaled voxel Laplacian."""
    v=np.asarray(volume,dtype=np.float32);spacing=np.asarray(spacing_um,dtype=float)
    if v.ndim!=3 or spacing.shape!=(3,) or np.any(spacing<=0) or not np.all(np.isfinite(spacing)) or sigma_um<=0:
        raise ValueError('Invalid volume, spacing or Gaussian scale.')
    out=np.zeros_like(v);sigmas=float(sigma_um)/spacing
    for axis in range(3):
        order=[0,0,0];order[axis]=2
        derivative=ndimage.gaussian_filter(v,sigma=sigmas,order=order,mode='reflect')
        out-=derivative*np.float32(float(sigma_um)**2/spacing[axis]**2)
    return out


def select_candidates(response,spacing_um,cfg):
    response=np.asarray(response);spacing=np.asarray(spacing_um,dtype=float)
    if response.ndim!=3 or not np.isfinite(response).all():raise ValueError('Invalid detector response.')
    size=tuple(int(2*math.ceil(cfg['nms_distance_um']/s)+1) for s in spacing)
    mask=(response==ndimage.maximum_filter(response,size=size,mode='nearest'))
    threshold=max(0.,float(np.quantile(response,cfg['candidate_quantile'])))
    mask&=response>threshold
    # Exclude 6-micron borders so patch descriptors do not depend on padding.
    margin=np.ceil(6./spacing).astype(int)
    for axis,m in enumerate(margin):
        if 2*m>=response.shape[axis]:raise ValueError('ROI too small for physical patch descriptors.')
        sl=[slice(None)]*3;sl[axis]=slice(0,m);mask[tuple(sl)]=False
        sl[axis]=slice(-m,None);mask[tuple(sl)]=False
    coords=np.argwhere(mask);raw_count=len(coords)
    if raw_count:
        scores=response[tuple(coords.T)]
        # argwhere order is lexicographic; stable score sorting gives deterministic ties.
        order=np.argsort(-scores,kind='stable')[:cfg['max_peak_proposals']]
        accepted=[]
        for j in order:
            p=coords[j]
            if accepted and np.any(np.linalg.norm((np.asarray(accepted)-p)*spacing,axis=1)<cfg['nms_distance_um']):continue
            accepted.append(p)
            if len(accepted)>=cfg['max_candidates_per_frame']:break
        coords=np.asarray(accepted,dtype=int).reshape(-1,3)
    else:coords=np.empty((0,3),dtype=int)
    return coords,{'threshold':threshold,'raw_peak_count':raw_count,'kept_count':len(coords),
                   'candidate_cap_reached':len(coords)==cfg['max_candidates_per_frame'],
                   'proposal_cap_reached':raw_count>cfg['max_peak_proposals']}


def patch_descriptors(raw,coord,spacing_um):
    spacing=np.asarray(spacing_um,dtype=float);coord=np.asarray(coord,dtype=int)
    radii=np.ceil(6./spacing).astype(int)
    if np.any(coord-radii<0) or np.any(coord+radii>=np.asarray(raw.shape)):
        raise ValueError('Candidate is too near the ROI boundary for a complete patch.')
    slices=tuple(slice(int(c-r),int(c+r+1)) for c,r in zip(coord,radii))
    patch=np.asarray(raw[slices],dtype=np.float64)
    axes=[np.arange(-r,r+1)*s for r,s in zip(radii,spacing)]
    coords=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    radius=np.linalg.norm(coords,axis=-1)
    core=patch[radius<=2.5];shell=patch[(radius>=4.)&(radius<=6.)]
    bg=float(np.median(shell));mad=float(np.median(np.abs(shell-bg)))
    core_mean=float(core.mean());noise=1.4826*mad
    inside=radius<=4.;points=coords[inside]
    weights=np.maximum(patch[inside]-bg,0.);total=float(weights.sum())
    # Explicitly missing morphology if there is no positive local signal.
    eig=np.full(3,np.nan)
    if total>0:
        center=np.sum(points*weights[:,None],axis=0)/total
        delta=points-center
        covariance=(delta.T*weights)@delta/total
        eig=np.maximum(np.linalg.eigvalsh(covariance),0.)
    return {'core_mean_raw':core_mean,'core_p90_raw':float(np.quantile(core,.9)),
            'shell_median_raw':bg,'shell_mad_raw':mad,'core_minus_shell_raw':core_mean-bg,
            'robust_contrast':(core_mean-bg)/max(noise,1.),
            'positive_patch_mass_raw':total,'weighted_variance_small_um2':float(eig[0]),
            'weighted_variance_mid_um2':float(eig[1]),'weighted_variance_large_um2':float(eig[2]),
            'weighted_shape_ratio':float((eig[2]+1e-8)/(eig[0]+1e-8))}


def candidate_features(raw,coords,responses,spacing_um,sigmas,frame,offsets,cfg):
    spacing=np.asarray(spacing_um,dtype=float);rows=[]
    for index,p in enumerate(coords):
        global_p=p+np.asarray(offsets);physical=global_p*spacing
        values=np.asarray([r[tuple(p)] for r in responses],dtype=float)
        positive=np.maximum(values,0.);denom=positive.sum()
        weights=positive/denom if denom>0 else np.zeros_like(positive)
        nonzero=weights[weights>0]
        entropy=-float(np.sum(nonzero*np.log(nonzero)))
        row={'node_id':int(frame*cfg['max_candidates_per_frame']+index),'t':int(frame),
             'z':int(global_p[0]),'y':int(global_p[1]),'x':int(global_p[2]),
             'z_um':float(physical[0]),'y_um':float(physical[1]),'x_um':float(physical[2]),
             'best_sigma_um':float(sigmas[int(np.argmax(values))]),'log_response':float(values.max()),
             'scale_support_count':int(np.sum(positive>=0.5*positive.max())) if positive.max()>0 else 0,
             'scale_response_entropy':entropy,
             'roi_boundary_distance_um':float(np.min(np.minimum(p,np.asarray(raw.shape)-1-p)*spacing))}
        for s,v in zip(sigmas,values):row[f'log_sigma_{s:g}um']=float(v)
        row.update(patch_descriptors(raw,p,spacing))
        rows.append(row)
    if not rows:return pd.DataFrame(columns=['node_id','t','z','y','x','z_um','y_um','x_um'])
    result=pd.DataFrame(rows);points=result[['z_um','y_um','x_um']].to_numpy()
    tree=cKDTree(points)
    if len(points)>1:
        distances,_=tree.query(points,k=2)
        result['candidate_nearest_neighbor_um']=distances[:,1]
    else:result['candidate_nearest_neighbor_um']=np.nan
    for radius in (5.,10.):
        result[f'candidate_neighbors_within_{radius:g}um']=[len(ids)-1 for ids in tree.query_ball_point(points,r=radius)]
    return result


def phase_shift(source,target):
    """Image-only integer-voxel shift SOURCE -> TARGET; periodic diagnostic, not optical flow."""
    a=np.asarray(source,dtype=float);b=np.asarray(target,dtype=float)
    if a.ndim!=3 or a.shape!=b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Phase correlation requires matching finite 3-D ROIs.')
    if np.array_equal(a,b):return np.zeros(3,dtype=float),{'identical_roi':True,'phase_peak_ratio':None}
    a=a-a.mean();b=b-b.mean()
    if a.std()==0 or b.std()==0:return np.zeros(3,dtype=float),{'identical_roi':False,'phase_peak_ratio':None}
    spectrum=np.fft.fftn(b)*np.conj(np.fft.fftn(a))
    magnitude=np.abs(spectrum);norm=np.zeros_like(spectrum)
    np.divide(spectrum,magnitude,out=norm,where=magnitude>1e-12)
    correlation=np.fft.ifftn(norm).real
    peak=np.unravel_index(np.argmax(correlation),correlation.shape)
    shift=np.asarray(peak,dtype=float);shape=np.asarray(a.shape)
    shift=np.where(shift>shape//2,shift-shape,shift)
    ratio=float(correlation[peak]/max(np.mean(np.abs(correlation)),1e-12))
    return shift,{'identical_roi':False,'phase_peak_ratio':ratio}


def pair_features(source,target,drift_um,cfg):
    """Generate top-k geometric pairs; no label-derived predecessor or candidate filtering."""
    columns=['source_id','target_id','t_source','t_target','rank_by_distance','distance_um',
             'dz_um','dy_um','dx_um','drift_residual_um','drift_residual_z_um','drift_residual_y_um',
             'drift_residual_x_um','mutual_nearest','nearest_competitor_gap_um',
             'core_mean_difference_raw','core_mean_ratio','contrast_difference','sigma_difference_um',
             'source_crowding_10um','target_crowding_10um']
    if source.empty or target.empty:return pd.DataFrame(columns=columns)
    if source['t'].nunique()!=1 or target['t'].nunique()!=1 or int(target.iloc[0]['t'])-int(source.iloc[0]['t'])!=1:
        raise ValueError('Only adjacent frames are allowed in this pilot.')
    a=source[['z_um','y_um','x_um']].to_numpy();b=target[['z_um','y_um','x_um']].to_numpy()
    distances=np.linalg.norm(b[None,:,:]-a[:,None,:],axis=2)
    forward=np.argmin(distances,axis=1);reverse=np.argmin(distances,axis=0);rows=[]
    for i in range(len(source)):
        ordering=np.argsort(distances[i],kind='stable');gap=float(distances[i,ordering[1]]-distances[i,ordering[0]]) if len(ordering)>1 else np.nan
        for rank,j in enumerate(ordering[:cfg['pair_top_k']],1):
            distance=float(distances[i,j])
            if distance>cfg['pair_radius_um']:continue
            s=source.iloc[i];t=target.iloc[j];d=b[j]-a[i];residual=d-np.asarray(drift_um)
            rows.append([int(s.node_id),int(t.node_id),int(s.t),int(t.t),rank,distance,*map(float,d),
                float(np.linalg.norm(residual)),*map(float,residual),bool(forward[i]==j and reverse[j]==i),gap,
                float(t.core_mean_raw-s.core_mean_raw),float(t.core_mean_raw/max(abs(s.core_mean_raw),1.)),
                float(t.robust_contrast-s.robust_contrast),float(t.best_sigma_um-s.best_sigma_um),
                int(s.candidate_neighbors_within_10um),int(t.candidate_neighbors_within_10um)])
    return pd.DataFrame(rows,columns=columns)
