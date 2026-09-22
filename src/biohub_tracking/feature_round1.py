"""Round 1: eight image/node families, 16 declared descriptors per family.

The existing detector outputs are held fixed. Every operation uses the current
frame only. Fixed per-frame normalization is available at inference, not fitted
on labels. Hessian responses near boundaries depend on reflected image support.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage
from .feature_math import (patch, moments, ratio, normalize, quantized_glcm,
    sample_scale_features, points_anisotropy, finite_or_nan)


def node_groups(raw, normal, p, gradients, spacing, local_coord, node, frame_nodes, scale_values, norm_info):
    groups={}
    radial={}
    for i,(low,high) in enumerate([(0.,1.5),(1.5,3.),(3.,4.5),(4.5,6.)]):
        mask=(p.radius>=low)&(p.radius<=high) if i==0 else (p.radius>low)&(p.radius<=high)
        v=p.normal[mask]
        for name,value in zip(['mean','median','std','p90'],[v.mean(),np.median(v),v.std(),np.quantile(v,.9)]):
            radial[f'ring{i}_{name}']=value
    groups['radial']=radial;groups['scale']=scale_values
    out={}
    voxel_volume=float(np.prod(spacing))
    for radius in [3,5]:
        mass,centroid,cov,eig,orientation=moments(p,radius)
        vals=[mass*voxel_volume,np.linalg.norm(centroid),*eig,ratio(eig[2],eig[0]),ratio(eig[1]-eig[0],eig[2]),ratio(3*max(float(np.prod(eig)),0.)**(1/3),eig.sum())]
        for key,value in zip(['mass','centroid_offset','eig_min','eig_mid','eig_max','elongation','flatness','isotropy'],vals):out[f'r{radius}_{key}']=value
    groups['moments']=out
    texture={}
    step=np.maximum(1,np.rint(1.625/np.asarray(spacing)).astype(int))
    for name,shift in [('z',(step[0],0,0)),('y',(0,step[1],0)),('x',(0,0,step[2])),('xy',(0,step[1],step[2]))]:
        for metric,value in zip(['contrast','homogeneity','energy','entropy'],quantized_glcm(p.normal,shift)):
            texture[f'{name}_{metric}']=value
    groups['texture']=texture
    structure={};gp=np.stack([g[p.slices] for g in gradients],axis=-1)
    for radius in [3,5]:
        values=gp[p.radius<=radius];mag=np.linalg.norm(values,axis=1)
        tensor=values.T@values/len(values);eig,vec=np.linalg.eigh(tensor);eig=np.maximum(eig,0.)
        vals=[mag.mean(),mag.std(),np.quantile(mag,.9),*eig,ratio(eig[2]-eig[1],eig.sum()),vec[0,-1]**2 if eig[-1]>1e-12 else np.nan]
        for key,value in zip(['gradient_mean','gradient_std','gradient_p90','tensor_min','tensor_mid','tensor_max','coherence','principal_z2'],vals):structure[f'r{radius}_{key}']=value
    groups['structure']=structure
    context={};world=np.array([node[k] for k in ['z_um','y_um','x_um']])
    other=frame_nodes.loc[frame_nodes.node_id!=node['node_id']]
    delta=other[['z_um','y_um','x_um']].to_numpy(float)-world
    distance=np.linalg.norm(delta,axis=1)
    for radius in [8,12,16,20]:
        keep=distance<=radius;selected=other.loc[keep]
        vals=[int(keep.sum()),float(distance[keep].mean()) if keep.any() else np.nan,
              points_anisotropy(delta[keep]),ratio(float(selected.core_mean_raw.mean()),node['core_mean_raw']) if keep.any() else np.nan]
        for key,value in zip(['count','mean_distance','anisotropy','neighbor_intensity_ratio'],vals):context[f'r{radius}_{key}']=value
    groups['context']=context
    inside=p.radius<=5.;positive=np.maximum(p.raw-p.background,0.)*inside;mass=positive.sum()
    threshold=p.background+.5*max(0.,float(p.raw[inside].max()-p.background))
    foreground=(p.raw>threshold)&inside
    labels,count=ndimage.label(foreground,structure=ndimage.generate_binary_structure(3,1))
    center_label=int(labels[p.center]);component=(labels==center_label).sum() if center_label>0 else 0
    peak=np.unravel_index(np.argmax(np.where(inside,p.raw,-np.inf)),p.raw.shape)
    peak_offset=np.linalg.norm(p.xyz[peak])
    border=float(np.min(np.minimum(local_coord,np.array(raw.shape)-1-local_coord)*spacing))
    saturation=np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype,np.integer) else np.inf
    core=p.raw[p.radius<=1.5]
    quality_values=[np.mean(p.raw==0),np.mean(p.raw==saturation),p.noise/1.4826,
        ratio(float(core.mean()-p.background),max(p.noise,1.)),peak_offset,
        ratio(positive[p.radius<=1.5].sum(),mass),ratio(positive[p.radius<=3].sum(),mass),
        foreground.sum()*voxel_volume,component*voxel_volume,count,ratio(core.mean(),p.background),
        border,norm_info['high']-norm_info['low'],float(norm_info['flat']),p.normal.std(),p.normal[p.center]-p.normal.mean()]
    quality_names=['zero_fraction','saturation_fraction','shell_mad','core_snr','center_peak_offset','concentration_1p5','concentration_3','above_half_volume','central_component_volume','component_count','core_shell_ratio','border_margin_um','frame_dynamic_range','frame_flat_flag','patch_variation','center_minus_patch_mean']
    groups['quality']=dict(zip(quality_names,quality_values))
    response=np.array([scale_values['s'+str(s).replace('.','p')+'_log'] for s in [1.5,2.5,3.5,5.0]])
    scales=np.array([1.5,2.5,3.5,5.]);positive=np.maximum(response,0.);total=positive.sum();w=positive/total if total>0 else np.zeros(4)
    nz=w[w>0];mean=float(np.dot(w,scales)) if total>0 else np.nan
    sorted_response=np.sort(response);best=int(np.argmax(response))
    names=['best_scale','range','max','min','mean','std','positive_count','entropy','top_margin','edge_peak','monotonic_increase','curvature','low_high_ratio','weighted_mean','weighted_std','negative_fraction']
    vals=[scales[best],np.ptp(response),response.max(),response.min(),response.mean(),response.std(),(response>0).sum(),-np.sum(nz*np.log(nz)) if total>0 else np.nan,
          sorted_response[-1]-sorted_response[-2],float(best in [0,3]),float(np.all(np.diff(response)>0)),
          np.gradient(np.gradient(response,scales),scales)[best],ratio(response[0],response[-1]),mean,np.sqrt(np.dot(w,(scales-mean)**2)) if total>0 else np.nan,np.mean(response<0)]
    groups['scale_shape']=dict(zip(names,vals))
    return {family:finite_or_nan(values) for family,values in groups.items()}


def frame_features(raw, frame_nodes, spacing, offsets, declared, emit=lambda **_:None):
    normal,info=normalize(raw);coords=frame_nodes[['z','y','x']].to_numpy(int)-np.asarray(offsets)
    scale_values=sample_scale_features(normal,spacing,coords)
    gradients=[ndimage.gaussian_filter(normal,sigma=1./np.asarray(spacing),order=tuple(int(i==j) for i in range(3)),mode='reflect')/spacing[j] for j in range(3)]
    rows=[]
    for i,(_,n) in enumerate(frame_nodes.iterrows()):
        center=coords[i];p=patch(raw,normal,center,spacing)
        groups=node_groups(raw,normal,p,gradients,np.asarray(spacing),center,n,frame_nodes,scale_values[i],info)
        result={'node_id':int(n.node_id),'t':int(n.t)}
        for family,names in declared.items():
            if set(groups[family])!=set(names):raise ValueError(f'Round 1 schema mismatch: {family}.')
            result.update({f'r1_{family}__{key}':groups[family][key] for key in names})
        rows.append(result)
        if (i+1)%8==0:emit(event='node_features',completed=i+1,total=len(frame_nodes))
    return pd.DataFrame(rows)
