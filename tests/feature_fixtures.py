"""Tiny deterministic synthetic data for USER-RUN tests, never experimental evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from biohub_tracking.feature_math import normalize,patch,moments
from biohub_tracking.feature_round1 import frame_features
from biohub_tracking.feature_round2 import make_context,pair_frame_features

ROOT=Path(__file__).resolve().parents[1]


def small_scene():
    spacing=np.array([1.,1.,1.]);offsets=np.array([0,0,0]);frames=np.arange(4);grid=np.indices((21,41,41),dtype=float)
    images=[];rows=[]
    for t in frames:
        image=np.full((21,41,41),30.,float);centers=[]
        for j in range(3):
            center=np.array([10,12+8*j,12+8*j+int(t%2)])
            squared=sum((grid[a]-center[a])**2 for a in range(3));image+=(800+50*j+10*t)*np.exp(-squared/(2*(2.+.1*j)**2));centers.append(center)
        image=np.rint(image).astype(np.uint16);normal,_=normalize(image);images.append(image)
        for j,c in enumerate(centers):
            p=patch(image,normal,c,spacing);mass,cent,cov,eig,o=moments(p,5.)
            core=p.raw[p.radius<=1.5]
            rows.append(dict(node_id=int(10*t+j),t=int(t),z=int(c[0]),y=int(c[1]),x=int(c[2]),z_um=float(c[0]),y_um=float(c[1]),x_um=float(c[2]),
              core_mean_raw=float(core.mean()),robust_contrast=float((core.mean()-p.background)/max(1.,p.noise)),best_sigma_um=2.,
              roi_boundary_distance_um=float(np.min(np.minimum(c,np.array(image.shape)-1-c))),weighted_shape_ratio=float(eig[-1]/eig[0]),log_response=.5+j*.01))
    nodes=pd.DataFrame(rows);edges=[]
    for t in frames[:-1]:
        for _,s in nodes.loc[nodes.t==t].iterrows():
            for _,u in nodes.loc[nodes.t==t+1].iterrows():
                distance=float(np.linalg.norm(u[['z_um','y_um','x_um']].to_numpy(float)-s[['z_um','y_um','x_um']].to_numpy(float)))
                edges.append(dict(source_id=int(s.node_id),target_id=int(u.node_id),t_source=int(t),t_target=int(t+1),distance_um=distance))
    return np.stack(images),frames,offsets,spacing,nodes,pd.DataFrame(edges)


def constructed():
    cfg=json.loads((ROOT/'configs/feature_rounds.json').read_text());images,frames,offsets,spacing,nodes,pairs=small_scene()
    one=pd.concat([frame_features(images[i],nodes.loc[nodes.t==t],spacing,offsets,cfg['round1']['families']) for i,t in enumerate(frames)],ignore_index=True)
    context=make_context(images,frames,offsets,spacing,nodes,one);byframe={int(t):nodes.loc[nodes.t==t] for t in frames}
    allpairs=[];allhypotheses=[]
    for t in frames[:-1]:
        table,division=pair_frame_features(pairs.loc[pairs.t_source==t],pairs,context,byframe,cfg['round2']['families'])
        allpairs.append(table);allhypotheses.append(division)
    return cfg,images,frames,offsets,spacing,nodes,pairs,one,pd.concat(allpairs,ignore_index=True),pd.concat(allhypotheses,ignore_index=True),context,byframe
