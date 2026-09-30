"""Aggregate VERIFIED official per-video count outputs, not a graph-matching scorer.

The official scorer must supply edge/division TP/FP/FN and coarse true-node
estimates. This module deliberately cannot score unlabeled feature matrices.
Reference: organizer metrics.md, reviewed 2026-09-13, equations for adjusted
edge Jaccard, sample-size weighting, and micro-averaged division Jaccard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COUNTS=['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','predicted_nodes','estimated_true_nodes']


def aggregate(counts):
    if counts.empty or any(c not in counts for c in COUNTS):raise ValueError('Official per-video counts are required; feature tables are not scores.')
    v=counts[COUNTS].to_numpy(float)
    if not np.isfinite(v).all() or (v<0).any() or (counts.estimated_true_nodes<=0).any():raise ValueError('Invalid official count inputs.')
    if np.any(v[:,:7]!=np.rint(v[:,:7])):raise ValueError('Counts must be integral; the coarse true-node estimate may be nonintegral.')
    weights=counts.edge_tp+counts.edge_fp+counts.edge_fn
    raw=np.divide(counts.edge_tp.to_numpy(float),weights.to_numpy(float),out=np.zeros(len(counts)),where=weights.to_numpy()>0)
    penalty=1-.1*(counts.predicted_nodes-counts.estimated_true_nodes)/counts.estimated_true_nodes
    adjusted=np.maximum(0.,raw*penalty.to_numpy(float))
    edge=float(np.dot(adjusted,weights)/weights.sum()) if weights.sum()>0 else 0.
    d=counts[['division_tp','division_fp','division_fn']].sum();denom=float(d.sum());division=float(d.division_tp/denom) if denom>0 else 0.
    return {'adjusted_edge_jaccard':edge,'division_jaccard':division,'combined_score':edge+.1*division}


def compare_verified(baseline,challenger,seed=20260914,repeats=1000):
    keys=['sample_id','embryo_id','split_hash','scorer_sha256','detector_hash']
    for table in (baseline,challenger):
        if any(c not in table for c in keys) or table.sample_id.duplicated().any():raise ValueError('Unique samples and provenance columns are required.')
    a=baseline.sort_values('sample_id').reset_index(drop=True);b=challenger.sort_values('sample_id').reset_index(drop=True)
    if not a[keys].equals(b[keys]):raise ValueError('Split, scorer, detector or sample mismatch; comparison is invalid.')
    for c in ['split_hash','scorer_sha256','detector_hash']:
        if a[c].nunique()!=1:raise ValueError('One explicit provenance value per controlled comparison is required.')
    result={'baseline':aggregate(a),'challenger':aggregate(b),'bootstrap_unit':'embryo','seed':seed,'repeats':repeats}
    result['delta']=result['challenger']['combined_score']-result['baseline']['combined_score']
    groups=sorted(a.embryo_id.unique())
    if len(groups)<3:
        result['interval']=None;result['interval_status']='insufficient_independent_embryos';return result
    if not 100<=repeats<=2000:raise ValueError('Bootstrap repetitions outside bounded range.')
    rng=np.random.default_rng(seed);deltas=[]
    for _ in range(repeats):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        x=pd.concat([a.loc[a.embryo_id==g] for g in sampled],ignore_index=True)
        y=pd.concat([b.loc[b.embryo_id==g] for g in sampled],ignore_index=True)
        deltas.append(aggregate(y)['combined_score']-aggregate(x)['combined_score'])
    result['interval']=[float(x) for x in np.quantile(deltas,[.025,.975])]
    result['interval_status']='exploratory_paired_embryo_bootstrap_not_correction_for_many_experiments'
    return result
