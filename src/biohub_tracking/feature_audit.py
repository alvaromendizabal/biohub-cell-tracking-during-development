"""Integrity and redundancy diagnostics; neither feature importance nor validation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def audit(table,prefix):
    columns=[c for c in table if c.startswith(prefix)]
    if len(columns)!=128:raise ValueError(f'Expected 128 declared descriptors, found {len(columns)}.')
    values=table[columns].to_numpy(float)
    if np.isinf(values).any():raise ValueError('Infinite feature value; NaN is required for missing evidence.')
    rows=[]
    for c in columns:
        v=table[c];finite=v[np.isfinite(v)]
        rows.append({'feature':c,'family':c.split('__')[0].split('_',1)[1],
            'rows':len(v),'finite_rows':len(finite),'missing_fraction':float(v.isna().mean()),'distinct_values':int(v.nunique()),
            'constant_in_pilot':bool(v.nunique()<=1),'mean':float(finite.mean()) if len(finite) else np.nan,
            'std':float(finite.std(ddof=0)) if len(finite) else np.nan,
            'decision':'unlabeled_quality_diagnostic_only'})
    quality=pd.DataFrame(rows);corr=table[columns].corr(min_periods=8);redundant=[]
    for i,a in enumerate(columns):
        for b in columns[i+1:]:
            value=corr.loc[a,b]
            if np.isfinite(value) and abs(value)>=.995:
                redundant.append({'feature_a':a,'feature_b':b,'pearson_r':float(value),'paired_finite_rows':int((table[a].notna()&table[b].notna()).sum()),'scope':'same tiny development crop; not independent evidence'})
    return quality,pd.DataFrame(redundant,columns=['feature_a','feature_b','pearson_r','paired_finite_rows','scope'])


def planned_ablations(families):
    """Fixed detector, same scorer/linker/splits. No classifier is fitted here."""
    families=list(families)
    return ([{'experiment':'baseline','families':[]}]+[{'experiment':'add_'+f,'families':[f]} for f in families]+
            [{'experiment':'all','families':families}]+[{'experiment':'remove_'+f,'families':[x for x in families if x!=f]} for f in families])
