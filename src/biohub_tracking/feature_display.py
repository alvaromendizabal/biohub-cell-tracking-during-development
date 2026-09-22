"""Persistent inline Plotly galleries without a Jupyter Plotly MIME extension.

One self-contained HTML iframe per notebook contains all figures and one copy
of plotly.js. The iframe has a visible loading/error panel, a render counter,
and a verification code. No CDN, browser launch, server extension, or ports.
A returned Python display call is NOT recorded as visual success.
"""
from __future__ import annotations
import html,json,re
from pathlib import Path
from .feature_io import (safe,verify,digest,read_json,write_json,atomic,fingerprint,utc)


def compatibility_spec(spec):
    """Keep the source data; use an explicitly titled XY projection for 3-D plots."""
    spec=json.loads(json.dumps(spec))
    projected=False
    for trace in spec.get('data',[]):
        if trace.get('type')=='scatter3d':
            projected=True;trace['type']='scatter';trace.pop('z',None);trace.pop('scene',None)
            if isinstance(trace.get('marker'),dict):trace['marker'].pop('sizemode',None)
    if projected:
        layout=spec.setdefault('layout',{});layout.pop('scene',None)
        title=layout.get('title',{});title=title.get('text','Candidate positions') if isinstance(title,dict) else title
        layout['title']={'text':str(title)+' | XY compatibility projection'}
        layout['xaxis']={'title':{'text':'X (micrometers)'}};layout['yaxis']={'title':{'text':'Y (micrometers)'}}
    return spec


def build_page(specs,js_text,title,subtitle,code='',external_js=False):
    if not 1<=len(specs)<=10:raise ValueError('Gallery must have one through ten figures.')
    payload=json.dumps([compatibility_spec(s) for s in specs],ensure_ascii=True).replace('<','\\u003c').replace('&','\\u0026')
    if len(payload)>2*1024**2:raise ValueError('Gallery figure JSON exceeds two MiB; reduce the data before display.')
    # Raw JS is trusted only after its reviewed local file hash has been checked.
    script='<script src="plotly.min.js"></script>' if external_js else '<script>'+re.sub(r'</script',r'<\\/script',js_text,flags=re.I)+'</script>'
    panels=''.join(f'<section><div class="figure" id="figure{i}"></div><p id="status{i}">Figure {i+1}: waiting for JavaScript.</p></section>' for i in range(len(specs)))
    head='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:15px system-ui,sans-serif;margin:0;background:#f6f8fa;color:#192530;padding:20px}h1{font-size:24px;margin:0 0 10px}p{line-height:1.5}.banner{border:1px solid #aeb9c4;background:white;padding:14px;position:sticky;top:0;z-index:20}section{background:white;border:1px solid #d6dde4;border-radius:8px;margin:18px 0;padding:12px}.figure{height:480px;min-width:250px}.small{font-size:12px}code{font-weight:700}button{padding:8px 14px;cursor:pointer}#result{font-weight:650}summary{cursor:pointer}
</style></head><body>'''
    header='<h1>'+html.escape(title)+'</h1><p>'+html.escape(subtitle)+'</p><div class="banner"><div id="result">JavaScript has not started. If this stays here, the notebook trust or browser policy blocked embedded scripts; do not rerun feature computation.</div><div id="verify"></div></div>'
    tail='''<script type="application/json" id="figure-data">'''+payload+'''</script><script>
(function(){
 const result=document.getElementById('result'); const verify=document.getElementById('verify');
 const specs=JSON.parse(document.getElementById('figure-data').textContent); let finished=0;
 function fail(message){ result.textContent='DISPLAY ERROR: '+message+' — computation must stay stopped.'; }
 window.addEventListener('error',e=>fail(e.message||'browser script error'));
 if(!window.Plotly){fail('Plotly library did not load');return;}
 result.textContent='Plotly '+Plotly.version+' loaded. Rendering '+specs.length+' figures.';
 const timer=setTimeout(()=>{if(finished<specs.length)fail('not all charts completed within 30 seconds');},30000);
 (async function(){
  for(let i=0;i<specs.length;i++){
   const f=specs[i]; const layout=Object.assign({},f.layout||{});
   layout.height=480; layout.autosize=true; delete layout.width;
   try{
    await Plotly.newPlot('figure'+i,f.data||[],layout,{responsive:true,displaylogo:false,scrollZoom:false});
    finished++;document.getElementById('status'+i).textContent='Figure '+(i+1)+' rendered. Hover, zoom, or use the image-export button.';
    result.textContent='Rendered '+finished+' of '+specs.length+' Plotly figures.';
   }catch(e){document.getElementById('status'+i).textContent='Figure error: '+String(e);}
  }
  clearTimeout(timer);
  if(finished===specs.length){verify.textContent='DISPLAY CODE: '+__VERIFICATION_CODE__+' | Save, close, reopen, and test a hover before recording visual verification.';}
  else{fail(String(specs.length-finished)+' chart(s) failed');}
 })();
})();
</script></body></html>'''.replace('__VERIFICATION_CODE__',json.dumps(code or 'not a verification gate'))
    return head+header+panels+script+tail


def iframe_markup(page):
    if len(page.encode())>12*1024**2:raise ValueError('Self-contained gallery exceeds twelve MiB.')
    # A single isolated output avoids repeated multi-megabyte inline JS bundles.
    return '<div data-biohub-inline="self-contained"><p><strong>Interactive Plotly gallery inside this notebook.</strong> Scroll inside the panel to see all figures.</p><iframe title="Biohub inline Plotly gallery" sandbox="allow-scripts allow-downloads" referrerpolicy="no-referrer" style="width:100%;height:1100px;border:1px solid #ccd2d9" srcdoc="'+html.escape(page,quote=True)+'"></iframe><p>If the panel is blank, do not run a feature round. Check notebook trust and inspect the saved HTML; file paths alone are not visual success.</p></div>'


def visual_sources(root):
    cfg=read_json(Path(root)/'configs/visualization_sources.json')
    bundle=verify(root,cfg['js_path'],cfg['js_sha256'],8*1024**2).read_text(encoding='utf-8')
    specs=[]
    for relative,sha in cfg['figures'].items():specs.append(read_json(verify(root,relative,sha,1024**2)))
    signature=fingerprint({'js':cfg['js_sha256'],'figures':cfg['figures'],'display_source':digest(Path(__file__))})
    return specs,bundle,signature


def show_pilot_gallery(root):
    from IPython.display import display,HTML
    root=Path(root).resolve();print('PILOT_GALLERY_BUILD_STARTED — reusing saved figures; no feature computation.',flush=True)
    specs,js,signature=visual_sources(root);code=signature[:8].upper()
    page=build_page(specs,js,'Biohub | completed pilot, visible evidence','Eight figures from your completed four-frame pilot. Training fits: 0. Official score: not measured. The original 3-D plot is shown as an XY projection for compatibility.',code)
    path=root/'outputs/features/visualization/pilot_gallery.html';atomic(path,page)
    display(HTML(iframe_markup(page)))
    write_json(root/'outputs/features/visualization/display_request.json',{'utc':utc(),'signature':signature,'figure_count':len(specs),'display_request_returned':True,'visual_success':False,'html':path.relative_to(root).as_posix(),'html_sha256':digest(path)})
    print('PILOT_GALLERY_DISPLAY_REQUEST_RETURNED — this is not visual verification.',flush=True)


def attest_visible(root,code,saved_reopened_and_hovered=False):
    root=Path(root);_,_,signature=visual_sources(root)
    if not saved_reopened_and_hovered or str(code).strip().upper()!=signature[:8].upper():
        raise ValueError('STOP: enter the DISPLAY CODE actually shown inside the rendered gallery, and confirm save/reopen plus hover. Do not bypass a blank chart.')
    write_json(root/'outputs/features/visualization/visual_gate.json',{'utc':utc(),'signature':signature,'verification_type':'user_attestation_after_browser_render_code','saved_reopened_and_hovered':True,'figure_count':8})
    print('PLOTLY_VISUAL_GATE_RECORDED — user-confirmed, not an automatic browser test.',flush=True)


def require_visible(root):
    _,_,signature=visual_sources(root)
    gate=read_json(safe(root,'outputs/features/visualization/visual_gate.json'))
    if gate.get('signature')!=signature or gate.get('saved_reopened_and_hovered') is not True:raise ValueError('Complete notebook 05 visual gate before running either feature round.')
    return gate


def _record_specs(figures):return [json.loads(f.to_json()) for f in figures]


def _correlation_panel(table,prefix,go):
    import numpy as np
    columns=[c for c in table if c.startswith(prefix)];families=list(dict.fromkeys(c.split('__')[0] for c in columns));corr=table[columns].corr(min_periods=8).abs();values=[]
    for a in families:
        row=[]
        for b in families:
            ca=[c for c in columns if c.split('__')[0]==a];cb=[c for c in columns if c.split('__')[0]==b]
            block=corr.loc[ca,cb].to_numpy(dtype=float,copy=True)
            if a==b:
                np.fill_diagonal(block,np.nan)
            finite=block[np.isfinite(block)];row.append(float(finite.mean()) if len(finite) else None)
        values.append(row)
    return go.Figure(go.Heatmap(z=values,x=families,y=families,zmin=0,zmax=1,colorbar={'title':'Mean |r|'})).update_layout(title='Observed family redundancy — not predictive importance',xaxis_title='Feature family',yaxis_title='Feature family')


def _missing_panel(quality,go):
    view=quality.sort_values(['missing_fraction','feature'],ascending=[False,True]).head(18)
    return go.Figure(go.Bar(x=view.missing_fraction.tolist(),y=view.feature.tolist(),orientation='h')).update_layout(title='Missing evidence in this pilot — highest missing fractions',xaxis_title='Fraction of rows with undefined value',yaxis_title='Descriptor',margin={'l':280},yaxis={'autorange':'reversed'})


def round_figures(root,stage):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from .feature_io import source_signature
    root=Path(root);receipt=read_json(root/f'outputs/features/{stage}_receipt.json')
    if receipt.get('status')!='completed' or receipt.get('source_signature')!=source_signature(root):raise ValueError('No completed compatible feature round exists.')
    table=pd.read_csv(verify(root,receipt['table'],receipt['artifacts'][receipt['table']]))
    quality=pd.read_csv(verify(root,receipt['quality'],receipt['artifacts'][receipt['quality']]))
    nodes=pd.read_csv(root/'outputs/pilot/node_features.csv');basepairs=pd.read_csv(root/'outputs/pilot/pair_features.csv');figs=[]
    if stage=='round1':
        figs.append(go.Figure(compatibility_spec(read_json(root/'outputs/pilot/figures/04_mip_candidates.plotly.json'))))
        fig=go.Figure()
        for t,g in table.groupby('t'):
            cols=[f'r1_scale__s{str(s).replace(".","p")}_log' for s in [1.5,2.5,3.5,5.0]]
            fig.add_trace(go.Scatter(x=[1.5,2.5,3.5,5.],y=g[cols].mean().tolist(),mode='lines+markers',name=f'Frame {t}'))
        figs.append(fig.update_layout(title='Expanded physical-scale responses at fixed candidates',xaxis_title='Gaussian scale (micrometers)',yaxis_title='Mean normalized negative Laplacian'))
        fig=go.Figure()
        for t,g in table.groupby('t'):fig.add_trace(go.Scatter(x=[.75,2.25,3.75,5.25],y=[float(g[f'r1_radial__ring{i}_mean'].mean()) for i in range(4)],mode='lines+markers',name=f'Frame {t}'))
        figs.append(fig.update_layout(title='Physical radial signal profiles',xaxis_title='Ring midpoint (micrometers)',yaxis_title='Normalized intensity'))
        figs.append(go.Figure(go.Scatter(x=table.r1_moments__r5_mass.tolist(),y=table.r1_moments__r5_elongation.tolist(),mode='markers',text=table.node_id.astype(str).tolist(),marker={'color':table.t.tolist(),'showscale':True,'colorbar':{'title':'Frame'}})).update_layout(title='Intensity-moment morphology — not segmentation measurements',xaxis_title='Positive integrated signal (raw intensity × cubic micrometers)',yaxis_title='Largest / smallest moment eigenvalue'))
        fig=go.Figure()
        for t,g in table.groupby('t'):fig.add_trace(go.Bar(x=['Z','Y','X','XY'],y=[float(g[f'r1_texture__{a}_entropy'].mean()) for a in ['z','y','x','xy']],name=f'Frame {t}'))
        figs.append(fig.update_layout(title='Directional co-occurrence texture',xaxis_title='Physical offset direction',yaxis_title='Mean GLCM entropy (nats)',barmode='group'))
        fig=go.Figure()
        for t,g in table.groupby('t'):fig.add_trace(go.Scatter(x=[8,12,16,20],y=[float(g[f'r1_context__r{r}_count'].mean()) for r in [8,12,16,20]],name=f'Frame {t}',mode='lines+markers'))
        figs.append(fig.update_layout(title='Candidate crowding across physical neighborhood sizes',xaxis_title='Radius (micrometers)',yaxis_title='Mean detected neighbors; boundary-truncated'))
        figs.extend([_missing_panel(quality,go),_correlation_panel(table,'r1_',go)])
    else:
        merged=table.merge(basepairs[['source_id','target_id','distance_um']],on=['source_id','target_id'],validate='one_to_one')
        figs.append(go.Figure(go.Scatter(x=merged.distance_um.tolist(),y=merged.r2_correspondence__r5_ncc.tolist(),mode='markers',marker={'color':merged.t_source.tolist(),'showscale':True,'colorbar':{'title':'Source frame'}})).update_layout(title='Geometry versus observed patch similarity — candidate links, not labels',xaxis_title='Displacement (micrometers)',yaxis_title='Patch normalized cross-correlation'))
        columns=[c for c in table if c.startswith('r2_appearance__') and c.endswith('relative_difference')]
        figs.append(go.Figure(go.Bar(x=table[columns].abs().median().tolist(),y=columns,orientation='h')).update_layout(title='Appearance changes across fixed candidate pairs',xaxis_title='Median absolute relative difference',yaxis_title='Descriptor',margin={'l':270}))
        fig=go.Figure()
        for t,g in table.groupby('t_source'):fig.add_trace(go.Box(y=g.r2_history__prediction_residual.dropna().tolist(),name=f'Frame {t}',boxpoints='all'))
        figs.append(fig.update_layout(title='Predicted-history motion residual — first-frame history is missing',xaxis_title='Source frame',yaxis_title='Residual (micrometers)'))
        figs.append(go.Figure(go.Scatter(x=table.r2_competition__out_rank.tolist(),y=table.r2_competition__in_rank.tolist(),mode='markers')).update_layout(title='Competing outgoing and incoming candidate links',xaxis_title='Outgoing distance rank',yaxis_title='Incoming distance rank'))
        figs.append(go.Figure(go.Scatter(x=table.r2_neighborhood__k3_motion_residual.tolist(),y=table.r2_neighborhood__k5_motion_residual.tolist(),mode='markers')).update_layout(title='Local collective-motion residual at two neighborhood sizes',xaxis_title='3-neighbor residual (micrometers)',yaxis_title='5-neighbor residual (micrometers)'))
        division=pd.read_csv(verify(root,receipt['division_table'],receipt['artifacts'][receipt['division_table']]))
        fig=go.Figure(go.Scatter(x=division.midpoint_offset.tolist(),y=division.mass_conservation_error.tolist(),mode='markers'))
        if division.empty:fig.add_annotation(text='No sibling hypotheses exist in this bounded candidate set.',showarrow=False)
        figs.append(fig.update_layout(title='Division hypotheses — not predicted or confirmed divisions',xaxis_title='Parent-to-daughter-midpoint offset (micrometers)',yaxis_title='Absolute approximate mass-ratio deviation from 1'))
        figs.extend([_missing_panel(quality,go),_correlation_panel(table,'r2_',go)])
    return _record_specs(figs),receipt


def show_round_gallery(root,stage):
    from IPython.display import display,HTML
    root=Path(root).resolve();require_visible(root)
    specs,receipt=round_figures(root,stage);_,js,_=visual_sources(root)
    target=root/'outputs/features/figures'/stage;target.mkdir(parents=True,exist_ok=True)
    atomic(target/'plotly.min.js',js)
    for i,spec in enumerate(specs,1):
        name=f'{stage}_{i:02d}';atomic(target/(name+'.plotly.json'),json.dumps(spec)+'\n')
        atomic(target/(name+'.html'),build_page([spec],None,name,'Actual user-run pilot features; no labels or score.',external_js=True))
    page=build_page(specs,js,'Biohub | '+stage+' feature investigation',f'8 families • 128 declared descriptors • {receipt["rows"]} rows • no training fits • no official metric')
    output=target/'gallery.html';atomic(output,page)
    display(HTML(iframe_markup(page)))
    write_json(root/f'outputs/features/{stage}_visual_request.json',{'utc':utc(),'gallery':output.relative_to(root).as_posix(),'sha256':digest(output),'figures':len(specs),'display_request_returned':True,'browser_visual_verification':'pending_user'})
    print(stage.upper()+'_GALLERY_DISPLAY_REQUEST_RETURNED — verify Rendered 8 of 8, save, reopen, and hover.',flush=True)
