"""Notebook 28: exact error attribution, not a new predictive result."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from png_validation import validate_notebook
NOTEBOOK='28_exact_tracking_error_budget.ipynb'
SENTINEL='BIOHUB_EXACT_ERROR_BUDGET_NOTEBOOK_COMPLETE'

def build_cells(out,smoke=False):
 import nbformat as nf
 setup=f'''from pathlib import Path
import json,warnings
import plotly.graph_objects as go
from IPython.display import Image,display
OUT=Path({str(out)!r})
def show(fig,height=580):
    fig.update_layout(width=1040,height=height,margin=dict(l=310,r=60,t=100,b=115))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=DeprecationWarning)
        png=fig.to_image(format='png',width=1040,height=height,scale=1)
    display(Image(data=png))
'''
 if smoke:return [nf.v4.new_code_cell(setup+"\nfig=go.Figure(go.Bar(x=['Renderer check'],y=[1]))\nfig.update_layout(title='Renderer preflight only — not an experiment result',yaxis_title='Test value')\nshow(fig,460)"),nf.v4.new_code_cell("print('BIOHUB_REPLAY_PLOTLY_SENTINEL')")]
 return [
 nf.v4.new_markdown_cell('# 28 · Exact tracking error budget\n\n## Why stop the previous changes?\n\nNotebook 26 established zero eligible learned daughter pairs. Notebook 27 completed five full-movie comparisons but accepted zero temporal exchanges. Neither changed the score. We now use the **unchanged, saved final predictions**, rather than trying another unmeasured parameter adjustment.\n\n**Official baseline snapshot: 0.947; leader snapshot: 0.974, September 21 at 17:51 UTC.** No fresh leaderboard query is made. The five-movie score is a different evaluation setting.\n\n## Objective\n\nPartition every counted false positive and false negative under the pinned organizer evaluator, distinguish raw detection coverage from postprocessing losses, and identify each missed division using the evaluator’s window-specific matching. This is ground-truth-assisted diagnosis, not inference, training, or an achieved improvement. No prediction graph is changed or written.'),
 nf.v4.new_code_cell(setup+'''\nr=json.loads((OUT/'result.json').read_text())
assert r['status']=='complete' and len(r['measured_movies'])==5
assert r['prediction_changes']==r['model_fits']==r['network_requests']==r['gpu_inference_calls']==0
print('Frozen local control:',format(r['baseline_summary']['score'],'.9f'))
print('Verified counts:',r['counts'])
print('Movies:',', '.join(r['measured_movies']))
print('New audits:',r['new_movie_audits'],'Reused audits:',r['reused_movies'])
'''),
 nf.v4.new_markdown_cell('## Reconciliation comes before interpretation\n\nOriginal graph checksums and ground-truth metadata/chunks are verified. Counts and scalar metrics must reproduce all five saved controls exactly, within a 1e-10 numeric tolerance. Each counted false-negative edge appears in exactly one category. False-positive labels come from the evaluator’s sparse-annotation validity mask: **an unmatched prediction is not automatically a false positive**.\n\nA missing endpoint means unmatched under the 7 µm one-to-one assignment, not proof that no nucleus exists. The separate raw-to-final coverage comparison is diagnostic; smoothing or competing assignments can change matching even when a prediction ID survives.'),
 nf.v4.new_code_cell('''labels=[];values=[]
for group,key in [('Missed','fn_categories'),('False','fp_categories')]:
    for label,value in sorted(r[key].items(),key=lambda x:x[1]):
        labels.append(group+': '+label.replace('_',' '));values.append(value)
fig=go.Figure(go.Bar(y=labels,x=values,orientation='h'))
fig.update_layout(title='Which edge error mechanisms explain the frozen control?',xaxis_title='Exactly reconciled annotated edge count',yaxis_title='')
show(fig,max(580,70*len(labels)))
'''),
 nf.v4.new_markdown_cell('## Division failures use event-specific matching\n\nThe organizer allows a fork one frame early or late and requires two distinct daughter branches. A full-graph edge match must not be substituted for each division’s local parent/daughter matching. Event IDs, supporting predictions, local forks, and rejected topology are saved in the diagnostic records. These few annotated divisions do not establish generalization across embryos.'),
 nf.v4.new_code_cell('''labels=[k.replace('_',' ') for k in r['missed_division_categories']]
vals=list(r['missed_division_categories'].values())
if not labels:labels=['No missed divisions'];vals=[0]
fig=go.Figure(go.Bar(y=labels,x=vals,orientation='h'))
fig.update_layout(title='Why were annotated division events missed?',xaxis_title='Missed events under exact window matching',yaxis_title='')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Count sensitivities: priorities, not improvements\n\nThe following arithmetic asks how the local score would change if one class of counted errors were removed while other counts and node totals stayed fixed. **These use ground truth, are not model results or forecast gains, overlap, and must not be added together.** Recovering missed detections can change node totals and matching; actual improvements may be far smaller or negative. The analysis does not generate an oracle prediction or submission.'),
 nf.v4.new_code_cell('''items=list(reversed(r['count_sensitivities']))
labels=[x['scenario'].replace('recover_missed_edges__','Missed: ').replace('_',' ') for x in items]
fig=go.Figure(go.Bar(y=labels,x=[x['delta_from_local_baseline'] for x in items],orientation='h'))
fig.update_layout(title='Hypothetical count sensitivity — NOT measured model gains',xaxis_title='Local score change with other terms fixed (non-additive)',yaxis_title='')
show(fig,max(600,65*len(items)))
'''),
 nf.v4.new_markdown_cell('## Decision and reproducibility\n\nThe next modeling hypothesis must address an error class actually represented here and demonstrate an actionable input before any broader compute. No threshold sweep, data download, training, DeepCenter call, detector call, or submission occurs in this audit. CPU-only graph analysis reuses the five saved reference predictions.\n\nThis cohort has been used repeatedly and comprises two embryos; pretrained-model independence is not established. Any later positive screen requires independent validation. Annotation-dependent files stay under diagnostic outputs and are not inference inputs.\n\nPinned source: RoyerLab kaggle-cell-tracking-competition, commit 075fc5f5a52d11077f9dc2b074644618f26939e2; `metrics.py` and `division_metrics.py` are unchanged.'),
 nf.v4.new_code_cell("print('Decision:',r['decision'])\nprint('No prediction changes; not approved for submission:',r['approved_for_submission'])\nprint('Leading diagnostic targets:',json.dumps(r['ranked_research_targets'][:3],indent=2))\nprint('"+SENTINEL+"')")]

def execute(request):
 import nbformat
 from nbclient import NotebookClient
 from traitlets.config import Config
 out=Path(request['out']);root=Path(request['root']);smoke=request.get('smoke',False)
 nb=nbformat.v4.new_notebook(cells=build_cells(out,smoke));nb.metadata.kernelspec={'name':'biohub-cell-tracking','display_name':'Python (Biohub Cell Tracking)','language':'python'}
 path=out/('plotly_preflight.ipynb' if smoke else NOTEBOOK)
 for c in nb.cells:
    if c.cell_type=='code':compile(c.source,str(path),'exec')
 with tempfile.TemporaryDirectory(prefix='bhe-',dir='/tmp') as d:
    cfg=Config();cfg.KernelManager.transport='ipc';cfg.KernelManager.ip=str(Path(d)/'k')
    client=NotebookClient(nb,timeout=90,startup_timeout=45,kernel_name='biohub-cell-tracking',resources={'metadata':{'path':str(root)}},allow_errors=False,record_timing=True,config=cfg)
    if client.create_kernel_manager().transport!='ipc':raise RuntimeError('IPC_NOT_ENABLED')
    try:nb=client.execute()
    finally:nbformat.write(nb,path)
 receipt=validate_notebook(path,1 if smoke else 3,'BIOHUB_REPLAY_PLOTLY_SENTINEL' if smoke else SENTINEL)
 (out/('plotly_preflight_receipt.json' if smoke else 'notebook_verification.json')).write_text(json.dumps(receipt,indent=2))
 print('NOTEBOOK_VERIFIED',str(path),receipt,flush=True)

if __name__=='__main__':
 import sys
 execute(json.loads(Path(sys.argv[1]).read_text()))
