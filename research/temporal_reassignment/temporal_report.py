"""Executed notebook 27 using the previously working PNG validator and IPC kernel."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from png_validation import validate_notebook

SENTINEL='BIOHUB_TEMPORAL_REASSIGNMENT_NOTEBOOK_COMPLETE'
NOTEBOOK='27_bidirectional_temporal_reassignment.ipynb'

def build_cells(out,smoke=False):
 import nbformat as nf
 setup=f'''from pathlib import Path
import json,warnings
import plotly.graph_objects as go
from IPython.display import Image,display
OUT=Path({str(out)!r})
def show(fig,height=560):
    fig.update_layout(width=1040,height=height,margin=dict(l=125,r=40,t=95,b=115))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=DeprecationWarning)
        png=fig.to_image(format='png',width=1040,height=height,scale=1)
    display(Image(data=png))
'''
 if smoke:return [nf.v4.new_code_cell(setup+"\nfig=go.Figure(go.Bar(x=['Renderer check'],y=[1]))\nfig.update_layout(title='Rendering preflight only — no modeling result',yaxis_title='Test value')\nshow(fig,460)"),nf.v4.new_code_cell("print('BIOHUB_REPLAY_PLOTLY_SENTINEL')")]
 return [
 nf.v4.new_markdown_cell('# 27 · Bidirectional temporal reassignment\n\n**Official baseline:** 0.947; **last captured leader:** 0.974 (September 21, 17:51 UTC). No leaderboard request is made here.\n\n## Why the previous direction stopped\n\nAll eight cached graph inputs had zero usable two-daughter proposals. The previous retention mechanism was therefore inapplicable; it is not retuned or applied in this experiment. The one completed comparison remains a negative result in Notebook 26.\n\n## New hypothesis\n\nA selected one-to-one edge can still connect the wrong trajectories. The reference motion assignment uses the preceding position and a damped forward velocity. We test whether **agreement from both past and future trajectory segments** can correct a small number of strongly inconsistent identity exchanges. This does not require the discarded dense learned probability matrix.\n\nTemporal-context association is supported as a research direction by Gallusser & Weigert, *Trackastra*, ECCV 2024 / arXiv:2405.15700. This is an independently implemented geometric experiment, **not** a reproduction of Trackastra or the leaderboard leader.\n\n## Frozen protocol\n\nFive complete cached development movies, selected before this experiment, span two embryos. They were previously used by the public notebook; pretrained-model independence is unverified. This is exploratory screening, not independent validation or an official score.'),
 nf.v4.new_code_cell(setup+'''\nr=json.loads((OUT/'result.json').read_text())
assert r['status']=='complete' and len(r['measured_movies'])==5
assert r['model_fits']==r['main_network_inference_calls']==r['network_requests']==0
print('Measured movies:',', '.join(r['measured_movies']))
print('Decision:',r['decision'])
print('New frozen postprocessing runs:',r['new_frozen_postprocessing_runs'])
print('Reused frozen results:',r['reused_frozen'])
'''),
 nf.v4.new_markdown_cell('## What changes, and what does not\n\nFor each eligible edge, three detections before and three after the boundary estimate two-step velocities. Costs use **unsmoothed cached detection coordinates**. An exchange must improve both links by at least 0.75 µm, the total by at least 2 µm, and halve the total discrepancy. Each new forward and backward residual must be at most 2 µm; context acceleration must be at most 1.5 µm. The new step is limited to 10 µm. These are fixed screening settings, not fitted parameters.\n\nProposals use prediction-only information. Labels enter only after the candidate graph is saved. Accepted exchanges have disjoint six-frame contexts. Node coordinates, node counts, edge counts, individual node degrees, and existing division edges are invariant. New links do not inherit a fabricated learned probability. The original full postprocessing and the exact metric remain unchanged.'),
 nf.v4.new_code_cell('''fig=go.Figure()
labels=['Five cached movies']+[f'Embryo {e}' for e in sorted(r['by_embryo'])]
for arm,label in [('frozen','Frozen reference'),('temporal_reassignment','Bidirectional reassignment')]:
    v=[r['summaries'][arm]['score']]+[r['by_embryo'][e][arm]['score'] for e in sorted(r['by_embryo'])]
    fig.add_bar(name=label,x=labels,y=v,text=[f'{x:.6f}' for x in v],textposition='outside')
fig.update_layout(title='Does temporal reassignment improve the complete local metric?',yaxis_title='Pinned organizer-formula score',barmode='group')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Error trade-off\n\nThe metric uses sparse annotation-aware edge counts and topology-aware division scoring, not ordinary accuracy. A geometry improvement is not automatically a biological improvement. The frozen historical proxy is used only to check replay parity.'),
 nf.v4.new_code_cell('''fig=go.Figure()
for arm,label in [('frozen','Frozen reference'),('temporal_reassignment','Bidirectional reassignment')]:
    rows=[x[arm]['metric'] for x in r['per_movie']]
    fig.add_bar(name=label,x=['Wrong edges','Missed edges','False divisions','Missed divisions'],y=[sum(v[k] for v in rows) for k in ['edge_fp','edge_fn','division_fp','division_fn']])
fig.update_layout(title='Are tracking errors reduced without damaging divisions?',yaxis_title='Annotated error counts',barmode='group')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Where the candidate acts\n\nSwapping two links is one intervention, not two recovered cells. These counts describe actions; only the evaluator above determines their benefit.'),
 nf.v4.new_code_cell('''a=[x['temporal_reassignment']['audit'] for x in r['per_movie']]
fig=go.Figure(go.Bar(y=r['measured_movies'],x=[x['accepted_swaps'] for x in a],orientation='h'))
fig.update_layout(title='Accepted six-frame identity exchanges by movie',xaxis_title='Two-link exchanges',yaxis_title='Cached development movie')
show(fig,600)
'''),
 nf.v4.new_markdown_cell('## Decision\n\nScreening requires at least 0.002 local combined-score improvement, adjusted-edge loss no worse than 0.0005, no loss for either embryo, no loss of division true positives or increase in false divisions, and at least one intervention. A passing screen earns independent confirmation only. A negative result retains the 0.947 reference; no sweep or automatic submission follows.\n\nNo training or main detector/transformer inference was run. Cached frozen graphs are reused; missing frozen graphs may require DeepCenter postprocessing on the AWS GPU. All downloads and external actions are disabled. Results are not a claim to have closed the official 0.027 gap.'),
 nf.v4.new_code_cell("print(json.dumps({k:r[k] for k in ['decision','score_delta','adjusted_edge_delta','embryo_deltas','accepted_swaps','screen_passed','approved_for_submission']},indent=2))\nprint('"+SENTINEL+"')")]

def execute(request):
 import nbformat
 from nbclient import NotebookClient
 from traitlets.config import Config
 out=Path(request['out']);root=Path(request['root']);smoke=request.get('smoke',False)
 nb=nbformat.v4.new_notebook(cells=build_cells(out,smoke))
 nb.metadata.kernelspec={'name':'biohub-cell-tracking','display_name':'Python (Biohub Cell Tracking)','language':'python'}
 path=out/('plotly_preflight.ipynb' if smoke else NOTEBOOK)
 for c in nb.cells:
    if c.cell_type=='code':compile(c.source,str(path),'exec')
 with tempfile.TemporaryDirectory(prefix='bht-',dir='/tmp') as d:
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
