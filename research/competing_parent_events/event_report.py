"""Notebook 29: competing-parent event learning, exploratory embryo-disjoint screen."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from png_validation import validate_notebook
NOTEBOOK='29_competing_parent_division_model.ipynb'
SENTINEL='BIOHUB_COMPETING_PARENT_NOTEBOOK_COMPLETE'

def build_cells(out,smoke=False):
 import nbformat as nf
 setup=f'''from pathlib import Path
import json,warnings
import plotly.graph_objects as go
from IPython.display import Image,display
OUT=Path({str(out)!r})
def show(fig,height=580):
    fig.update_layout(width=1040,height=height,margin=dict(l=100,r=55,t=100,b=110))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=DeprecationWarning)
        png=fig.to_image(format='png',width=1040,height=height,scale=1)
    display(Image(data=png))
'''
 if smoke:return [nf.v4.new_code_cell(setup+"\nfig=go.Figure(go.Bar(x=['Renderer check'],y=[1]))\nfig.update_layout(title='Renderer preflight only — not an experiment result',yaxis_title='Test value')\nshow(fig,460)"),nf.v4.new_code_cell("print('BIOHUB_REPLAY_PLOTLY_SENTINEL')")]
 return [
 nf.v4.new_markdown_cell('# 29 · Competing-parent division model\n\n## Objective and evidence\n\nThe exact error audit found four missed divisions with parent and daughter support but no fork. Their missing daughter links were already assigned to other parents. The earlier add-only retention rule could not correct that case. This experiment **generates new division alternatives and explicitly compares a division with two independent continuations**.\n\nThe scored public reference remains frozen: official snapshot **0.947**, leader snapshot **0.974** (September 21, 17:51 UTC). These are not the five-movie development scores. No Kaggle action occurs.\n\nThis is a new small graph-feature model, not a claimed HOCT, Trackastra, or winning-method reproduction. The higher-order literature motivates evaluating competing lineage hypotheses rather than assuming that the strongest independent edge is sufficient.'),
 nf.v4.new_code_cell(setup+'''\nr=json.loads((OUT/'result.json').read_text())
assert r['status']=='complete' and r['approved_for_submission'] is False
contract=json.loads((OUT/'experiment_contract.json').read_text())
coverage=json.loads((OUT/'coverage.json').read_text())
print('Cohort:',', '.join(r['measured_movies']))
print('Frozen control:',format(r['baseline']['score'],'.9f'))
print('Fixed feature count:',len(contract['features']))
print('New fits:',r['new_fits'],'Reused fits:',r['reused_fits'])
print('Decision:',r['decision'])
'''),
 nf.v4.new_markdown_cell('## Hypotheses are generated before annotations\n\nFor each unary parent, examine up to four alternate next-frame detections within 15 µm. An alternative may already have a different unary parent. A transaction removes that competing incoming edge and adds the second daughter to the proposed dividing parent. Existing division edges are protected; nodes and coordinates remain unchanged.\n\nFeatures use relative geometry, parent/daughter motion, trajectory support, and the competing incoming association. Movie IDs, embryo IDs, absolute time, GT coordinates, and annotation-derived identities are excluded. All five feature arrays are saved before opening annotations. Radius and feature design are fixed for this screen, not tuned on its result.\n\nThe following **label-assisted coverage** check is not a model score: it asks whether the generated alternatives include an annotated event. Unassessable proposals are ignored during learning, not made negative.'),
 nf.v4.new_code_cell('''labels=[c['stem'] for c in coverage]
fig=go.Figure()
fig.add_bar(name='Annotated divisions',x=labels,y=[c['annotated_divisions'] for c in coverage])
fig.add_bar(name='Already recovered',x=labels,y=[c['baseline_recovered'] for c in coverage])
fig.add_bar(name='Missed with a new positive proposal',x=labels,y=[c['missed_divisions_with_positive_new_proposal'] for c in coverage])
fig.update_layout(title='Does the proposal population contain recoverable divisions?',barmode='group',xaxis_title='Previously used development movie',yaxis_title='Annotation-assisted event count')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Two embryo-disjoint learning tasks\n\nTrain on embryo 44b6 and predict embryo 6bba; then reverse. Fit scaling, model coefficients, and the selection threshold on the training embryo only. Positive proposals are weighted by distinct division event so duplicate temporal variants do not count as independent events. The regularized logistic score is **not a calibrated probability**. The threshold exceeds the largest training-negative score and is at least 0.5.\n\nAt least two distinct annotated positive division events and twenty known negative proposals are required in each training embryo. Otherwise the run stops before fitting. There are at most two fits, no parameter sweep, and at most 32 accepted nonconflicting events per movie. This budget is a safety limit, not a biological assumption.\n\nAll held-out graphs are frozen before candidate scoring. The original metric is unchanged. Proposal-level training labels are conservative proxies using exact baseline division-window matches and known lineage components; final acceptance uses full graph evaluation.'),
 nf.v4.new_code_cell('''fig=go.Figure()
fig.add_bar(name='Frozen control',x=['Combined local score','Adjusted edge Jaccard'],y=[r['baseline']['score'],r['baseline']['adj_edge_jaccard']])
if r.get('candidate') is not None:
    fig.add_bar(name='Embryo-disjoint event model',x=['Combined local score','Adjusted edge Jaccard'],y=[r['candidate']['score'],r['candidate']['adj_edge_jaccard']])
    print('Actual score delta:',r['score_delta'])
else:
    print('No model comparison: training-readiness gate was not met.')
fig.update_layout(title='Matched five-movie comparison — NOT leaderboard performance',barmode='group',yaxis_title='Pinned local metric',xaxis_title='Evaluation component')
show(fig)
'''),
 nf.v4.new_markdown_cell('## What changed in the tracking graph?\n\nNew divisions can improve division recall but also damage correct incoming associations. Both effects are measured; a rise in probability score alone is not an improvement. The original nodes, coordinates, and reference files remain unchanged. All new graphs are experimental and kept out of the competition submission path.'),
 nf.v4.new_code_cell('''fig=go.Figure()
if r.get('candidate') is not None:
    keys=['division_tp','division_fp','division_fn']
    names=['Correct divisions','Incorrect divisions','Missed divisions']
    fig.add_bar(name='Frozen control',x=names,y=[r['baseline'][k] for k in keys])
    fig.add_bar(name='Event model',x=names,y=[r['candidate'][k] for k in keys])
    print('Accepted new forks:',r['accepted_new_forks'])
    print('Reassigned children:',r['reassigned_children'])
    for row in r['embryos']:print('Embryo',row['embryo'],'local score delta',row['delta'])
else:
    readiness=json.loads((OUT/'training_readiness.json').read_text())
    fig.add_bar(name='Distinct training divisions',x=[x['heldout_embryo'] for x in readiness],y=[x['distinct_training_divisions'] for x in readiness])
fig.update_layout(title='Division outcomes and readiness are counted explicitly',barmode='group',yaxis_title='Count')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Decision and limitations\n\nPromotion requires local score gain ≥0.002, adjusted-edge loss ≤0.0005, no embryo-level score loss, no loss of division true positives, and no additional division false positives. Otherwise reject the fixed model. No automatic submission or follow-on run occurs.\n\n**Only six annotated division events exist in these two repeatedly examined development embryos.** Cross-embryo fitting prevents direct training-row leakage but does not undo prior researcher exposure or establish pretrained-checkpoint independence. A positive screen earns independent confirmation, not a claim that the leader was beaten.\n\nResearch context: Trackastra (Gallusser and Weigert, 2024), Higher-Order Cell Tracking Transformer (Bragantini, Theodoro and Royer, July 2026). Their full architectures and published benchmark gains are not reproduced or claimed here. Pin: RoyerLab evaluator commit 075fc5f5a52d11077f9dc2b074644618f26939e2. No new pretrained weights are used.'),
 nf.v4.new_code_cell("print('Decision:',r['decision'])\nprint('Approved for submission:',r['approved_for_submission'])\nprint('Preserved reference; no downloads, GPU inference or Kaggle actions.')\nprint('"+SENTINEL+"')")]
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
