"""Notebook 30: image-appearance versus geometry association ablation."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from png_validation import validate_notebook
NOTEBOOK='30_image_appearance_association.ipynb'
SENTINEL='BIOHUB_IMAGE_ASSOCIATION_NOTEBOOK_COMPLETE'

def build_cells(out,smoke=False):
 import nbformat as nf
 setup=f'''from pathlib import Path
import json,warnings
import plotly.graph_objects as go
from IPython.display import Image,display
OUT=Path({str(out)!r})
def show(fig,height=580):
    fig.update_layout(width=1040,height=height,margin=dict(l=110,r=55,t=100,b=115))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=DeprecationWarning)
        png=fig.to_image(format='png',width=1040,height=height,scale=1)
    display(Image(data=png))
'''
 if smoke:return [nf.v4.new_code_cell(setup+"\nfig=go.Figure(go.Bar(x=['Renderer check'],y=[1]))\nfig.update_layout(title='Rendering preflight only — not a modeling result',yaxis_title='Test value')\nshow(fig,460)"),nf.v4.new_code_cell("print('BIOHUB_REPLAY_PLOTLY_SENTINEL')")]
 return [
 nf.v4.new_markdown_cell('# 30 · Image appearance and learned continuation association\n\n## Decision from the previous experiment\n\nThe geometry-only division-event classifier produced 108 new forks but no net score improvement. Its cross-embryo training folds had only two or three distinct positive division events. The model is rejected and is not used here.\n\n**Objective:** test continuation decisions with much more annotated edge supervision, and measure whether local microscopy appearance complements trajectory geometry. The released 0.947 detector/reference predictions stay frozen. We do not retrain an image backbone, introduce external pretrained weights, or claim reproduction of the 0.974 leader.\n\nOfficial snapshots (September 21, 17:51 UTC): 0.947 reference, 0.974 leader. These are distinct from all local metrics below.'),
 nf.v4.new_code_cell(setup+'''
r=json.loads((OUT/'result.json').read_text())
coverage=json.loads((OUT/'coverage.json').read_text())
contract=json.loads((OUT/'experiment_contract.json').read_text())
assert r['status']=='complete' and not r['approved_for_submission']
print('Decision:',r['decision'])
print('Completed real-data fits:',r['new_fits'],'Reused:',r['reused_fits'])
print('Geometry features:',len(contract['geometry_features']),'Appearance pair features:',len(contract['appearance_features']))
print('Baseline local combined score:',format(r['baseline']['score'],'.9f'))
'''),
 nf.v4.new_markdown_cell('## Fixed candidates and local image representation\n\nAll candidate pairs and image descriptors are generated before opening annotations. Each current unary source considers its incumbent target and up to six nearby targets within 12 µm. Only targets of existing eligible unary edges are considered; unmatched detections are not invented or removed.\n\nThe 20 geometric features describe relative displacements, forward/backward motion, history support, and neighborhood density. At each candidate endpoint, a 5×5×5 stencil samples the cached image at fixed physical offsets from −3 to +3 µm. Nine pair features summarize intensity-profile similarity, contrast, local extent, and boundary support. These are **hand-designed image descriptors**, not deep embeddings. No absolute time, position, movie identifier, embryo identifier, or annotation identity enters the feature vector.\n\nImage frames are read once, cached per frame with checksums, and reused. The standard baseline graphs are not recomputed.'),
 nf.v4.new_code_cell('''fig=go.Figure()
fig.add_bar(name='Positive candidate links',x=[c['stem'] for c in coverage],y=[c['positive_pairs'] for c in coverage])
fig.add_bar(name='Evaluable negative links',x=[c['stem'] for c in coverage],y=[c['negative_pairs'] for c in coverage])
fig.update_layout(title='Measured edge supervision — not six division events',xaxis_title='Development movie',yaxis_title='Sparse-supervision candidate count',barmode='group')
show(fig)
print('Unassessable pairs are excluded from learning:',sum(c['ignored_pairs'] for c in coverage))
'''),
 nf.v4.new_markdown_cell('## Matched two-embryo ablation\n\nFor each arm, train on 44b6 and evaluate 6bba, then reverse: four fixed model fits total. A histogram gradient-boosted classifier uses 100 trees, learning rate 0.05, at most seven leaves/depth three, minimum 30 samples per leaf, and L2 regularization 1. No random validation split, class reweighting, threshold search, or held-out-data early stopping is used. Training requires at least 100 positive and 200 evaluable negative links per fold.\n\nThe matched arms are geometry only and geometry plus image appearance. Sparse labels follow the evaluator’s endpoint rules; every incumbent label is checked against the pinned scorer. Unknown pairs are never blanket negatives. The classifier outputs ranking scores; external probability calibration is not claimed.\n\nA full frame-wise assignment maximizes learned log odds plus a fixed log(2) bonus for incumbent edges. Entire exchange cycles can be accepted only if all replacement scores are at least 0.5 and their aggregate gain exceeds that bonus. This goes beyond the previous two-link hand-set temporal rule while preserving the original system as a preference.'),
 nf.v4.new_code_cell('''labels=['Frozen reference']+list(r.get('arms',{}))
vals=[r['baseline']['score']]+[v['metric']['score'] for v in r.get('arms',{}).values()]
fig=go.Figure(go.Bar(x=labels,y=vals,text=[format(v,'.6f') for v in vals],textposition='outside'))
fig.update_layout(title='Full pinned local metric — not the public leaderboard',xaxis_title='Matched association system',yaxis_title='Combined local score')
show(fig)
for arm,v in r.get('arms',{}).items():
    print(arm,'score delta',v['score_delta'],'changed edges',v['changed_edges'],'screen passed',v['screen_passed'])
    for e in v['embryos']:print(' ',e['embryo'],'delta',e['delta'])
print('Increment from image appearance:',r.get('appearance_increment'))
'''),
 nf.v4.new_markdown_cell('## Did changed decisions affect annotated errors?\n\nThe preceding event model changed many edges with no net metric gain. This report explicitly counts newly correct links, lost correct links, and changes whose old/new pairs are both unassessable. These post-hoc diagnostics cannot feed back into the current inference. All held-out prediction graphs were saved before any candidate metric was calculated.\n\nNodes, coordinates, edge counts, and every node’s incoming/outgoing degree are preserved. Existing divisions and their two-hop neighborhoods are immutable. This candidate cannot create missing divisions or detections; it targets recoverable continuation errors.'),
 nf.v4.new_code_cell('''fig=go.Figure()
mods=r.get('modifications',[])
for field,name in [('new_true_positive_edges','New correct links'),('lost_true_positive_edges','Lost correct links'),('both_unassessable','Both unassessable')]:
    fig.add_bar(name=name,x=['geometry','geometry_appearance'],y=[sum(v[field] for v in mods if v['arm']==a) for a in ['geometry','geometry_appearance']])
fig.update_layout(title='Intervention outcomes, with sparse annotations kept explicit',xaxis_title='Held-out prediction arm',yaxis_title='Changed-edge count',barmode='group')
show(fig)
'''),
 nf.v4.new_markdown_cell('## Promotion, limitations, and next decision\n\nA screen must improve the combined local metric by at least 0.002, lose no more than 0.0005 adjusted edge Jaccard, avoid score loss on either embryo, and avoid losing division true positives or increasing division false positives. The appearance arm must also be compared directly with geometry only; merely finishing or generating many exchanges is not progress. No automatic submission is allowed.\n\nThese five movies and two embryos have been repeatedly inspected. Cross-embryo fits prevent direct row leakage but do not undo researcher exposure, correlated biological observations, or unverified public-checkpoint training overlap. A positive screen requires independent confirmation before another official notebook submission. A negative screen rejects these fixed settings; it does not establish that richer learned visual representations cannot work.\n\nReferences: frozen RoyerLab metric commit 075fc5f5a52d11077f9dc2b074644618f26939e2; scikit-learn histogram boosting documentation; SciPy linear_sum_assignment documentation. Trackastra, arXiv:2405.15700, motivates contextual association, but its transformer is not reproduced by this experiment.'),
 nf.v4.new_code_cell("print('Final decision:',r['decision'])\nprint('Approved for submission:',r['approved_for_submission'])\nprint('AWS only; no downloads, installations, pretrained-network inference, or Kaggle actions.')\nprint('"+SENTINEL+"')")]

def execute(request):
 import nbformat
 from nbclient import NotebookClient
 from traitlets.config import Config
 out=Path(request['out']);root=Path(request['root']);smoke=request.get('smoke',False)
 nb=nbformat.v4.new_notebook(cells=build_cells(out,smoke));nb.metadata.kernelspec={'name':'biohub-cell-tracking','display_name':'Python (Biohub Cell Tracking)','language':'python'}
 path=out/('plotly_preflight.ipynb' if smoke else NOTEBOOK)
 for c in nb.cells:
  if c.cell_type=='code':compile(c.source,str(path),'exec')
 with tempfile.TemporaryDirectory(prefix='bha-',dir='/tmp') as d:
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
