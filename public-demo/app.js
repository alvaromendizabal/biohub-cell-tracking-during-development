/* Offline synthetic lineage explorer. Annotation identities are never predictions. */
(function (root) {
  'use strict';
  const edgeKey = (source, target) => `${source}:${target}`;
  const clampFrame = (frame, count) => Math.max(0, Math.min(count - 1, Math.round(Number(frame) || 0)));
  function lineageFor(id, edges) {
    const forward = new Map(), reverse = new Map();
    for (const edge of edges) {
      if (!forward.has(edge.source)) forward.set(edge.source, []);
      if (!reverse.has(edge.target)) reverse.set(edge.target, []);
      forward.get(edge.source).push(edge.target); reverse.get(edge.target).push(edge.source);
    }
    function walk(map) {
      const seen = new Set([id]), pending = [id];
      while (pending.length) for (const node of map.get(pending.pop()) || []) if (!seen.has(node)) { seen.add(node); pending.push(node); }
      seen.delete(id); return seen;
    }
    const ancestors = walk(reverse), descendants = walk(forward);
    return { ancestors, descendants, all: new Set([id, ...ancestors, ...descendants]) };
  }
  function agreement(predicted, expected) {
    const tp = [...predicted].filter(key => expected.has(key)).length;
    const fp = predicted.size - tp, fn = expected.size - tp;
    return {true_positive: tp, false_positive: fp, false_negative: fn,
      precision: tp + fp ? tp / (tp + fp) : 0, recall: tp + fn ? tp / (tp + fn) : 0,
      f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0,
      jaccard: tp + fp + fn ? tp / (tp + fp + fn) : 0};
  }
  function divisionSet(edges) {
    const children = new Map();
    for (const edge of edges) { if (!children.has(edge.source)) children.set(edge.source, new Set()); children.get(edge.source).add(edge.target); }
    return new Set([...children].filter(([, targets]) => targets.size === 2).map(([source, targets]) => `${source}:${[...targets].sort((a,b) => a-b).join(',')}`));
  }
  function measureSolution(data, edges, parameters) {
    const nodes = new Map(data.nodes.map(n => [n.id, n])), predicted = new Set(edges.map(e => edgeKey(e.source,e.target)));
    const expected = new Set(data.truth_edges.map(e => edgeKey(e.source,e.target))), candidates = new Set(data.candidates.map(e => edgeKey(e.source,e.target)));
    const incoming = new Map(), outgoing = new Map(), adjacency = new Map(data.nodes.map(n => [n.id,new Set()]));
    const issues = [];
    let known = true;
    for (const edge of edges) {
      incoming.set(edge.target,(incoming.get(edge.target)||0)+1);
      if (!outgoing.has(edge.source)) outgoing.set(edge.source,[]);
      outgoing.get(edge.source).push(edge.target);
      if (!nodes.has(edge.source) || !nodes.has(edge.target)) {known=false; continue;}
      adjacency.get(edge.source).add(edge.target); adjacency.get(edge.target).add(edge.source);
    }
    if (!known) throw new Error('Cannot evaluate an unknown graph endpoint.');
    const maxIn = Math.max(0,...incoming.values()), maxOut = Math.max(0,...[...outgoing.values()].map(x => x.length));
    const forward = edges.every(e => nodes.get(e.target).t-nodes.get(e.source).t >= 1 && nodes.get(e.target).t-nodes.get(e.source).t <= parameters.max_gap);
    const allowed = [...predicted].every(e => candidates.has(e));
    if (predicted.size!==edges.length) issues.push('duplicate edges');
    if (maxIn>1) issues.push('multiple parents');
    if (maxOut>parameters.max_children) issues.push('child capacity exceeded');
    if (!forward) issues.push('invalid time direction or gap');
    if ([...outgoing.values()].some(ids => ids.length===2 && new Set(ids.map(id=>nodes.get(id).t)).size>1)) issues.push('division children occupy different frames');
    if (!allowed) issues.push('selected edge is not a candidate');
    const seen = new Set(); let components = 0;
    for (const id of nodes.keys()) if (!seen.has(id)) { components++; const pending=[id]; while(pending.length) {const current=pending.pop(); if(seen.has(current))continue;seen.add(current);pending.push(...adjacency.get(current));} }
    const edge = agreement(predicted,expected), division = agreement(divisionSet(edges),divisionSet(data.truth_edges));
    const metrics = {true_positive:edge.true_positive,false_positive:edge.false_positive,false_negative:edge.false_negative};
    for (const k of ['precision','recall','f1','jaccard']) metrics[`edge_${k}`]=edge[k];
    for (const k of ['true_positive','false_positive','false_negative','precision','recall','f1']) metrics[`division_${k}`]=division[k];
    Object.assign(metrics,{selected_edges:predicted.size,gap_edges:[...predicted].filter(key=>{const [s,t]=key.split(':').map(Number);return nodes.get(t).t-nodes.get(s).t>1;}).length,division_events:divisionSet(edges).size,components});
    return {metrics,checks:{valid_topology:issues.length===0,issues:issues.sort(),max_in_degree:maxIn,max_out_degree:maxOut,forward_time_only:forward,all_edges_are_candidates:allowed}};
  }
  function exportResult(data, state, solution) {
    return {schema:'biohub-public-explorer-result-v1',evidence_type:'SYNTHETIC_ONLY',
      provenance:data.provenance,config:data.config,summary:data.summary,
      computation:solution.id==='live'?'Browser graph optimization on authored synthetic candidates':'Precomputed Python solver output; browser inspection only',
      metric_scope:'Whole authored synthetic movie; no biological or official competition performance',
      solution, nodes:data.nodes,candidates:data.candidates,truth_edges:data.truth_edges,
      review_state:{frame_index:state.frame,projection:state.projection,selected_detection_id:state.selectedId,overlays:{...state.overlays}}};
  }
  function init(data, doc, win) {
    if (!data || data.provenance.evidence_type!=='SYNTHETIC_ONLY' || !data.frames.length || !data.solutions.length) throw new Error('The public synthetic fixture is unavailable or invalid.');
    const $ = id => { const el=doc.getElementById(id); if(!el)throw new Error(`Missing explorer control: ${id}`); return el; };
    const canvas=$('movie-canvas'), ctx=canvas.getContext('2d');
    if(!ctx)throw new Error('This viewer needs Canvas 2D support.');
    const nodes=new Map(data.nodes.map(n=>[n.id,n])), truth=new Set(data.truth_edges.map(e=>edgeKey(e.source,e.target)));
    const solutions=[...data.solutions];
    const state={frame:0,policyId:'balanced',selectedId:null,projection:'xy',playing:false,speed:1,eventFilter:'all',overlays:{ids:false,links:true,divisions:true,errors:true}};
    let hitPoints=[],timer=null,disposed=false;
    const cleanup=[];
    const on=(element,type,handler)=>{element.addEventListener(type,handler);cleanup.push(()=>element.removeEventListener(type,handler));};
    const element=(tag,text,className)=>{const el=doc.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;};
    const svgElement=(tag,attrs)=>{const el=doc.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs||{}))el.setAttribute(k,String(v));return el;};
    const current=()=>solutions.find(s=>s.id===state.policyId);
    const fmt=value=>Number(value).toFixed(3);
    const images=data.frames.map(frame=>{const img=new win.Image();img.onload=()=>{if(!disposed && state.frame===frame.index)draw();};img.onerror=()=>{if(state.frame===frame.index){$('load-message').hidden=false;$('load-message').textContent='The bundled synthetic frame could not be decoded. Try the XZ coordinate view.';}};img.src=frame.image_data_uri;return img;});
    const allZ=data.frames.flatMap(f=>f.cells.map(n=>n.z)), minZ=Math.min(...allZ)-1,maxZ=Math.max(...allZ)+1;
    function draw() {
      if(disposed)return;
      const rect=canvas.getBoundingClientRect(), w=Math.max(1,rect.width),h=Math.max(1,rect.height),dpr=Math.min(2,win.devicePixelRatio||1);
      if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
      ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1522';ctx.fillRect(0,0,w,h);
      const imageWidth=data.config.width,imageHeight=data.config.height,scale=Math.min((w-44)/imageWidth,(h-64)/imageHeight),iw=imageWidth*scale,ih=imageHeight*scale,ox=(w-iw)/2,oy=(h-ih)/2-5;
      const project=n=>({x:ox+n.x*scale,y:state.projection==='xy'?oy+n.y*scale:oy+ih-(n.z-minZ)/(maxZ-minZ)*ih});
      ctx.strokeStyle='#233c45';ctx.lineWidth=.5;
      for(let x=0;x<=imageWidth;x+=20){ctx.beginPath();ctx.moveTo(ox+x*scale,oy);ctx.lineTo(ox+x*scale,oy+ih);ctx.stroke();}
      for(let y=0;y<=imageHeight;y+=16){ctx.beginPath();ctx.moveTo(ox,oy+y*scale);ctx.lineTo(ox+iw,oy+y*scale);ctx.stroke();}
      if(state.projection==='xy'){
        const img=images[state.frame];if(img.complete&&img.naturalWidth){ctx.globalAlpha=.92;ctx.imageSmoothingEnabled=true;ctx.drawImage(img,ox,oy,iw,ih);ctx.globalAlpha=1;$('load-message').hidden=true;}
      }else{
        $('load-message').hidden=true;
        for(const n of data.frames[state.frame].cells){const p=project(n),r=Math.max(5,n.radius*scale*1.6),grad=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r);grad.addColorStop(0,`rgba(182,229,214,${Math.min(.95,n.intensity)})`);grad.addColorStop(.28,'rgba(91,167,159,.5)');grad.addColorStop(1,'rgba(27,80,83,0)');ctx.fillStyle=grad;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();}
        ctx.fillStyle='#9bb1b7';ctx.font='11px ui-monospace,monospace';ctx.fillText('Z',ox-16,oy+8);ctx.fillText('X',ox+iw-3,oy+ih+14);
      }
      const solution=current(), selected=state.selectedId!==null?lineageFor(state.selectedId,solution.edges).all:new Set(), active=new Set(solution.edges.map(e=>edgeKey(e.source,e.target)));
      function link(edge,missing){const a=nodes.get(edge.source),b=nodes.get(edge.target);if(!a||!b||b.t!==state.frame)return;const p=project(a),q=project(b),wrong=!truth.has(edgeKey(edge.source,edge.target)),related=selected.has(a.id)&&selected.has(b.id);ctx.save();ctx.strokeStyle=missing||wrong&&state.overlays.errors?'#f49886':related?'#c7ea7a':'#779999';ctx.globalAlpha=related||missing||wrong?.95:.6;ctx.lineWidth=related?2:1.2;ctx.setLineDash(missing||wrong&&state.overlays.errors?[3,4]:b.t-a.t>1?[5,3]:[]);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.setLineDash([]);ctx.beginPath();ctx.arc(p.x,p.y,2.3,0,Math.PI*2);ctx.stroke();ctx.restore();}
      if(state.overlays.links)for(const e of solution.edges)link(e,false);
      if(state.overlays.errors)for(const e of data.truth_edges)if(!active.has(edgeKey(e.source,e.target)))link(e,true);
      hitPoints=[];
      for(const n of data.frames[state.frame].detections){const p=project(n),r=Math.max(6,n.radius*scale*.95),related=selected.has(n.id),isSelected=n.id===state.selectedId;
        ctx.strokeStyle=n.is_false_positive&&state.overlays.errors?'#f49886':related?'#c7ea7a':'#85bbb8';ctx.lineWidth=isSelected?2.4:1.2;ctx.globalAlpha=selected.size&&!related?.55:1;
        ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();
        if(isSelected){ctx.beginPath();ctx.arc(p.x,p.y,r+4,0,Math.PI*2);ctx.lineWidth=.7;ctx.stroke();}
        if(state.overlays.ids){ctx.font='11px ui-monospace,monospace';ctx.fillStyle=related?'#e8ffc6':'#d1e4e1';ctx.fillText(String(n.id),p.x+r+4,p.y+3);}
        if(state.overlays.divisions&&solution.edges.filter(e=>e.source===n.id).length===2){ctx.strokeStyle='#c7ea7a';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(p.x+r+3,p.y-r-4);ctx.lineTo(p.x+r+7,p.y-r);ctx.lineTo(p.x+r+11,p.y-r-4);ctx.moveTo(p.x+r+7,p.y-r);ctx.lineTo(p.x+r+7,p.y-r+5);ctx.stroke();}
        ctx.globalAlpha=1;hitPoints.push({id:n.id,x:p.x,y:p.y,r:Math.max(r,12)});
      }
      canvas.setAttribute('aria-label',`Synthetic ${state.projection.toUpperCase()} view, frame ${state.frame+1} of ${data.frames.length}, ${hitPoints.length} detections. ${state.selectedId===null?'No selection.':`Detection ${state.selectedId} selected.`}`);
    }
    function updateGraph(lineage) {
      const svg=$('lineage-svg');svg.replaceChildren();
      const items=[...lineage.all].map(id=>nodes.get(id)).filter(Boolean).sort((a,b)=>a.t-b.t||a.id-b.id);
      const tracks=[...new Set(items.map(n=>n.track_id))];
      const point=n=>({x:18+n.t/(data.frames.length-1)*244,y:tracks.length===1?80:34+tracks.indexOf(n.track_id)/(tracks.length-1)*99});
      const guide=svgElement('line',{x1:18+state.frame/(data.frames.length-1)*244,y1:20,x2:18+state.frame/(data.frames.length-1)*244,y2:150,stroke:'#d3ddcb','stroke-dasharray':'3 3'});svg.appendChild(guide);
      for(const e of current().edges)if(lineage.all.has(e.source)&&lineage.all.has(e.target)){const p=point(nodes.get(e.source)),q=point(nodes.get(e.target));svg.appendChild(svgElement('line',{x1:p.x,y1:p.y,x2:q.x,y2:q.y,stroke:truth.has(edgeKey(e.source,e.target))?'#7bada2':'#cb8070','stroke-width':1.5,...(e.kind==='gap'?{'stroke-dasharray':'3 2'}:{})}));}
      for(const n of items){const p=point(n),circle=svgElement('circle',{cx:p.x,cy:p.y,r:n.id===state.selectedId?5:3.8,fill:n.id===state.selectedId?'#c7ea7a':'#117e78',stroke:n.id===state.selectedId?'#5e8953':'#f6f8f3','stroke-width':1,tabindex:0,role:'button','aria-label':`Detection ${n.id}, frame ${n.t+1}. Select and jump.`});circle.addEventListener('click',()=>selectCell(n.id,true));circle.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectCell(n.id,true);}});svg.appendChild(circle);}
      for(const frame of [0,3,7,11].filter(n=>n<data.frames.length)){const text=svgElement('text',{x:18+frame/(data.frames.length-1)*244,y:167,fill:'#58686f','font-size':11,'text-anchor':'middle'});text.textContent=String(frame+1).padStart(2,'0');svg.appendChild(text);}
      svg.setAttribute('aria-label',`${items.length} detection nodes in the selected predicted ancestry and descendants. Horizontal position is frame; rows use synthetic truth annotations for layout only.`);
    }
    function updateSelection() {
      const select=$('cell-select');select.replaceChildren(element('option','Select a cell in this frame'));select.children[0].value='';
      const frameNodes=[...data.frames[state.frame].detections];
      if(state.selectedId!==null&&!frameNodes.some(n=>n.id===state.selectedId))frameNodes.push(nodes.get(state.selectedId));
      for(const n of frameNodes){const option=element('option',`Detection ${n.id}${n.t!==state.frame?` · frame ${n.t+1}`:''}${n.is_false_positive?' · false detection':''}`);option.value=String(n.id);select.appendChild(option);}
      select.value=state.selectedId===null?'':String(state.selectedId);
      $('selection-empty').hidden=state.selectedId!==null;$('selection-detail').hidden=state.selectedId===null;
      if(state.selectedId===null)return;
      const node=nodes.get(state.selectedId),lineage=lineageFor(node.id,current().edges);
      $('selected-cell-id').textContent=`Detection ${node.id}`;$('selected-cell-frame').textContent=`FRAME ${String(node.t+1).padStart(2,'0')}`;
      const details=$('cell-details');details.replaceChildren();
      for(const [key,value] of [['Synthetic truth track',node.track_id],['Position · x / y / z',`${node.x.toFixed(1)} / ${node.y.toFixed(1)} / ${node.z.toFixed(1)}`],['Incoming / outgoing links',`${current().edges.filter(e=>e.target===node.id).length} / ${current().edges.filter(e=>e.source===node.id).length}`]]){details.appendChild(element('dt',key));details.appendChild(element('dd',String(value)));}
      $('selection-note').textContent=node.is_false_positive?'This authored false detection is an annotation for audit, not a biological cell.':`${lineage.ancestors.size} ancestors and ${lineage.descendants.size} descendants in the selected graph. Truth-track names are annotations, not solver predictions.`;
      $('lineage-count').textContent=`${lineage.all.size} nodes`;updateGraph(lineage);
    }
    function updateEvents() {
      const events=data.events.filter(e=>state.eventFilter==='all'||e.type===state.eventFilter), markers=$('event-markers');markers.replaceChildren();
      for(const event of events){const button=element('button');button.style.left=`${event.frame/(data.frames.length-1)*100}%`;button.dataset.type=event.type;button.setAttribute('aria-label',`Frame ${event.frame+1}: ${event.description}`);button.title=`Frame ${event.frame+1}: ${event.type.replaceAll('_',' ')}`;button.addEventListener('click',()=>jumpEvent(event));markers.appendChild(button);}
      const currentEvents=data.events.filter(e=>e.frame===state.frame), note=$('event-note');
      note.querySelector('.event-tag').textContent=currentEvents.length?'ANNOTATED EVENT':'SEQUENCE NOTE';
      note.querySelector('p').textContent=currentEvents.length?currentEvents.map(e=>e.description).join(' '):'No authored event in this frame. Select a detection to inspect its predicted lineage, or jump to a timeline marker.';
    }
    function renderMetrics() {
      const solution=current(),m=solution.metrics,cards=$('metric-cards');cards.replaceChildren();
      const rows=[['Edge F1',fmt(m.edge_f1),`Precision ${fmt(m.edge_precision)} · recall ${fmt(m.edge_recall)}`],['Wrong / missing links',`${m.false_positive} / ${m.false_negative}`,`${m.true_positive} matched reference edges`],['Divisions recovered',`${m.division_true_positive} / ${data.summary.truth_divisions}`,`${m.division_false_positive} extra predicted division${m.division_false_positive===1?'':'s'}`],['Gap links',String(m.gap_edges),`${m.selected_edges} selected links · ${m.components} components`]];
      for(const [label,value,note] of rows){const card=element('div',undefined,'metric-card');card.appendChild(element('span',label,'metric-label'));card.appendChild(element('strong',value));card.appendChild(element('span',note,'metric-note'));cards.appendChild(card);}
      $('policy-name').textContent=solution.label;$('policy-description').textContent=solution.description;
      $('topology-status').textContent=solution.checks.valid_topology?'Topology checks pass':`${solution.checks.issues.length} topology issue${solution.checks.issues.length===1?'':'s'}`;$('topology-status').className=`topology-status${solution.checks.valid_topology?'':' invalid'}`;
      $('policy-json').textContent=JSON.stringify({execution:solution.id==='live'?'Computed in this browser':'Saved Python solver output',parameters:solution.parameters,checks:solution.checks,solver:solution.solver||data.provenance.solver,evidence_type:'SYNTHETIC_ONLY'},null,2);
      const body=$('comparison-body');body.replaceChildren();
      for(const row of solutions){const tr=element('tr');tr.className=`${row.id===state.policyId?'active ':''}${row.id==='live'?'live-row':''}`;const label=element('td'),button=element('button',row.label);button.setAttribute('aria-pressed',String(row.id===state.policyId));button.addEventListener('click',()=>setPolicy(row.id));label.appendChild(button);tr.appendChild(label);tr.appendChild(element('td',String(row.metrics.selected_edges)));
        for(const key of ['edge_f1','division_f1']){const td=element('td'),meter=element('span',undefined,'table-meter');meter.appendChild(element('span',fmt(row.metrics[key])));const track=element('span',undefined,'meter-track'),bar=element('i');bar.style.width=`${row.metrics[key]*100}%`;track.appendChild(bar);meter.appendChild(track);td.appendChild(meter);tr.appendChild(td);}
        tr.appendChild(element('td',row.checks.valid_topology?'Pass':row.checks.issues.join('; ')));body.appendChild(tr);}
    }
    function setFrame(frame) {state.frame=clampFrame(frame,data.frames.length);$('frame-slider').value=String(state.frame);$('frame-slider').setAttribute('aria-valuetext',`Frame ${state.frame+1} of ${data.frames.length}`);$('frame-counter').textContent=`Frame ${String(state.frame+1).padStart(2,'0')} / ${data.frames.length}`;$('viewer-tooltip').hidden=true;updateSelection();updateEvents();draw();}
    function selectCell(id,jump=false) {const numeric=id===null||id===''?null:Number(id);if(numeric!==null&&!nodes.has(numeric))throw new Error('Unknown selected detection.');state.selectedId=numeric;if(jump&&numeric!==null)setFrame(nodes.get(numeric).t);else{updateSelection();draw();}}
    function setPolicy(id) {if(!solutions.some(s=>s.id===id))throw new Error('Unknown association policy.');state.policyId=id;$('policy-select').value=id;renderMetrics();updateSelection();draw();}
    function jumpEvent(event) {setPlaying(false);setFrame(event.frame);const selected=event.node_ids.find(id=>nodes.has(id)&&nodes.get(id).t===event.frame)||event.node_ids.find(id=>nodes.has(id));selectCell(selected===undefined?null:selected);}
    function setPlaying(playing) {state.playing=playing;if(timer!==null){win.clearInterval(timer);timer=null;}if(playing)timer=win.setInterval(()=>setFrame((state.frame+1)%data.frames.length),550/state.speed);$('play-icon').textContent=playing?'Ⅱ':'▶';$('play-button').setAttribute('aria-label',playing?'Pause movie':'Play movie');$('play-button').setAttribute('aria-pressed',String(playing));}
    function solveLive() {
      if(!win.BiohubGraph)throw new Error('The offline graph solver is unavailable.');
      const parameters={minimum_score:Number($('score-slider').value),max_children:Number($('children-select').value),max_gap:$('gap-toggle').checked?2:1};
      const result=win.BiohubGraph.solveTrackingGraph(data.candidates,parameters),measured=measureSolution(data,result.edges,parameters);
      const solution={id:'live',label:'Live browser solve',description:'Exact degree-constrained adjacent links, then optional residual gap closing. Computed locally from the same public candidates.',...result,...measured};
      const index=solutions.findIndex(s=>s.id==='live');if(index<0){solutions.push(solution);const option=element('option',solution.label);option.value='live';$('policy-select').appendChild(option);}else solutions[index]=solution;
      setPolicy('live');$('solve-status').textContent=`Solved locally: ${solution.edges.length} links, edge F1 ${fmt(solution.metrics.edge_f1)}, ${solution.checks.valid_topology?'topology checks pass':'topology issues found'}.`;return solution;
    }
    const policySelect=$('policy-select');policySelect.replaceChildren();for(const solution of solutions){const option=element('option',solution.label);option.value=solution.id;policySelect.appendChild(option);}policySelect.disabled=false;
    $('frame-slider').max=String(data.frames.length-1);$('frame-slider').disabled=false;$('frame-end').textContent=String(data.frames.length).padStart(2,'0');$('play-button').disabled=false;$('export-button').disabled=false;
    on(policySelect,'change',()=>setPolicy(policySelect.value));on($('frame-slider'),'input',()=>{setPlaying(false);setFrame($('frame-slider').value);});
    on($('play-button'),'click',()=>setPlaying(!state.playing));on($('speed-select'),'change',()=>{state.speed=Number($('speed-select').value);setPlaying(state.playing);});
    on($('scenario-select'),'change',()=>{state.eventFilter=$('scenario-select').value;const event=data.events.find(e=>e.type===state.eventFilter);if(event)jumpEvent(event);else setFrame(0);updateEvents();});
    on($('cell-select'),'change',()=>selectCell($('cell-select').value));on($('clear-selection'),'click',()=>selectCell(null));
    for(const key of Object.keys(state.overlays))on($(`toggle-${key}`),'change',()=>{state.overlays[key]=$(`toggle-${key}`).checked;draw();});
    for(const projection of ['xy','xz'])on($(`view-${projection}`),'click',()=>{state.projection=projection;$('view-xy').setAttribute('aria-pressed',String(projection==='xy'));$('view-xz').setAttribute('aria-pressed',String(projection==='xz'));$('projection-label').textContent=`${projection.toUpperCase()} projection`;$('canvas-caption').textContent=projection==='xy'?'Python-rendered synthetic projection':'Depth view from authored cell coordinates · not an acquired image';draw();});
    on($('score-slider'),'input',()=>{$('score-value').textContent=Number($('score-slider').value).toFixed(2);});
    on($('solve-button'),'click',()=>{try{solveLive();}catch(error){$('solve-status').textContent=`Solve failed: ${error.message}`;}});
    function hit(event){const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;return hitPoints.filter(p=>(p.x-x)**2+(p.y-y)**2<=p.r**2).sort((a,b)=>(a.x-x)**2+(a.y-y)**2-((b.x-x)**2+(b.y-y)**2))[0];}
    on(canvas,'click',event=>{const found=hit(event);selectCell(found?found.id:null);});
    on(canvas,'pointermove',event=>{const found=hit(event),tip=$('viewer-tooltip');canvas.style.cursor=found?'pointer':'default';tip.hidden=!found;if(found){tip.textContent=`Detection ${found.id} · select to trace`;const rect=canvas.getBoundingClientRect();tip.style.left=`${Math.max(8,Math.min(rect.width-175,found.x+12))}px`;tip.style.top=`${Math.max(8,found.y-35)}px`;}});
    on(canvas,'pointerleave',()=>{$('viewer-tooltip').hidden=true;});
    on(canvas,'keydown',event=>{if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();setPlaying(false);setFrame(state.frame+(event.key==='ArrowRight'?1:-1));}else if(event.key===' '){event.preventDefault();setPlaying(!state.playing);}else if(event.key==='Escape'){selectCell(null);}});
    on($('export-button'),'click',()=>{const blob=new win.Blob([JSON.stringify(exportResult(data,state,current()),null,2)],{type:'application/json'}),url=win.URL.createObjectURL(blob),a=element('a');a.href=url;a.download=`biohub-${state.policyId}-review.json`;doc.body.appendChild(a);a.click();a.remove();win.setTimeout(()=>win.URL.revokeObjectURL(url),1000);});
    on(doc,'visibilitychange',()=>{if(doc.hidden)setPlaying(false);});on(win,'resize',draw);
    let observer=null;if(win.ResizeObserver){observer=new win.ResizeObserver(draw);observer.observe(canvas);}
    setPolicy('balanced');setFrame(0);selectCell(data.frames[0].detections[0].id);
    return {getState:()=>({...state,overlays:{...state.overlays}}),getSolution:()=>current(),setFrame,selectCell,setPolicy,solveLive,draw,destroy(){disposed=true;setPlaying(false);cleanup.forEach(fn=>fn());if(observer)observer.disconnect();}};
  }
  const API={edgeKey,clampFrame,lineageFor,measureSolution,exportResult,init};
  if(typeof module!=='undefined'&&module.exports)module.exports=API;
  root.BiohubExplorer=API;
  if(root.document){try{root.biohubExplorer=init(root.BIOHUB_DEMO_DATA,root.document,root);}catch(error){const message=root.document.getElementById('load-message');if(message){message.hidden=false;message.textContent=`Explorer could not start: ${error.message}`;}if(root.console)root.console.error(error);}}
})(typeof window!=='undefined'?window:globalThis);
