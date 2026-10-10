'use strict';
// Node-only behavioral test: the production app initializes against a small DOM
// implementation. Canvas commands are recorded; no browser or screenshots are
// simulated as visual evidence.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const App = require('../public-demo/app.js');
const Graph = require('../public-demo/graph.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../public-demo/data.json'), 'utf8'));
let checks = 0;
function check(name, fn) { fn(); checks++; console.log(`PASS ${name}`); }
check('all saved policy metrics and topology match Python exactly', () => {
  for (const policy of data.solutions) {
    const result=App.measureSolution(data,policy.edges,policy.parameters);
    assert.deepEqual(result.metrics,policy.metrics,policy.id);
    assert.deepEqual(result.checks,policy.checks,policy.id);
  }
});
check('directed ancestry follows divisions and terminates on cycles', () => {
  const edges=[{source:1,target:2},{source:2,target:3},{source:2,target:4},{source:4,target:5}];
  const branch=App.lineageFor(4,edges);
  assert.deepEqual([...branch.ancestors].sort(),[1,2]);
  assert.deepEqual([...branch.descendants],[5]);
  assert(!branch.all.has(3));
  assert.equal(App.lineageFor(1,[{source:1,target:2},{source:2,target:1}]).all.size,2);
});
check('frame clamps and unknown graph endpoints are rejected', () => {
  assert.equal(App.clampFrame(-4,12),0);assert.equal(App.clampFrame(99,12),11);
  assert.throws(()=>App.measureSolution(data,[{source:0,target:999999}],data.solutions[0].parameters),/unknown/);
});
class Element {
  constructor(tag='div') {this.tagName=tag.toUpperCase();this.children=[];this.parentNode=null;this.style={};this.dataset={};this.attributes={};this.listeners=new Map();this.value='';this.checked=false;this.disabled=false;this.hidden=false;this._text='';this.className='';}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(e=>e.textContent).join('');}
  appendChild(child){child.parentNode=this;this.children.push(child);return child;}
  replaceChildren(...children){this._text='';this.children=[];children.forEach(e=>this.appendChild(e));}
  setAttribute(key,value){this.attributes[key]=String(value);}
  getAttribute(key){return this.attributes[key];}
  addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}
  removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}
  fire(type,props={}){const event={target:this,preventDefault(){this.defaultPrevented=true;},...props};for(const fn of this.listeners.get(type)||[])fn(event);return event;}
  click(){this.fire('click');}
  remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(e=>e!==this);}
  querySelector(query){const predicate=query.startsWith('.')?e=>e.className.split(' ').includes(query.slice(1)):e=>e.tagName===query.toUpperCase();const pending=[...this.children];while(pending.length){const child=pending.shift();if(predicate(child))return child;pending.push(...child.children);}return null;}
  getBoundingClientRect(){return {width:720,height:448,left:0,top:0};}
}
function createDOM() {
  const html=fs.readFileSync(path.join(__dirname,'../public-demo/index.html'),'utf8');
  const elements=new Map();
  for(const match of html.matchAll(/<([a-zA-Z0-9-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const el=new Element(match[1]);el.checked=/\bchecked\b/.test(match[2]);el.disabled=/\bdisabled\b/.test(match[2]);el.hidden=/\bhidden\b/.test(match[2]);
    const value=match[2].match(/\bvalue="([^"]*)"/);if(value)el.value=value[1];elements.set(match[3],el);
  }
  const commands=[];
  const context=new Proxy({createRadialGradient:()=>({addColorStop(){}})}, {get(target,key){if(key in target)return target[key];return (...args)=>commands.push([key,...args]);},set(target,key,value){target[key]=value;return true;}});
  elements.get('movie-canvas').getContext=()=>context;
  const eventNote=elements.get('event-note');const tag=new Element('span');tag.className='event-tag';eventNote.appendChild(tag);eventNote.appendChild(new Element('p'));
  elements.get('children-select').value='2';elements.get('speed-select').value='1';elements.get('scenario-select').value='all';
  const doc=new Element('document');doc.body=new Element('body');doc.hidden=false;doc.getElementById=id=>elements.get(id)||null;doc.createElement=tag=>new Element(tag);doc.createElementNS=(_,tag)=>new Element(tag);
  const win=new Element('window'),timers=new Map();let nextTimer=1;const downloads=[],revoked=[];
  win.Image=class {constructor(){this.complete=true;this.naturalWidth=160;}set src(value){this.source=value;}};
  win.devicePixelRatio=1;win.BiohubGraph=Graph;win.Blob=Blob;
  win.URL={createObjectURL(blob){downloads.push(blob);return `blob:test-${downloads.length}`;},revokeObjectURL(url){revoked.push(url);}};
  win.setInterval=fn=>{const id=nextTimer++;timers.set(id,fn);return id;};win.clearInterval=id=>timers.delete(id);win.setTimeout=fn=>{fn();};
  win.ResizeObserver=class {observe(){}disconnect(){this.disconnected=true;}};
  return {doc,win,elements,commands,timers,downloads,revoked,html};
}
const env=createDOM();const ui=App.init(data,env.doc,env.win);const el=id=>env.elements.get(id);
check('real app initializes all policies, metrics, image, and accessible selection',()=>{
  assert.equal(ui.getState().policyId,'balanced');assert.equal(ui.getState().selectedId,0);
  assert.equal(el('policy-select').children.length,6);assert.equal(el('metric-cards').children.length,4);
  assert.equal(el('comparison-body').children.length,6);assert.equal(el('frame-slider').max,'11');
  assert.equal(el('load-message').hidden,true);assert(env.commands.some(row=>row[0]==='drawImage'));
  assert.match(el('cell-details').textContent,/Synthetic truth track/);
  assert.match(el('movie-canvas').getAttribute('aria-label'),/frame 1 of 12/);
});
check('scrubber, keyboard step, play, speed, and hidden-tab pause are connected',()=>{
  el('frame-slider').value='4';el('frame-slider').fire('input');assert.equal(ui.getState().frame,4);
  const evt=el('movie-canvas').fire('keydown',{key:'ArrowRight'});assert.equal(evt.defaultPrevented,true);assert.equal(ui.getState().frame,5);
  el('movie-canvas').fire('keydown',{key:' '});assert.equal(ui.getState().playing,true);assert.equal(env.timers.size,1);
  [...env.timers.values()][0]();assert.equal(ui.getState().frame,6);
  el('speed-select').value='2';el('speed-select').fire('change');assert.equal(ui.getState().speed,2);assert.equal(env.timers.size,1);
  env.doc.hidden=true;env.doc.fire('visibilitychange');assert.equal(ui.getState().playing,false);assert.equal(env.timers.size,0);
});
check('event navigation uses actual annotated frames and selected detected cells',()=>{
  el('scenario-select').value='division';el('scenario-select').fire('change');assert.equal(ui.getState().frame,4);
  assert.equal(el('event-markers').children.length,2);assert(data.events.find(e=>e.type==='division').node_ids.includes(ui.getState().selectedId));
  el('event-markers').children[1].click();assert.equal(ui.getState().frame,7);
  assert.match(el('event-note').textContent,/authored parent/);
  el('scenario-select').value='missed_detection';el('scenario-select').fire('change');assert.equal(ui.getState().frame,6);assert.equal(ui.getState().selectedId,null);
});
check('cell list and lineage nodes select, jump, and clear',()=>{
  el('cell-select').value=String(data.frames[6].detections[0].id);el('cell-select').fire('change');assert.notEqual(ui.getState().selectedId,null);
  const node=el('lineage-svg').children.find(e=>e.tagName==='CIRCLE');assert(node);node.fire('keydown',{key:'Enter'});
  assert.equal(ui.getState().frame,data.nodes.find(n=>n.id===ui.getState().selectedId).t);
  el('clear-selection').click();assert.equal(ui.getState().selectedId,null);assert.equal(el('selection-empty').hidden,false);
});
check('canvas click and projection/overlay toggles update actual rendering state',()=>{
  ui.setFrame(0);const n=data.frames[0].detections[0],scale=Math.min((720-44)/160,(448-64)/128);
  el('movie-canvas').fire('click',{clientX:(720-160*scale)/2+n.x*scale,clientY:(448-128*scale)/2-5+n.y*scale});assert.equal(ui.getState().selectedId,n.id);
  el('view-xz').click();assert.equal(ui.getState().projection,'xz');assert.match(el('canvas-caption').textContent,/not an acquired image/);
  assert(env.commands.some(row=>row[0]==='arc'));
  el('toggle-errors').checked=false;el('toggle-errors').fire('change');assert.equal(ui.getState().overlays.errors,false);
  el('view-xy').click();assert.equal(el('view-xy').getAttribute('aria-pressed'),'true');
});
check('policy menu and comparison buttons switch exact fixture solutions',()=>{
  el('policy-select').value='no_divisions';el('policy-select').fire('change');assert.equal(ui.getSolution().metrics.division_true_positive,0);
  el('comparison-body').children[0].children[0].children[0].click();assert.equal(ui.getState().policyId,'balanced');
});
check('live solve controls compute new edges and real independently matching metrics',()=>{
  el('score-slider').value='1.2';el('score-slider').fire('input');assert.equal(el('score-value').textContent,'1.20');
  el('solve-button').click();assert.equal(ui.getState().policyId,'live');assert.equal(el('policy-select').children.length,7);
  const conservative=data.solutions.find(s=>s.id==='conservative');assert.deepEqual(ui.getSolution().metrics,conservative.metrics);
  el('children-select').value='1';el('gap-toggle').checked=false;el('score-slider').value='2.4';el('solve-button').click();
  assert.equal(ui.getSolution().metrics.selected_edges,0);assert.equal(ui.getSolution().checks.valid_topology,true);
  assert.equal(el('policy-select').children.length,7);assert.match(el('solve-status').textContent,/Solved locally: 0 links/);
});
check('exports distinguish browser computation from saved outputs and preserve annotations',()=>{
  const result=App.exportResult(data,ui.getState(),ui.getSolution());assert.equal(result.evidence_type,'SYNTHETIC_ONLY');assert.match(result.computation,/Browser graph optimization/);
  assert.equal(result.provenance.private_assets_used,false);assert.equal(result.provenance.official_score,null);
  assert.equal(result.solution.metrics.selected_edges,0);assert.equal(result.nodes.length,data.nodes.length);
  el('export-button').click();assert.equal(env.downloads.length,1);assert.equal(env.revoked.length,1);
});
check('offline source ordering, no remote runtime scripts, no autoplay',()=>{
  assert(env.html.indexOf('src="data.js"')<env.html.indexOf('src="graph.js"'));
  assert(env.html.indexOf('src="graph.js"')<env.html.indexOf('src="app.js"'));
  assert(!/<script[^>]+src="https?:/i.test(env.html));assert.equal(ui.getState().playing,false);
  assert(!fs.readFileSync(path.join(__dirname,'../public-demo/app.js'),'utf8').includes('.innerHTML'));
});
check('dispose cancels playback and detaches persistent event handlers',()=>{
  el('play-button').click();assert.equal(env.timers.size,1);ui.destroy();assert.equal(env.timers.size,0);
  const before=ui.getState().frame;el('movie-canvas').fire('keydown',{key:'ArrowRight'});assert.equal(ui.getState().frame,before);
});
env.downloads[0].text().then(raw=>{const result=JSON.parse(raw);assert.equal(result.solution.id,'live');assert.equal(result.solution.edges.length,0);console.log(`PASS exported JSON bytes parse and match active computation\n${checks+1} frontend checks passed.`);}).catch(error=>{console.error(error);process.exitCode=1;});
