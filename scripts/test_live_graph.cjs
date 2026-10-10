'use strict';
const assert = require('node:assert/strict');
const { solveGraph, solveTrackingGraph } = require('../public-demo/graph.js');

const row = (source_id, target_id, edge_score) => ({ source_id, target_id, edge_score });
const trap = [row(1, 3, 10), row(1, 4, 9), row(2, 3, 9), row(2, 4, -1)];
assert.deepEqual(solveGraph(trap, { maxChildren: 1 }).edges, [[1, 4], [2, 3]]);
assert.equal(solveGraph(trap, { maxChildren: 1 }).objective, 18);
assert.deepEqual(solveGraph([...trap].reverse(), { maxChildren: 1 }).edges, [[1, 4], [2, 3]]);
assert.deepEqual(solveGraph(trap, { minScore: 10 }).edges, [[1, 3]]);
assert.deepEqual(solveGraph([]).edges, []);
assert.deepEqual(solveGraph([row(1, 3, 0), row(2, 3, -2)]).edges, []);
assert.equal(solveGraph([row(1, 3, 9), row(1, 4, 8), row(1, 5, 7), row(2, 5, 6)]).objective, 23);
assert.throws(() => solveGraph([row(1, 3, NaN)]), /score/);
assert.throws(() => solveGraph([row(1, 3, 1), row(1, 3, 2)]), /Duplicate/);
assert.throws(() => solveGraph([row(1, 1, 2)]), /endpoints/);
assert.throws(() => solveGraph(trap, { maxChildren: 3 }), /controls/);

// Compare a deterministic collection of small graphs with exhaustive search.
let seed = 21;
const next = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
for (let trial = 0; trial < 40; trial++) {
  const candidates = [];
  for (let source = 1; source <= 3; source++) {
    for (let target = 4; target <= 6; target++) candidates.push(row(source, target, Math.floor(next() * 21) - 5));
  }
  for (const capacity of [1, 2]) {
    let best = 0;
    for (let mask = 0; mask < (1 << candidates.length); mask++) {
      const incoming = new Map(), outgoing = new Map();
      let score = 0, valid = true;
      for (let index = 0; index < candidates.length; index++) if (mask & (1 << index)) {
        const edge = candidates[index];
        incoming.set(edge.target_id, (incoming.get(edge.target_id) || 0) + 1);
        outgoing.set(edge.source_id, (outgoing.get(edge.source_id) || 0) + 1);
        if (incoming.get(edge.target_id) > 1 || outgoing.get(edge.source_id) > capacity) valid = false;
        score += edge.edge_score;
      }
      if (valid) best = Math.max(best, score);
    }
    assert.equal(solveGraph(candidates, { maxChildren: capacity }).objective, best);
  }
}
console.log('LIVE_GRAPH_PASSED: 80 exhaustive-optimum comparisons plus constraints, threshold, determinism and invalid inputs');

// Cross-language evidence: the committed synthetic fixture is generated with
// Python's SciPy MILP. The independent JS optimizer must select the same graph.
const fixture = require('../public-demo/data.json');
let compared = 0;
for (const solution of fixture.solutions.filter(row => row.id !== 'independent')) {
  const computed = solveTrackingGraph(fixture.candidates, solution.parameters);
  assert.deepEqual(computed.edges, solution.edges, `${solution.id}: browser/Python edge mismatch`);
  compared++;
}
assert.equal(compared, 5);
const gapCase = [{ source: 1, target: 2, dt: 1, score: 2 }, { source: 1, target: 3, dt: 2, score: 3 }];
assert.deepEqual(solveTrackingGraph(gapCase).edges.map(e => [e.source, e.target]), [[1, 2]]);
assert.throws(() => solveTrackingGraph(gapCase, { max_gap: 3 }), /controls/);
console.log('TRACKING_POLICY_PARITY_PASSED: five Python MILP policies and residual-gap constraint');
