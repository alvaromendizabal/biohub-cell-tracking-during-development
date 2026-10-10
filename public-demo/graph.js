/* Public synthetic demonstration: exact degree-constrained edge selection.
 * This generic min-cost-flow formulation contains no private tracking recipe.
 * It maximizes total retained edge score, with at most maxChildren outgoing
 * edges per source and one incoming edge per target. Biology needs additional
 * evidence; a structurally valid graph can still be incorrect.
 */
(function (root) {
  'use strict';
  function solveGraph(candidates, options = {}) {
    const { scoreKey = 'edge_score', minScore = 0, maxChildren = 2 } = options;
    if (!Array.isArray(candidates) || candidates.length > 20000) throw new Error('Invalid candidate collection.');
    if (![1, 2].includes(maxChildren) || !Number.isFinite(minScore)) throw new Error('Invalid graph controls.');
    const seen = new Set();
    const eligible = [];
    for (const row of candidates) {
      if (!row || !Number.isSafeInteger(row.source_id) || !Number.isSafeInteger(row.target_id) || row.source_id === row.target_id) throw new Error('Invalid candidate endpoints.');
      if (typeof row[scoreKey] !== 'number' || !Number.isFinite(row[scoreKey]) || Math.abs(row[scoreKey]) > 1e6) throw new Error('Invalid candidate score.');
      const key = `${row.source_id}:${row.target_id}`;
      if (seen.has(key)) throw new Error('Duplicate candidate edge.');
      seen.add(key);
      if (row[scoreKey] >= minScore && row[scoreKey] > 0) eligible.push(row);
    }
    eligible.sort((a, b) => a.source_id - b.source_id || a.target_id - b.target_id);
    const sources = [...new Set(eligible.map(e => e.source_id))].sort((a, b) => a - b);
    const targets = [...new Set(eligible.map(e => e.target_id))].sort((a, b) => a - b);
    const sourceIndex = new Map(sources.map((id, i) => [id, i + 1]));
    const targetIndex = new Map(targets.map((id, i) => [id, i + 1 + sources.length]));
    const sink = sources.length + targets.length + 1;
    const residual = Array.from({ length: sink + 1 }, () => []);
    function addEdge(from, to, capacity, cost) {
      const forward = { to, capacity, cost, reverse: residual[to].length };
      const backward = { to: from, capacity: 0, cost: -cost, reverse: residual[from].length };
      residual[from].push(forward);
      residual[to].push(backward);
      return forward;
    }
    for (const source of sources) addEdge(0, sourceIndex.get(source), maxChildren, 0);
    for (const target of targets) addEdge(targetIndex.get(target), sink, 1, 0);
    const selectionEdges = eligible.map(row => addEdge(sourceIndex.get(row.source_id), targetIndex.get(row.target_id), 1, -row[scoreKey]));
    // Each shortest augmenting path may undo an earlier association. This is
    // global optimization, unlike independent or irrevocable greedy choices.
    for (;;) {
      const distance = new Float64Array(sink + 1).fill(Infinity);
      const previousNode = new Int32Array(sink + 1).fill(-1);
      const previousEdge = new Int32Array(sink + 1).fill(-1);
      distance[0] = 0;
      for (let pass = 0; pass < sink; pass++) {
        let changed = false;
        for (let from = 0; from <= sink; from++) {
          if (!Number.isFinite(distance[from])) continue;
          for (let i = 0; i < residual[from].length; i++) {
            const edge = residual[from][i];
            const proposal = distance[from] + edge.cost;
            if (edge.capacity > 0 && proposal < distance[edge.to] - 1e-12) {
              distance[edge.to] = proposal;
              previousNode[edge.to] = from;
              previousEdge[edge.to] = i;
              changed = true;
            }
          }
        }
        if (!changed) break;
      }
      if (!Number.isFinite(distance[sink]) || distance[sink] >= -1e-12) break;
      for (let node = sink; node !== 0; node = previousNode[node]) {
        const from = previousNode[node];
        if (from < 0) throw new Error('Invalid residual path.');
        const edge = residual[from][previousEdge[node]];
        edge.capacity--;
        residual[node][edge.reverse].capacity++;
      }
    }
    const selected = eligible.filter((_, index) => selectionEdges[index].capacity === 0);
    return {
      edges: selected.map(row => [row.source_id, row.target_id]),
      objective: selected.reduce((sum, row) => sum + row[scoreKey], 0),
      selectedCount: selected.length,
      eligibleCount: eligible.length,
      candidateCount: candidates.length,
      controls: { scoreKey, minScore, maxChildren },
      algorithm: 'degree-constrained maximum-weight bipartite matching',
      evidence_type: 'SYNTHETIC_ONLY'
    };
  }
  function solveTrackingGraph(candidates, options = {}) {
    const { minimum_score = 0, max_children = 2, max_gap = 2 } = options;
    if (!Array.isArray(candidates) || ![1, 2].includes(max_gap) || ![1, 2].includes(max_children) || !Number.isFinite(minimum_score)) throw new Error('Invalid tracking controls.');
    const adapted = candidates.map(row => {
      if (![1, 2].includes(row.dt)) throw new Error('Invalid temporal gap.');
      return { source_id: row.source, target_id: row.target, edge_score: row.score - minimum_score, original: row };
    });
    // Validate all candidates, including those a chosen policy may exclude.
    solveGraph(adapted, { minScore: 1e6 });
    const adjacent = adapted.filter(row => row.original.dt === 1);
    const first = solveGraph(adjacent, { maxChildren: max_children });
    const outgoing = new Set(first.edges.map(edge => edge[0]));
    const incoming = new Set(first.edges.map(edge => edge[1]));
    const remaining = max_gap === 2 ? adapted.filter(row => row.original.dt === 2 && !outgoing.has(row.source_id) && !incoming.has(row.target_id)) : [];
    const gaps = solveGraph(remaining, { maxChildren: 1 });
    const pairs = [...first.edges, ...gaps.edges].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const lookup = new Map(candidates.map(row => [`${row.source}:${row.target}`, row]));
    const count = new Map();
    for (const [source] of pairs) count.set(source, (count.get(source) || 0) + 1);
    return {
      edges: pairs.map(([source, target]) => {
        const row = lookup.get(`${source}:${target}`);
        return { source, target, score: row.score, kind: count.get(source) === 2 ? 'division' : row.dt === 2 ? 'gap' : 'continuation' };
      }),
      objective: first.objective + gaps.objective,
      parameters: { minimum_score, max_children, max_gap },
      solver: 'Browser min-cost flow: adjacent associations, then residual gap closing',
      evidence_type: 'SYNTHETIC_ONLY'
    };
  }
  const api = Object.freeze({ solveGraph, solveTrackingGraph });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BiohubGraph = api;
})(globalThis);
