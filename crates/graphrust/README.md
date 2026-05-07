# GraphRust

GraphRust is the graph analytics crate in this workspace.

Current scope:

- Directed and undirected CSR graph construction from edge lists.
- Neighbor access with bounds-checked APIs.
- Breadth-first-search distances.
- Connected components for undirected-style traversal.
- Iterative PageRank.
- A same-data NetworkX comparison harness.

It is not a NetworkX replacement yet. Weighted graphs, multigraphs, graph IO, shortest-path variants, centrality families, community detection, and matrix-backed algorithms are roadmap items.
