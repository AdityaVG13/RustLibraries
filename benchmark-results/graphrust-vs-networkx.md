# GraphRust vs NetworkX

Same-data local benchmark for the implemented GraphRust BFS slice.
This is not a full NetworkX replacement claim.

- NetworkX version: `3.6.1`
- GraphRust wins: 1
- NetworkX wins: 0
- Checksum failures: 0
- Global NetworkX replacement claim: false

| Case | GraphRust ms | NetworkX ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `bfs_undirected_5000_30000` | 0.046 | 1.247 | 27.15x | graphrust | ok |
