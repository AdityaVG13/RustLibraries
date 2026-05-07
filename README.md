# RustLibraries

Flagship-first porting lab for rebuilding top Python library ideas in pure Rust.

Working names: **NumRust** for the NumPy-core slice, **StatsRust** for the StatsModels-style slice, **SciRust** for the SciPy-style slice, and **LearnRust** for the scikit-learn-style slice. Core crate: `numrs-core`.

Current flagship: `numrs-core`, a Rust-first NumPy-core foundation. It targets typed n-dimensional arrays, shape/stride metadata, uniform-array metadata, zero-copy views, NumPy-style slicing, reshape, transpose, flip, moveaxis, concatenate, stack, roll, multi-array broadcasting, elementwise kernels, reductions, 2-D dot products, NumPy-style matrix multiplication, and tensor contractions.

## Status

This is not a full NumPy replacement yet, and no crate in this workspace should be treated as a full Python-library replacement until its documented parity gates are met. Current v0 slices are deliberately narrow and tested. The architecture is aimed above Python libraries on Rust-native strengths: explicit ownership, checked layout contracts, zero-copy view transforms, uniform-array plans, broadcast plans that avoid input materialization, and fast paths for contiguous kernels.

| Crate | Python target | Current parity status | Allowed claim |
| --- | --- | --- | --- |
| `numrs-core` / NumRust | NumPy core | Partial | Wins only on the benchmarked NumPy-core slice. |
| `statsrust` | StatsModels | Partial | Wins only on the benchmarked statistics slice. |
| `scirust` | SciPy | Partial | Wins only on the benchmarked scientific-routines slice. |
| `framerust` | Pandas | Partial | Wins only on the benchmarked groupby slice. |
| `graphrust` | NetworkX | Partial | Wins only on the benchmarked graph slice. |
| `mediaextractrust` | Python extraction libraries | Partial | Wins only on the benchmarked extraction slice. |
| `validaterust` | Pydantic | Partial | Wins only on the benchmarked validation slice. |
| `imagerust` | Pillow / scikit-image | Partial | Wins only on the benchmarked image-processing slice. |
| `textrust` | NLTK / spaCy | Partial | Wins only on the benchmarked text-processing slice. |
| `learnrust` | scikit-learn | Partial | Wins only on the benchmarked classical-ML slice. |

## Benchmark Snapshot

Full tables, rerun commands, and claim gates live in [`docs/benchmark-dashboard.md`](docs/benchmark-dashboard.md). Current rule: implemented slices may claim measured wins; full-library replacement claims stay false until coverage and external evidence justify them.

| Library slice | Rust crate | Python baseline | Cases | Rust wins | Python wins | Full parity? | Speedup summary | Report |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| NumPy targeted same-data | `numrs-core` | NumPy 2.4.4 | 10 | 10 | 0 | No | 1.67x geomean | [`numrust-vs-numpy.md`](benchmark-results/numrust-vs-numpy.md) |
| NumPy core, external ASV-derived | `numrs-core` | NumPy 2.4.4 | 84 | 82 | 2 | No | 8.41x geomean | [`external-numpy-asv-inspired.md`](benchmark-results/external-numpy-asv-inspired.md) |
| NumPy current-loss focused rerun | `numrs-core` | NumPy 2.4.4 | 2 | 2 | 0 | No | focused losses flip | [`external-numpy-loss-focused.md`](benchmark-results/external-numpy-loss-focused.md) |
| Statistics | `statsrust` | StatsModels 0.14.6 | 4 | 4 | 0 | No | 3.51x geomean | [`statsrust-vs-statsmodels.md`](benchmark-results/statsrust-vs-statsmodels.md) |
| Scientific routines | `scirust` | SciPy 1.17.1 | 9 | 9 | 0 | No | 19.11x geomean | [`scirust-vs-scipy.md`](benchmark-results/scirust-vs-scipy.md) |
| Data aggregation | `framerust` | Pandas 3.0.2 | 1 | 1 | 0 | No | 2.14x | [`framerust-vs-pandas.md`](benchmark-results/framerust-vs-pandas.md) |
| Graph analytics | `graphrust` | NetworkX 3.6.1 | 1 | 1 | 0 | No | 27.15x | [`graphrust-vs-networkx.md`](benchmark-results/graphrust-vs-networkx.md) |
| Document extraction | `mediaextractrust` | Python extraction libraries | 2 | 2 | 0 | No | 81.21x to 122.93x | [`mediaextractrust-vs-python.md`](benchmark-results/mediaextractrust-vs-python.md) |
| Data validation | `validaterust` | Pydantic 2.13.4 | 1 | 1 | 0 | No | 35.69x | [`validaterust-vs-pydantic.md`](benchmark-results/validaterust-vs-pydantic.md) |
| Image processing | `imagerust` | Pillow 12.2.0 | 1 | 1 | 0 | No | 16.53x | [`imagerust-vs-pillow.md`](benchmark-results/imagerust-vs-pillow.md) |
| Text processing | `textrust` | NLTK 3.9.4 | 1 | 1 | 0 | No | 23.92x | [`textrust-vs-nltk.md`](benchmark-results/textrust-vs-nltk.md) |
| Classical ML | `learnrust` | scikit-learn 1.8.0 | 3 | 3 | 0 | No | 6.53x geomean | [`learnrust-vs-sklearn.md`](benchmark-results/learnrust-vs-sklearn.md) |

## Progress TODO

Backfilled items are completed work imported from the pre-GitHub build history. New work should update this list in a dedicated TODO commit after the implementation or benchmark evidence commit.

### Backfilled Done

- [x] 2026-05-07 backfilled: Create a Rust workspace for NumRust, StatsRust, SciRust, RigorTrail, and the Python bridge. Evidence: `Cargo.toml`, `crates/*`.
- [x] 2026-05-07 backfilled: Implement NumRust typed arrays, shape/stride metadata, views, slicing, reshape, transpose, broadcasting, reductions, selected linalg, matmul, tensordot, and indexed writes. Evidence: `crates/numrs-core`, `crates/numrs-core/tests/numrs_core.rs`.
- [x] 2026-05-07 backfilled: Add primitive dtype metadata, casts, promotion records, bool/int/float/complex Python namespace tokens, and Array API smoke coverage. Evidence: `crates/numrs-core/src/dtype.rs`, `python/numrust`.
- [x] 2026-05-07 backfilled: Add Accelerate/vDSP and BLAS-backed fast paths for supported contiguous reductions, dot, matmul, inner, selected contractions, and NEON scalar multiply. Evidence: `docs/performance.md`, `crates/numrs-core/src/blas.rs`.
- [x] 2026-05-07 backfilled: Implement fancy indexing primitives: `take`, `take_axis`, boolean masks, `put`, `putmask`, `add_at`, and `maximum_at`. Evidence: `crates/numrs-core/tests/numrs_core.rs`.
- [x] 2026-05-07 backfilled: Add a PyO3-backed `numrust` namespace and verify import, dtype, arithmetic, indexing, reshape, matmul, sum, and mean smoke cases. Evidence: `crates/numrust-python`, `benchmarks/verify_array_api_namespace.py`.
- [x] 2026-05-07 backfilled: Pin NumPy ASV, Array API, SciPy, and StatsModels source references with hash verification. Evidence: `benchmark-results/external-source-lock.json`, `benchmarks/external_sources.py`.
- [x] 2026-05-07 backfilled: Add external NumPy ASV-derived benchmarks with real NumPy comparison, sharded pass aggregation, loss triage, focused loss reruns, and stability metadata. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07 backfilled: Run pinned upstream Array API probes without patching upstream tests. Evidence: `benchmark-results/array-api-tests-focused-probe.md`, `benchmark-results/array-api-tests-full-maxfail.md`.
- [x] 2026-05-07 backfilled: Start StatsRust and SciRust slices with tests and comparison harnesses. Evidence: `crates/statsrust`, `crates/scirust`, `benchmark-results/*statsmodels*`, `benchmark-results/*scipy*`.
- [x] 2026-05-07 backfilled: Invent and implement RigorTrail for evidence-ledger validation and claim gating. Evidence: `crates/rigortrail`, `docs/rigortrail.md`.
- [x] 2026-05-07 backfilled: Pass Rust format, clippy, workspace tests, Python benchmark schema tests, and source-lock verification before GitHub import. Evidence: `docs/completion-audit.md`.
- [x] 2026-05-07: Start FrameRust with typed columns, validated frames, first-seen-order groupby, and `count`/`sum`/`mean`/`min`/`max` aggregations. Evidence: `crates/framerust`.
- [x] 2026-05-07: Add same-data FrameRust vs Pandas benchmark evidence for the implemented groupby aggregation slice. Evidence: `benchmarks/compare_pandas.py`, `benchmark-results/framerust-vs-pandas.md`.
- [x] 2026-05-07: Start GraphRust with CSR graph construction, BFS distances, connected components, and PageRank. Evidence: `crates/graphrust`.
- [x] 2026-05-07: Add same-data GraphRust vs NetworkX benchmark evidence for the implemented BFS slice. Evidence: `benchmarks/compare_networkx.py`, `benchmark-results/graphrust-vs-networkx.md`.
- [x] 2026-05-07: Start MediaExtractRust with text/Markdown normalization, HTML text extraction, and uncompressed PDF literal text extraction. Evidence: `crates/mediaextractrust`.
- [x] 2026-05-07: Add same-data MediaExtractRust vs Python benchmark evidence for implemented HTML and PDF text extraction slices. Evidence: `benchmarks/compare_mediaextract.py`, `benchmark-results/mediaextractrust-vs-python.md`.
- [x] 2026-05-07: Start ValidateRust with Pydantic-style object schemas, required/optional fields, primitive/list type checks, and simple constraints. Evidence: `crates/validaterust`.
- [x] 2026-05-07: Add same-data ValidateRust vs Pydantic benchmark evidence for the implemented validation slice. Evidence: `benchmarks/compare_pydantic.py`, `benchmark-results/validaterust-vs-pydantic.md`.
- [x] 2026-05-07: Start ImageRust with PPM decode, grayscale conversion, nearest-neighbor resize, and thresholding. Evidence: `crates/imagerust`.
- [x] 2026-05-07: Add same-data ImageRust vs Pillow benchmark evidence for the implemented image-processing slice. Evidence: `benchmarks/compare_pillow.py`, `benchmark-results/imagerust-vs-pillow.md`.
- [x] 2026-05-07: Start TextRust with NLTK-style word/punctuation tokenization, lowercase word frequencies, and word bigrams. Evidence: `crates/textrust`.
- [x] 2026-05-07: Add same-data TextRust vs NLTK benchmark evidence for the implemented tokenization slice. Evidence: `benchmarks/compare_nltk.py`, `benchmark-results/textrust-vs-nltk.md`.
- [x] 2026-05-07: Optimize NumRust transposed `f64`/`f32` GEMM dispatch on macOS and regenerate full external NumPy ASV-derived evidence at 52 of 53 NumRust wins. Evidence: `crates/numrs-core/src/blas.rs`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Add NumRust `nonzero` and broadcasted `where_select` selection APIs, checksum-validated targeted NumPy benchmark rows, and contiguous selection fast paths. Latest targeted same-data evidence: 10 NumRust wins, 0 NumPy wins, 0 checksum failures. Evidence: `crates/numrs-core/src/array.rs`, `benchmarks/compare_numpy.py`, `benchmark-results/numrust-vs-numpy.md`.
- [x] 2026-05-07: Start LearnRust with dense matrices, `StandardScaler`, nearest-centroid classification, accuracy, and confusion-matrix metrics. Evidence: `crates/learnrust`.
- [x] 2026-05-07: Add same-data LearnRust vs scikit-learn benchmark evidence for the implemented preprocessing, nearest-centroid, and metrics slice. Evidence: `benchmarks/compare_sklearn.py`, `benchmark-results/learnrust-vs-sklearn.md`.
- [x] 2026-05-07: Add NumRust `flip`, `moveaxis`, and `roll`, plus pinned NumPy ASV manipulation rows for those operations. Step evidence: 65 NumRust wins, 2 NumPy wins, 8.47x geomean, 0 checksum failures. Evidence: `crates/numrs-core/src/array.rs`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Add same-dtype NumRust `broadcast_arrays` and the pinned NumPy ASV `BroadcastArrays.time_broadcast_arrays` row. Step evidence: 67 NumRust wins, 1 NumPy win, 8.67x geomean, 0 checksum failures. Evidence: `crates/numrs-core/src/array.rs`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Expand pinned NumPy ASV broadcast parameter coverage with four more `BroadcastArrays` and `BroadcastArraysTo` rows. Step evidence: 71 NumRust wins, 1 NumPy win, 9.07x geomean, 0 checksum failures. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Add pinned NumPy ASV `float32` broadcast manipulation rows for `BroadcastArrays` and `BroadcastArraysTo`. Step evidence: 73 NumRust wins, 1 NumPy win, 9.23x geomean, 0 checksum failures. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Add pinned NumPy ASV `int32` broadcast manipulation rows for `BroadcastArrays` and `BroadcastArraysTo`. Step evidence: 76 NumRust wins, 0 NumPy wins, 9.46x geomean, 0 checksum failures. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Add pinned NumPy ASV `float32` and `int32` `ConcatenateStackArrays` axis-0 owned-output rows for `concatenate` and `stack`. Latest external ASV evidence: 78 NumRust wins, 2 NumPy wins, 8.79x geomean, 0 checksum failures; focused-loss evidence leaves 1 of 2 rows as a NumPy near-tie. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`, `benchmark-results/external-numpy-loss-focused.md`.

### Open

- [ ] 2026-05-07: Add more externally derived NumPy ASV cases without filtering out losses. Target: `benchmarks/external_numpy_cases.py`.
- [ ] 2026-05-07: Convert only benchmark-proven wins into new SIMD/BLAS/layout-specialized kernels. Target: `docs/performance.md`.
- [ ] 2026-05-07: Stabilize current full external NumPy linalg near ties without hiding rows. Latest full report has 2 NumPy-winning linalg near ties and 5 near-tie rows; the focused 3-pass rerun leaves 1 NumPy near-tie winner. Target: `crates/numrs-core/src/blas.rs`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07: Optimize visible NumRust targeted losses without hiding rows: `where_select_f64_loop` and near-tie `dot_f64_192`. Latest targeted same-data evidence: 10 NumRust wins, 0 NumPy wins, 1.67x geomean, 0 checksum failures. Evidence: `crates/numrs-core/src/array.rs`, `benchmark-results/numrust-vs-numpy.md`.
- [ ] 2026-05-07: Grow StatsRust and SciRust with more externally derived benchmarks. Target: `benchmarks/compare_statsmodels.py`, `benchmarks/compare_scipy.py`.
- [ ] 2026-05-07: Keep release and CI work deferred until the maintainer asks for it. Target: no `.github/workflows` or release automation for now.

### Production-Grade Gates

- [ ] 2026-05-07: Define a shared enterprise readiness rubric for every library: stable Rust API, docs, examples, error model, feature flags, manual verification gates, fuzz/property tests, external benchmarks, security review, and MSRV.
- [ ] 2026-05-07: Add per-crate `README.md` files showing immediate Rust usage, Python-equivalent workflows, supported scope, unsupported scope, and benchmark status.
- [ ] 2026-05-07: Add Criterion or equivalent Rust-native benchmark suites alongside Python comparison harnesses.
- [ ] 2026-05-07: Add corpus-based correctness tests for parsing, aggregation, graph algorithms, statistics, and document extraction.
- [x] 2026-05-07: Add benchmark dashboards that separate full-suite scores, focused reruns, checksum failures, unsupported cases, and near ties. Evidence: `docs/benchmark-dashboard.md`.

### Python Library Port Backlog

- [ ] 2026-05-07: Finish NumRust toward NumPy-scale coverage: broader ndarray APIs, full dtype semantics, linear algebra backends, random, FFT, masked/indexing semantics, serialization interop, and neutral benchmarks against NumPy.
- [ ] 2026-05-07: Expand SciRust toward SciPy-scale coverage: optimize, integrate, interpolate, sparse, signal, spatial, stats, special functions, and externally derived SciPy benchmarks.
- [ ] 2026-05-07: Expand StatsRust toward StatsModels-scale coverage: GLM families, time series, robust covariance, formula-like model building, diagnostics, and StatsModels comparison benchmarks.
- [ ] 2026-05-07: Build FrameRust for Pandas-style dataframes and data aggregation: groupby, joins, pivots, rolling/window ops, time series, missing data, CSV/Parquet/Arrow interop, and benchmarks against Pandas plus relevant Rust incumbents.
- [ ] 2026-05-07: Build GraphRust for NetworkX-style graph work: directed/undirected/multigraphs, traversal, shortest paths, centrality, PageRank, community detection, graph IO, and benchmarks against NetworkX and other graph libraries.
- [ ] 2026-05-07: Build LearnRust for scikit-learn-style classical ML: preprocessing, pipelines, metrics, linear models, trees, neighbors, clustering, model selection, serialization, and benchmarks against scikit-learn.
- [ ] 2026-05-07: Build PlotRust for Matplotlib/Seaborn/Plotly-style visualization: static plots, statistical charts, interactive exports, themes, notebooks/web embedding, and visual regression tests.
- [ ] 2026-05-07: Build UXRust for Python-style UI/UX app workflows inspired by Streamlit, Gradio, Dash, and NiceGUI: declarative controls, state, layouts, charts, async tasks, deployment, accessibility, and Rust-first component APIs.
- [ ] 2026-05-07: Build MediaExtractRust for instant extraction from PDFs, Office docs, images, scans, HTML, markdown, email, and archives: text, layout, tables, metadata, OCR hooks, batch pipelines, and throughput/accuracy benchmarks against PyMuPDF, pdfplumber, python-docx, Pillow, Tesseract wrappers, and unstructured-style pipelines.
- [ ] 2026-05-07: Build ImageRust for Pillow/scikit-image-style image operations: decoding, transforms, filters, morphology, segmentation primitives, color management, metadata, and image quality benchmarks.
- [ ] 2026-05-07: Build TextRust for NLTK/spaCy-style NLP utilities: tokenization, normalization, stemming/lemmatization hooks, vectorization, entity/rule pipelines, and benchmarks against Python NLP tooling.
- [ ] 2026-05-07: Build RequestRust for requests/httpx-style high-level HTTP workflows if existing Rust clients do not cover the Python ergonomics: sessions, retries, auth, streaming, testing fixtures, and API compatibility examples.
- [ ] 2026-05-07: Build ValidateRust for Pydantic-style data validation: schema derivation, coercion, rich errors, JSON schema, serde integration, and benchmarks against Pydantic.
- [ ] 2026-05-07: Build WebAppRust for FastAPI-style service development: typed routing, validation, OpenAPI, auth hooks, background jobs, test client, and benchmarks against FastAPI where meaningful.
- [ ] 2026-05-07 deferred: Add CI only when requested. Target: future `.github/workflows/*`.
- [ ] 2026-05-07 deferred: Add release automation only when requested. Target: future release checklist and crate publishing workflow.

## Commands

```sh
cargo fmt --all --check
cargo test --workspace
cargo run -p numrs-core --example basic
cargo run --release -p numrs-core --example microbench
uv run --with numpy benchmarks/compare_numpy.py
uv run --with numpy --with pandas benchmarks/compare_pandas.py
uv run --with networkx benchmarks/compare_networkx.py
uv run --with beautifulsoup4 --with pypdf benchmarks/compare_mediaextract.py
uv run --with pydantic benchmarks/compare_pydantic.py
uv run --with pillow benchmarks/compare_pillow.py
uv run --with nltk benchmarks/compare_nltk.py
uv run --with numpy --with scikit-learn benchmarks/compare_sklearn.py
uv run benchmarks/external_sources.py --update-lock
uv run --with numpy python benchmarks/external_numpy_cases.py
uv run benchmarks/verify_array_api_namespace.py
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --focused --maxfail 25 --output-stem array-api-tests-focused-probe
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --full --maxfail 25 --output-stem array-api-tests-full-maxfail
uv run benchmarks/compare_statsmodels.py
uv run --with numpy --with scipy benchmarks/compare_scipy.py
```

## Layout

- `crates/numrs-core`: pure Rust NumPy-core flagship crate.
- `crates/numrust-python`: PyO3-backed Python namespace bridge for NumRust.
- `crates/statsrust`: StatsModels-style statistics crate.
- `crates/scirust`: SciPy-style numerical routines crate.
- `crates/rigortrail`: new benchmark/evaluation evidence-ledger crate.
- `crates/framerust`: Pandas-style data aggregation crate.
- `crates/graphrust`: NetworkX-style graph analytics crate.
- `crates/mediaextractrust`: document/media text extraction crate.
- `crates/validaterust`: Pydantic-style schema validation crate.
- `crates/imagerust`: Pillow-style image processing crate.
- `crates/textrust`: NLTK-style text processing crate.
- `crates/learnrust`: scikit-learn-style classical ML crate.
- `docs/research.md`: research notes and source map.
- `docs/novel-library-research.md`: research notes for the new invented crate.
- `docs/architecture.md`: architecture decisions and optimization roadmap.
- `docs/benchmark-dashboard.md`: benchmark scorecard, rerun matrix, and claim gates.
- `docs/performance.md`: current fast paths and benchmark hook.
- `docs/external-evidence.md`: pinned online benchmark and conformance sources.
- `docs/python-array-api.md`: Python namespace bridge and conformance roadmap.
- `docs/statsmodels-port.md`: second library port notes.
- `docs/scipy-port.md`: third library port notes.
- `docs/rigortrail.md`: RigorTrail design and usage notes.
- `benchmark-results/numrust-vs-numpy.md`: current NumRust vs NumPy evidence.
- `benchmark-results/framerust-vs-pandas.md`: current FrameRust vs Pandas evidence for the implemented aggregation slice.
- `benchmark-results/graphrust-vs-networkx.md`: current GraphRust vs NetworkX evidence for the implemented BFS slice.
- `benchmark-results/mediaextractrust-vs-python.md`: current MediaExtractRust vs Python evidence for implemented extraction slices.
- `benchmark-results/validaterust-vs-pydantic.md`: current ValidateRust vs Pydantic evidence for the implemented validation slice.
- `benchmark-results/imagerust-vs-pillow.md`: current ImageRust vs Pillow evidence for the implemented image-processing slice.
- `benchmark-results/textrust-vs-nltk.md`: current TextRust vs NLTK evidence for the implemented tokenization slice.
- `benchmark-results/learnrust-vs-sklearn.md`: current LearnRust vs scikit-learn evidence for the implemented preprocessing, nearest-centroid, and metrics slice.
- `benchmark-results/external-numpy-asv-inspired.md`: externally derived NumPy ASV evidence.
- `benchmark-results/array-api-tests-focused-probe.md`: pinned upstream focused Array API probe, currently 1109 passed, 4 skipped, out of 1113 collected.
- `benchmark-results/array-api-tests-full-maxfail.md`: pinned upstream full Array API 2023.12 probe, currently 1161 passed, 58 skipped, out of 1219 collected.
- `benchmark-results/statsrust-vs-statsmodels.md`: same-data StatsRust vs StatsModels evidence.
- `benchmark-results/scirust-vs-scipy.md`: same-data SciRust vs SciPy evidence.
- `logs/journey.md`: detailed build journey.
