# TODO

This file is intentionally committed and should be updated in normal git commits as work progresses.

## Benchmark Integrity

- Keep full external NumPy ASV-derived reports separate from focused loss reruns.
- Re-run `uv run benchmarks/external_sources.py --verify-pinned` before publishing benchmark claims.
- Preserve raw pass samples in JSON artifacts for any benchmark result cited in docs.

## NumRust

- Expand supported NumPy ASV cases without filtering losses.
- Add more dtype coverage for integer, boolean, and complex Array API behavior.
- Add portable SIMD backends where stable Rust support is strong enough.
- Continue replacing generic stride paths with layout-specialized kernels only when real benchmarks improve.
- [x] NumRust transposed GEMM v1: use column-major-swapped CBLAS for macOS transposed `f64`/`f32` 2-D GEMM and regenerate full external NumPy ASV-derived evidence at 52 of 53 NumRust wins.
- [x] NumRust selection v1: add `nonzero`, broadcasted `where_select`, checksum-validated targeted NumPy benchmark rows, and contiguous selection fast paths. Latest targeted evidence: 10 NumRust wins, 0 NumPy wins, 0 checksum failures.
- [x] NumRust targeted loss cleanup: optimize `where_select_f64_loop` and near-tie `dot_f64_192` without hiding Python-winning rows. Latest targeted evidence: 10 NumRust wins, 0 NumPy wins, 1.67x geomean, 0 checksum failures.
- [x] NumRust manipulation v2: add `flip`, `moveaxis`, and `roll`, plus pinned NumPy ASV manipulation rows. Step evidence: 65 NumRust wins, 2 NumPy wins, 8.47x geomean, 0 checksum failures.
- [x] NumRust broadcast arrays v1: add same-dtype `broadcast_arrays` and pinned NumPy ASV `BroadcastArrays.time_broadcast_arrays` evidence. Step evidence: 67 NumRust wins, 1 NumPy win, 8.67x geomean, 0 checksum failures.
- [x] NumRust broadcast ASV parameter expansion: add four more pinned `BroadcastArrays` and `BroadcastArraysTo` parameter rows. Step evidence: 71 NumRust wins, 1 NumPy win, 9.07x geomean, 0 checksum failures.
- [x] NumRust broadcast dtype expansion: add pinned `float32` `BroadcastArrays` and `BroadcastArraysTo` rows. Step evidence: 73 NumRust wins, 1 NumPy win, 9.23x geomean, 0 checksum failures.
- [x] NumRust integer broadcast dtype expansion: add pinned `int32` `BroadcastArrays` and `BroadcastArraysTo` rows. Step evidence: 76 NumRust wins, 0 NumPy wins, 9.46x geomean, 0 checksum failures.
- [x] NumRust concatenate/stack dtype expansion: add pinned `float32` and `int32` `ConcatenateStackArrays` axis-0 owned-output rows. Step evidence: 78 NumRust wins, 2 NumPy wins, 8.79x geomean, 0 checksum failures; focused-loss evidence leaves 1 of 2 rows as a NumPy near-tie.
- [x] NumRust concatenate/stack axis-1 dtype expansion: add pinned `float32` and `int32` `ConcatenateStackArrays` axis-1 owned-output rows. Step evidence: 82 NumRust wins, 2 NumPy wins, 8.41x geomean, 0 checksum failures; focused-loss evidence flips both current NumPy-winning rows to NumRust.
- [x] NumRust stats dtype expansion: add `float32` all-axis `mean`, `var`, and `std` support plus pinned `StatsReductions(dtype=float32)` rows for min, max, mean, std, prod, and var. Step evidence: 89 NumRust wins, 1 NumPy win, 10.84x geomean, 0 checksum failures; focused-loss evidence flips the current NumPy-winning row to NumRust.
- [x] NumRust integer stats dtype expansion: add `int64` all-axis `mean`, `var`, and `std` support plus pinned `StatsReductions(dtype=int64)` rows for min, max, mean, std, prod, and var. Step evidence: 95 NumRust wins, 1 NumPy win, 14.03x geomean, 0 checksum failures; focused-loss evidence flips the current NumPy-winning row to NumRust.
- [x] NumRust unsigned integer stats dtype expansion: add `uint64` all-axis `mean`, `var`, and `std` support plus pinned `StatsReductions(dtype=uint64)` rows for min, max, mean, std, prod, and var. Step evidence: 101 NumRust wins, 1 NumPy win, 17.47x geomean, 0 checksum failures; focused-loss evidence leaves the current NumPy-winning row as a NumPy near-tie.
- [x] NumRust bool stats dtype expansion: add `bool` all-axis `mean`, `var`, `std`, and `prod` support plus pinned `StatsReductions(dtype=bool_)` rows for min, max, mean, std, prod, and var. Latest external evidence: 108 NumRust wins, 0 NumPy wins, 22.07x geomean, 0 checksum failures; focused-loss evidence has 0 current loss rows.
- [ ] NumRust external linalg stability cleanup: stabilize current full-report near ties without hiding rows. Latest full report has 0 NumPy-winning supported rows and 6 linalg near-tie rows; keep stabilizing near ties without hiding rows.

## Ecosystem

- Grow StatsRust and SciRust with externally derived benchmarks where upstream suites exist.
- Use RigorTrail to gate broad replacement claims on source pins, checksums, unsupported cases, and benchmark scope.
- [x] Benchmark dashboard v0: table-first README snapshot plus dashboard doc for full-suite scores, focused reruns, checksum status, rerun commands, and claim gates.

## Release Readiness

- [ ] Defer CI until the maintainer asks for it.
- [ ] Defer release automation and crates.io publishing metadata until the maintainer asks for it.
- Keep TODO changes committed immediately after the implementation commits they describe when separate TODO commits are requested.

## Enterprise Library Backlog

- [ ] Define the production-grade readiness rubric shared by every library.
- [ ] FrameRust: Pandas-style dataframes, joins, groupby, windows, Arrow/Parquet/CSV, and benchmark evidence.
- [x] FrameRust v0: typed columns, validated frames, first-seen-order groupby, and numeric aggregations.
- [x] FrameRust benchmark v0: same-data groupby aggregation comparison against Pandas.
- [ ] GraphRust: NetworkX-style graph APIs and graph algorithm benchmarks.
- [x] GraphRust v0: CSR graphs, BFS distances, connected components, and PageRank.
- [x] GraphRust benchmark v0: same-data BFS comparison against NetworkX.
- [ ] UXRust: Streamlit/Gradio/Dash/NiceGUI-style Rust UI/UX workflow library.
- [ ] MediaExtractRust: ultra-fast extraction from PDFs, Office docs, images, scans, HTML, markdown, email, and archives.
- [x] MediaExtractRust v0: text/Markdown normalization, HTML text extraction, and uncompressed PDF literal text extraction.
- [x] MediaExtractRust benchmark v0: same-data HTML and uncompressed PDF text extraction comparison against Python libraries.
- [ ] LearnRust: scikit-learn-style classical ML APIs and benchmarks.
- [x] LearnRust v0: dense matrices, StandardScaler, nearest-centroid classification, accuracy, and confusion matrix.
- [x] LearnRust benchmark v0: same-data StandardScaler, nearest-centroid, and metrics comparison against scikit-learn.
- [ ] PlotRust: Matplotlib/Seaborn/Plotly-style visualization APIs.
- [ ] ImageRust: Pillow/scikit-image-style image processing APIs.
- [x] ImageRust v0: PPM decode, grayscale conversion, nearest-neighbor resize, and thresholding.
- [x] ImageRust benchmark v0: same-data PPM grayscale, resize, and threshold comparison against Pillow.
- [ ] TextRust: NLTK/spaCy-style text processing APIs.
- [x] TextRust v0: NLTK-style word/punctuation tokenization, lowercase word frequencies, and word bigrams.
- [x] TextRust benchmark v0: same-data wordpunct tokenization comparison against NLTK.
- [ ] ValidateRust: Pydantic-style validation APIs.
- [x] ValidateRust v0: object schemas, required/optional fields, primitive/list type checks, and simple constraints.
- [x] ValidateRust benchmark v0: same-data schema validation comparison against Pydantic.
- [ ] WebAppRust: FastAPI-style typed web app APIs.
