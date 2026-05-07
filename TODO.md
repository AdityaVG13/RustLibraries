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

## Ecosystem

- Grow StatsRust and SciRust with externally derived benchmarks where upstream suites exist.
- Use RigorTrail to gate broad replacement claims on source pins, checksums, unsupported cases, and benchmark scope.
- [x] Benchmark dashboard v0: table-first README snapshot plus dashboard doc for full-suite scores, focused reruns, checksum status, rerun commands, and claim gates.

## Release Readiness

- [ ] Defer CI until the maintainer asks for it.
- [ ] Defer release automation and crates.io publishing metadata until the maintainer asks for it.
- Keep TODO changes committed with the implementation commits they describe.

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
