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

## Ecosystem

- Grow StatsRust and SciRust with externally derived benchmarks where upstream suites exist.
- Use RigorTrail to gate broad replacement claims on source pins, checksums, unsupported cases, and benchmark scope.

## Release Readiness

- [ ] Defer CI until Aditya asks for it.
- [ ] Defer release automation and crates.io publishing metadata until Aditya asks for it.
- Keep TODO changes committed with the implementation commits they describe.

## Enterprise Library Backlog

- [ ] Define the production-grade readiness rubric shared by every library.
- [ ] FrameRust: Pandas-style dataframes, joins, groupby, windows, Arrow/Parquet/CSV, and benchmark evidence.
- [ ] GraphRust: NetworkX-style graph APIs and graph algorithm benchmarks.
- [ ] UXRust: Streamlit/Gradio/Dash/NiceGUI-style Rust UI/UX workflow library.
- [ ] MediaExtractRust: ultra-fast extraction from PDFs, Office docs, images, scans, HTML, markdown, email, and archives.
- [ ] LearnRust: scikit-learn-style classical ML APIs and benchmarks.
- [ ] PlotRust: Matplotlib/Seaborn/Plotly-style visualization APIs.
- [ ] ImageRust: Pillow/scikit-image-style image processing APIs.
- [ ] TextRust: NLTK/spaCy-style text processing APIs.
- [ ] ValidateRust: Pydantic-style validation APIs.
- [ ] WebAppRust: FastAPI-style typed web app APIs.
