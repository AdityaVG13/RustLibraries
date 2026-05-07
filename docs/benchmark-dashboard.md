# Benchmark Dashboard

This is the quick read for benchmark status. Every number below comes from committed benchmark artifacts in `benchmark-results/`. A row claims only the implemented slice named in that row. Global Python-library replacement stays false until API coverage, conformance, and neutral benchmarks cover the full upstream surface.

## Current Scorecard

| Area | Rust crate | Python baseline | Evidence tier | Cases | Rust wins | Python wins | Speedup summary | Checksum failures | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NumPy core | `numrs-core` | NumPy 2.4.4 | Pinned NumPy ASV-derived suite | 53 supported | 43 | 10 | 8.92x geomean | 0 | [`external-numpy-asv-inspired.md`](../benchmark-results/external-numpy-asv-inspired.md) |
| NumPy loss triage | `numrs-core` | NumPy 2.4.4 | Focused rerun of prior NumPy wins | 10 | 9 | 1 | 7 near ties within 2% | 0 | [`external-numpy-loss-focused.md`](../benchmark-results/external-numpy-loss-focused.md) |
| Statistics | `statsrust` | StatsModels 0.14.6 | Same-data implemented slice | 4 | 4 | 0 | 3.51x geomean | 0 | [`statsrust-vs-statsmodels.md`](../benchmark-results/statsrust-vs-statsmodels.md) |
| Scientific routines | `scirust` | SciPy 1.17.1 | SciPy ASV translations plus same-data slice | 9 | 9 | 0 | 19.11x geomean | 0 | [`scirust-vs-scipy.md`](../benchmark-results/scirust-vs-scipy.md) |
| Data aggregation | `framerust` | Pandas 3.0.2 | Same-data implemented slice | 1 | 1 | 0 | 2.14x | 0 | [`framerust-vs-pandas.md`](../benchmark-results/framerust-vs-pandas.md) |
| Graph analytics | `graphrust` | NetworkX 3.6.1 | Same-data implemented slice | 1 | 1 | 0 | 27.15x | 0 | [`graphrust-vs-networkx.md`](../benchmark-results/graphrust-vs-networkx.md) |
| Document extraction | `mediaextractrust` | Python extraction libraries | Same-data implemented slice | 2 | 2 | 0 | 81.21x to 122.93x | 0 | [`mediaextractrust-vs-python.md`](../benchmark-results/mediaextractrust-vs-python.md) |
| Data validation | `validaterust` | Pydantic 2.13.4 | Same-data implemented slice | 1 | 1 | 0 | 35.69x | 0 | [`validaterust-vs-pydantic.md`](../benchmark-results/validaterust-vs-pydantic.md) |
| Image processing | `imagerust` | Pillow 12.2.0 | Same-data implemented slice | 1 | 1 | 0 | 16.53x | 0 | [`imagerust-vs-pillow.md`](../benchmark-results/imagerust-vs-pillow.md) |
| Text processing | `textrust` | NLTK 3.9.4 | Same-data implemented slice | 1 | 1 | 0 | 23.92x | 0 | [`textrust-vs-nltk.md`](../benchmark-results/textrust-vs-nltk.md) |

## NumRust External Detail

| Metric | Value |
| --- | ---: |
| Pinned NumPy ASV commit | `80b1a07494964733f7d4571781608238f500e2dd` |
| Pinned Array API tests commit | `55fcc60179efa2680ddd6cd926ddf17b83530e2b` |
| Full passes per engine | 5 |
| Supported external cases | 53 |
| Unsupported external cases tracked | 1 |
| NumRust wins | 43 |
| NumPy wins | 10 |
| Geomean speedup vs NumPy | 8.92x |
| Near-tie cases within 2% | 1 |
| Checksum failures | 0 |
| Global NumPy replacement claim | false |

The remaining full-suite NumPy wins are concentrated in large 2-D linalg transpose/copy layouts and contiguous scalar multiply. The focused rerun flips 9 of those 10 rows to NumRust wins, but the full 53-case ASV-derived report remains the authoritative score until the whole suite is rerun.

## Conformance Snapshot

| Target | Source | Result | Report |
| --- | --- | --- | --- |
| Array API focused probe | Pinned upstream `array-api-tests` 2023.12 | 1109 passed, 4 skipped, 1113 collected | [`array-api-tests-focused-probe.md`](../benchmark-results/array-api-tests-focused-probe.md) |
| Array API full probe | Pinned upstream `array-api-tests` 2023.12 | 1161 passed, 58 skipped, 1219 collected | [`array-api-tests-full-maxfail.md`](../benchmark-results/array-api-tests-full-maxfail.md) |
| Source lock verification | Pinned raw upstream files with SHA-256 hashes | Re-runnable with `--verify-pinned` | [`external-source-lock.json`](../benchmark-results/external-source-lock.json) |

## Rerun Matrix

| Goal | Command | Primary output |
| --- | --- | --- |
| Rust tests | `cargo test --workspace` | Workspace correctness status |
| Rust formatting | `cargo fmt --all --check` | Formatting gate |
| Rust linting | `cargo clippy --workspace --all-targets -- -D warnings` | Warning-free lint gate |
| NumPy targeted smoke benchmark | `uv run benchmarks/compare_numpy.py` | `benchmark-results/numrust-vs-numpy.md` |
| NumPy ASV-derived benchmark | `uv run --with numpy python benchmarks/external_numpy_cases.py` | `benchmark-results/external-numpy-asv-inspired.md` |
| NumPy focused loss rerun | `uv run --with numpy python benchmarks/external_numpy_cases.py --rerun-losses --loss-passes 3` | `benchmark-results/external-numpy-loss-focused.md` |
| External source lock verification | `uv run benchmarks/external_sources.py --verify-pinned` | Hash verification for pinned sources |
| StatsModels comparison | `uv run benchmarks/compare_statsmodels.py` | `benchmark-results/statsrust-vs-statsmodels.md` |
| SciPy comparison | `uv run --with numpy --with scipy benchmarks/compare_scipy.py` | `benchmark-results/scirust-vs-scipy.md` |
| Pandas comparison | `uv run --with numpy --with pandas benchmarks/compare_pandas.py` | `benchmark-results/framerust-vs-pandas.md` |
| NetworkX comparison | `uv run --with networkx benchmarks/compare_networkx.py` | `benchmark-results/graphrust-vs-networkx.md` |
| Media extraction comparison | `uv run --with beautifulsoup4 --with pypdf benchmarks/compare_mediaextract.py` | `benchmark-results/mediaextractrust-vs-python.md` |
| Pydantic comparison | `uv run --with pydantic benchmarks/compare_pydantic.py` | `benchmark-results/validaterust-vs-pydantic.md` |
| Pillow comparison | `uv run --with pillow benchmarks/compare_pillow.py` | `benchmark-results/imagerust-vs-pillow.md` |
| NLTK comparison | `uv run --with nltk benchmarks/compare_nltk.py` | `benchmark-results/textrust-vs-nltk.md` |

## Claim Gates

| Gate | Required to claim "better than Python" | Current status |
| --- | --- | --- |
| Same input data | Rust and Python receive the same generated or pinned inputs | Met for current benchmark artifacts |
| Raw artifacts | JSON plus Markdown reports are committed | Met for current benchmark artifacts |
| Checksum parity | Output checksums must match or the row fails | 0 failures in the cited reports |
| Loss visibility | Python-winning rows must remain visible | Met in NumPy ASV-derived report |
| Scope honesty | Slice wins cannot be marketed as full-library parity | Enforced in reports and docs |
| External evidence | Prefer pinned upstream benchmarks where available | Met for NumPy ASV, Array API, and part of SciPy |
| Enterprise readiness | Stable APIs, docs, examples, error model, fuzz/property tests, and security review | In progress |

## Next Optimization Targets

| Priority | Target | Why |
| ---: | --- | --- |
| 1 | NumRust large 2-D linalg transpose/copy paths | These account for most full-suite NumPy wins. |
| 2 | NumRust contiguous scalar multiply | Still a NumPy win in the authoritative full run. |
| 3 | Broader externally derived SciPy and StatsModels cases | Current wins are strong but the benchmark surface is narrow. |
| 4 | More cases for FrameRust, GraphRust, MediaExtractRust, ValidateRust, ImageRust, and TextRust | The current slices all beat Python, but production-grade parity needs broader workloads. |
| 5 | Per-crate README files and Rust-native benchmark suites | Public users need crate-local usage, scope, and reproducible performance gates. |
