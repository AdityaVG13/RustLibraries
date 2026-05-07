# Benchmark Dashboard

This is the quick read for benchmark status. Every number below comes from committed benchmark artifacts in `benchmark-results/`. A row claims only the implemented slice named in that row. Global Python-library replacement stays false until API coverage, conformance, and neutral benchmarks cover the full upstream surface.

| Claim type | Current answer |
| --- | --- |
| Is NumRust a full NumPy replacement? | No. |
| Is any crate here full parity with its Python target? | No. |
| Can a crate claim it is faster on a benchmarked slice? | Yes, only for rows with matching checksums and committed raw artifacts. |
| Can focused reruns replace full-suite scores? | No. They are optimization signals until the full suite is rerun. |
| What happens when Python wins? | The row remains visible and the roadmap targets the loss. |

## Current Scorecard

| Area | Rust crate | Python baseline | Evidence tier | Cases | Rust wins | Python wins | Full parity? | Speedup summary | Checksum failures | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| NumPy targeted | `numrs-core` | NumPy 2.4.4 | Same-data implemented slice | 10 | 10 | 0 | No | 1.67x geomean | 0 | [`numrust-vs-numpy.md`](../benchmark-results/numrust-vs-numpy.md) |
| NumPy core | `numrs-core` | NumPy 2.4.4 | Pinned NumPy ASV-derived suite | 67 supported | 65 | 2 | No | 8.47x geomean | 0 | [`external-numpy-asv-inspired.md`](../benchmark-results/external-numpy-asv-inspired.md) |
| NumPy loss triage | `numrs-core` | NumPy 2.4.4 | Focused rerun of current NumPy wins | 2 | 2 | 0 | No | 1.00x to 1.02x focused wins | 0 | [`external-numpy-loss-focused.md`](../benchmark-results/external-numpy-loss-focused.md) |
| Statistics | `statsrust` | StatsModels 0.14.6 | Same-data implemented slice | 4 | 4 | 0 | No | 3.51x geomean | 0 | [`statsrust-vs-statsmodels.md`](../benchmark-results/statsrust-vs-statsmodels.md) |
| Scientific routines | `scirust` | SciPy 1.17.1 | SciPy ASV translations plus same-data slice | 9 | 9 | 0 | No | 19.11x geomean | 0 | [`scirust-vs-scipy.md`](../benchmark-results/scirust-vs-scipy.md) |
| Data aggregation | `framerust` | Pandas 3.0.2 | Same-data implemented slice | 1 | 1 | 0 | No | 2.14x | 0 | [`framerust-vs-pandas.md`](../benchmark-results/framerust-vs-pandas.md) |
| Graph analytics | `graphrust` | NetworkX 3.6.1 | Same-data implemented slice | 1 | 1 | 0 | No | 27.15x | 0 | [`graphrust-vs-networkx.md`](../benchmark-results/graphrust-vs-networkx.md) |
| Document extraction | `mediaextractrust` | Python extraction libraries | Same-data implemented slice | 2 | 2 | 0 | No | 81.21x to 122.93x | 0 | [`mediaextractrust-vs-python.md`](../benchmark-results/mediaextractrust-vs-python.md) |
| Data validation | `validaterust` | Pydantic 2.13.4 | Same-data implemented slice | 1 | 1 | 0 | No | 35.69x | 0 | [`validaterust-vs-pydantic.md`](../benchmark-results/validaterust-vs-pydantic.md) |
| Image processing | `imagerust` | Pillow 12.2.0 | Same-data implemented slice | 1 | 1 | 0 | No | 16.53x | 0 | [`imagerust-vs-pillow.md`](../benchmark-results/imagerust-vs-pillow.md) |
| Text processing | `textrust` | NLTK 3.9.4 | Same-data implemented slice | 1 | 1 | 0 | No | 23.92x | 0 | [`textrust-vs-nltk.md`](../benchmark-results/textrust-vs-nltk.md) |
| Classical ML | `learnrust` | scikit-learn 1.8.0 | Same-data implemented slice | 3 | 3 | 0 | No | 6.53x geomean | 0 | [`learnrust-vs-sklearn.md`](../benchmark-results/learnrust-vs-sklearn.md) |

## NumRust External Detail

| Metric | Value |
| --- | ---: |
| Pinned NumPy ASV commit | `80b1a07494964733f7d4571781608238f500e2dd` |
| Pinned Array API tests commit | `55fcc60179efa2680ddd6cd926ddf17b83530e2b` |
| Full passes per engine | 5 |
| Supported external cases | 67 |
| Unsupported external cases tracked | 1 |
| NumRust wins | 65 |
| NumPy wins | 2 |
| Geomean speedup vs NumPy | 8.47x |
| Near-tie cases within 2% | 4 |
| Checksum failures | 0 |
| Global NumPy replacement claim | false |

The remaining full-suite NumPy wins are `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` at 0.975x and `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` at 0.998x in the authoritative full report. The focused 3-pass rerun flips both rows to NumRust at 1.004x and 1.023x, with the first still a near tie. The full report remains authoritative, and both rows remain visible as timing-stability optimization targets.

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
| NumPy targeted smoke benchmark | `uv run --with numpy benchmarks/compare_numpy.py` | `benchmark-results/numrust-vs-numpy.md` |
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
| scikit-learn comparison | `uv run --with numpy --with scikit-learn benchmarks/compare_sklearn.py` | `benchmark-results/learnrust-vs-sklearn.md` |

## Claim Gates

| Gate | Required to claim "better than Python" | Current status |
| --- | --- | --- |
| Same input data | Rust and Python receive the same generated or pinned inputs | Met for current benchmark artifacts |
| Raw artifacts | JSON plus Markdown reports are committed | Met for current benchmark artifacts |
| Checksum parity | Output checksums must match or the row fails | 0 failures in the cited reports |
| Loss visibility | Python-winning rows must remain visible | Met in NumPy ASV-derived report |
| Scope honesty | Slice wins cannot be marketed as full-library parity | Enforced in reports and docs |
| Full parity | A Rust crate must cover the Python library's real API surface before any full-replacement claim | Not met for any crate yet |
| External evidence | Prefer pinned upstream benchmarks where available | Met for NumPy ASV, Array API, and part of SciPy |
| Enterprise readiness | Stable APIs, docs, examples, error model, fuzz/property tests, and security review | In progress |

## Next Optimization Targets

| Priority | Target | Why |
| ---: | --- | --- |
| 1 | NumRust transposed-view linalg timing stability | The authoritative full run still has two NumPy wins on transposed matmul rows. The focused rerun flips both, so the full-suite timing remains the target. |
| 2 | Broader NumRust ASV coverage | The current supported slice ranks higher, but full NumPy-scale scope needs more translated cases. |
| 3 | Broader externally derived SciPy and StatsModels cases | Current wins are strong but the benchmark surface is narrow. |
| 4 | More cases for FrameRust, GraphRust, MediaExtractRust, ValidateRust, ImageRust, TextRust, and LearnRust | The current slices all beat Python, but production-grade parity needs broader workloads. |
| 5 | Per-crate README files and Rust-native benchmark suites | Public users need crate-local usage, scope, and reproducible performance gates. |
