# Novel Library Research

## Question

Find a Rust library opportunity that is not already covered by common crates, then build it.

## Search Notes

Checked current Rust benchmarking and evidence/provenance directions:

- [Criterion.rs](https://bheisler.github.io/criterion.rs/book/index.html) is a statistics-driven Rust microbenchmarking library.
- [Zench](https://docs.rs/zench/latest/zench/) is a newer benchmark crate focused on cargo-test integration, reporting, and performance assertions.
- [franken_evidence](https://docs.rs/franken-evidence/latest/franken_evidence/) is an evidence-ledger crate for FrankenSuite decisions. Provenance and evidence crates exist, but the visible examples are supply-chain, container-key, AI decision, workflow, or audit-log oriented.

Gap found: a small Rust library that models benchmark claims as evidence ledgers:

- source pins,
- supported cases,
- unsupported cases,
- checksum status,
- win/geomean scoring,
- and a hard gate that prevents global claims when coverage is incomplete.

That gap matches this repo's central risk: benchmarks can be cherry-picked or skewed unless unsupported cases and source provenance are first-class data.

## Built

`crates/rigortrail` implements that idea. It is intentionally not a benchmark runner. Criterion and Zench already run measurements; RigorTrail records whether the resulting claim is defensible.
