# RigorTrail

`rigortrail` is a new Rust library for bias-resistant benchmark and evaluation claims.

## Core Idea

Most benchmark tools measure speed. RigorTrail records whether a claim is honest:

- Every case must cite a pinned source.
- Duplicate cases and missing sources are rejected.
- Failed checksums invalidate the ledger.
- Unsupported cases are counted, not hidden.
- A broad global claim is blocked until unsupported cases are gone.

## Example

```rust
use rigortrail::{CaseEvidence, EvidenceLedger, SourcePin, UnsupportedCase};

let ledger = EvidenceLedger::new("Candidate outranks baseline on supported external cases")
    .add_source(SourcePin::new("numpy-asv", "https://example.invalid/asv.py", "abc123"))
    .add_case(CaseEvidence::timed("sum", "numpy-asv", 2.0, 10.0))
    .add_case(CaseEvidence::timed("dot", "numpy-asv", 5.0, 4.0))
    .add_unsupported(UnsupportedCase::new("array-api-tests", "full Python Array API surface incomplete"));

let score = ledger.score().unwrap();
assert!(score.ranked_higher_by_wins);
assert!(!score.global_claim_allowed);
```

## Current Scope

- Evidence ledger builder.
- Source pins.
- Supported benchmark cases with timings and checksum status.
- Unsupported case accounting.
- Validation errors for missing sources, duplicate cases, failed checksums, invalid timings, and empty unsupported reasons.
- Score summary with candidate wins, baseline wins, ties, geometric mean speedup, supported/unsupported counts, and global-claim gate.
- Markdown rendering for reports.
