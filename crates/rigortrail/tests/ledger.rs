use rigortrail::{
    CaseEvidence, ChecksumStatus, EvidenceLedger, SourcePin, UnsupportedCase, ValidationError,
    Winner,
};

#[test]
fn scores_and_gates_benchmark_claims() {
    let ledger = EvidenceLedger::new("Candidate outranks baseline on supported external cases")
        .add_source(SourcePin::new(
            "numpy-asv",
            "https://example.invalid/asv.py",
            "abc123",
        ))
        .add_case(CaseEvidence::timed("sum", "numpy-asv", 2.0, 10.0))
        .add_case(CaseEvidence::timed("dot", "numpy-asv", 5.0, 4.0))
        .add_case(CaseEvidence::timed("take", "numpy-asv", 3.0, 6.0))
        .add_unsupported(UnsupportedCase::new(
            "array-api-tests",
            "needs a Python namespace",
        ));

    let score = ledger.score().unwrap();
    assert_eq!(score.supported_cases, 3);
    assert_eq!(score.unsupported_cases, 1);
    assert_eq!(score.candidate_wins, 2);
    assert_eq!(score.baseline_wins, 1);
    assert!(score.ranked_higher_by_wins);
    assert!(!score.global_claim_allowed);
    assert!((score.geomean_speedup - (5.0_f64 * 0.8 * 2.0).powf(1.0 / 3.0)).abs() < 1e-12);

    let markdown = ledger.render_markdown().unwrap();
    assert!(markdown.contains("Global claim allowed: false"));
    assert!(markdown.contains("array-api-tests"));
}

#[test]
fn allows_global_claim_only_when_no_uncovered_cases_remain() {
    let ledger = EvidenceLedger::new("Candidate is fully covered")
        .add_source(SourcePin::new(
            "source",
            "https://example.invalid/source",
            "abc123",
        ))
        .add_case(CaseEvidence::timed("fast", "source", 1.0, 2.0));

    let score = ledger.score().unwrap();
    assert_eq!(ledger.cases()[0].winner(), Winner::Candidate);
    assert!(score.global_claim_allowed);
}

#[test]
fn rejects_missing_sources_duplicate_cases_and_checksum_failures() {
    let missing_source =
        EvidenceLedger::new("bad").add_case(CaseEvidence::timed("case", "missing", 1.0, 2.0));
    assert!(matches!(
        missing_source.validate().unwrap_err(),
        ValidationError::MissingCaseSource { .. }
    ));

    let duplicate_case = EvidenceLedger::new("bad")
        .add_source(SourcePin::new("s", "https://example.invalid", "abc123"))
        .add_case(CaseEvidence::timed("case", "s", 1.0, 2.0))
        .add_case(CaseEvidence::timed("case", "s", 1.0, 2.0));
    assert_eq!(
        duplicate_case.validate().unwrap_err(),
        ValidationError::DuplicateCase("case".to_string())
    );

    let checksum_failure = EvidenceLedger::new("bad")
        .add_source(SourcePin::new("s", "https://example.invalid", "abc123"))
        .add_case(CaseEvidence::timed("case", "s", 1.0, 2.0).with_checksum(ChecksumStatus::Failed));
    assert_eq!(
        checksum_failure.validate().unwrap_err(),
        ValidationError::ChecksumFailed("case".to_string())
    );
}

#[test]
fn rejects_invalid_timings_and_empty_unsupported_reasons() {
    let invalid_timing = EvidenceLedger::new("bad")
        .add_source(SourcePin::new("s", "https://example.invalid", "abc123"))
        .add_case(CaseEvidence::timed("case", "s", 0.0, 2.0));
    assert_eq!(
        invalid_timing.validate().unwrap_err(),
        ValidationError::InvalidTiming("case".to_string())
    );

    let empty_reason = EvidenceLedger::new("bad")
        .add_source(SourcePin::new("s", "https://example.invalid", "abc123"))
        .add_case(CaseEvidence::timed("case", "s", 1.0, 2.0))
        .add_unsupported(UnsupportedCase::new("missing", ""));
    assert_eq!(
        empty_reason.validate().unwrap_err(),
        ValidationError::EmptyUnsupportedReason("missing".to_string())
    );
}
