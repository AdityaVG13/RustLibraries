//! `rigortrail` records benchmark and evaluation claims as auditable evidence.
//!
//! It is not a benchmark runner. It is the layer after measurement: source
//! pins, supported cases, unsupported cases, checksum status, score summaries,
//! and a hard gate that prevents broad claims when evidence is incomplete.

use std::collections::BTreeSet;
use std::fmt::{self, Write};

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceLedger {
    claim: String,
    sources: Vec<SourcePin>,
    cases: Vec<CaseEvidence>,
    unsupported: Vec<UnsupportedCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePin {
    pub id: String,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseEvidence {
    pub name: String,
    pub source_id: String,
    pub candidate_ms: f64,
    pub baseline_ms: f64,
    pub checksum: ChecksumStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedCase {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumStatus {
    Passed,
    Failed,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Winner {
    Candidate,
    Baseline,
    Tie,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub supported_cases: usize,
    pub unsupported_cases: usize,
    pub candidate_wins: usize,
    pub baseline_wins: usize,
    pub ties: usize,
    pub geomean_speedup: f64,
    pub ranked_higher_by_wins: bool,
    pub global_claim_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    EmptyClaim,
    EmptySourceId,
    DuplicateSource(String),
    DuplicateCase(String),
    MissingCaseSource { case: String, source_id: String },
    InvalidTiming(String),
    ChecksumFailed(String),
    EmptyUnsupportedReason(String),
}

impl EvidenceLedger {
    pub fn new(claim: impl Into<String>) -> Self {
        Self {
            claim: claim.into(),
            sources: Vec::new(),
            cases: Vec::new(),
            unsupported: Vec::new(),
        }
    }

    pub fn claim(&self) -> &str {
        &self.claim
    }

    pub fn sources(&self) -> &[SourcePin] {
        &self.sources
    }

    pub fn cases(&self) -> &[CaseEvidence] {
        &self.cases
    }

    pub fn unsupported_cases(&self) -> &[UnsupportedCase] {
        &self.unsupported
    }

    pub fn add_source(mut self, source: SourcePin) -> Self {
        self.sources.push(source);
        self
    }

    pub fn add_case(mut self, case: CaseEvidence) -> Self {
        self.cases.push(case);
        self
    }

    pub fn add_unsupported(mut self, case: UnsupportedCase) -> Self {
        self.unsupported.push(case);
        self
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.claim.trim().is_empty() {
            return Err(ValidationError::EmptyClaim);
        }

        let mut source_ids = BTreeSet::new();
        for source in &self.sources {
            if source.id.trim().is_empty() {
                return Err(ValidationError::EmptySourceId);
            }
            if !source_ids.insert(source.id.as_str()) {
                return Err(ValidationError::DuplicateSource(source.id.clone()));
            }
        }

        let mut case_names = BTreeSet::new();
        for case in &self.cases {
            if !case_names.insert(case.name.as_str()) {
                return Err(ValidationError::DuplicateCase(case.name.clone()));
            }
            if !source_ids.contains(case.source_id.as_str()) {
                return Err(ValidationError::MissingCaseSource {
                    case: case.name.clone(),
                    source_id: case.source_id.clone(),
                });
            }
            if !case.candidate_ms.is_finite()
                || !case.baseline_ms.is_finite()
                || case.candidate_ms <= 0.0
                || case.baseline_ms <= 0.0
            {
                return Err(ValidationError::InvalidTiming(case.name.clone()));
            }
            if case.checksum == ChecksumStatus::Failed {
                return Err(ValidationError::ChecksumFailed(case.name.clone()));
            }
        }

        for case in &self.unsupported {
            if case.reason.trim().is_empty() {
                return Err(ValidationError::EmptyUnsupportedReason(case.name.clone()));
            }
        }

        Ok(())
    }

    pub fn score(&self) -> Result<Score, ValidationError> {
        self.validate()?;

        let mut candidate_wins = 0usize;
        let mut baseline_wins = 0usize;
        let mut ties = 0usize;
        let mut log_speedup_sum = 0.0;

        for case in &self.cases {
            match case.winner() {
                Winner::Candidate => candidate_wins += 1,
                Winner::Baseline => baseline_wins += 1,
                Winner::Tie => ties += 1,
            }
            log_speedup_sum += (case.baseline_ms / case.candidate_ms).ln();
        }

        let geomean_speedup = if self.cases.is_empty() {
            1.0
        } else {
            (log_speedup_sum / self.cases.len() as f64).exp()
        };
        let ranked_higher_by_wins = candidate_wins > baseline_wins;
        let global_claim_allowed = ranked_higher_by_wins && self.unsupported.is_empty();

        Ok(Score {
            supported_cases: self.cases.len(),
            unsupported_cases: self.unsupported.len(),
            candidate_wins,
            baseline_wins,
            ties,
            geomean_speedup,
            ranked_higher_by_wins,
            global_claim_allowed,
        })
    }

    pub fn render_markdown(&self) -> Result<String, ValidationError> {
        let score = self.score()?;
        let mut out = String::new();
        writeln!(&mut out, "# Evidence Ledger").expect("write to string cannot fail");
        writeln!(&mut out).expect("write to string cannot fail");
        writeln!(&mut out, "Claim: `{}`", self.claim).expect("write to string cannot fail");
        writeln!(&mut out).expect("write to string cannot fail");
        writeln!(&mut out, "## Score").expect("write to string cannot fail");
        writeln!(&mut out).expect("write to string cannot fail");
        writeln!(&mut out, "- Supported cases: {}", score.supported_cases)
            .expect("write to string cannot fail");
        writeln!(&mut out, "- Unsupported cases: {}", score.unsupported_cases)
            .expect("write to string cannot fail");
        writeln!(&mut out, "- Candidate wins: {}", score.candidate_wins)
            .expect("write to string cannot fail");
        writeln!(&mut out, "- Baseline wins: {}", score.baseline_wins)
            .expect("write to string cannot fail");
        writeln!(&mut out, "- Geomean speedup: {:.2}x", score.geomean_speedup)
            .expect("write to string cannot fail");
        writeln!(
            &mut out,
            "- Global claim allowed: {}",
            score.global_claim_allowed
        )
        .expect("write to string cannot fail");

        writeln!(&mut out).expect("write to string cannot fail");
        writeln!(&mut out, "## Cases").expect("write to string cannot fail");
        writeln!(&mut out).expect("write to string cannot fail");
        writeln!(
            &mut out,
            "| Case | Source | Candidate ms | Baseline ms | Speedup | Winner |"
        )
        .expect("write to string cannot fail");
        writeln!(&mut out, "| --- | --- | ---: | ---: | ---: | --- |")
            .expect("write to string cannot fail");
        for case in &self.cases {
            writeln!(
                &mut out,
                "| `{}` | `{}` | {:.3} | {:.3} | {:.2}x | {:?} |",
                case.name,
                case.source_id,
                case.candidate_ms,
                case.baseline_ms,
                case.speedup(),
                case.winner()
            )
            .expect("write to string cannot fail");
        }

        if !self.unsupported.is_empty() {
            writeln!(&mut out).expect("write to string cannot fail");
            writeln!(&mut out, "## Unsupported Cases").expect("write to string cannot fail");
            writeln!(&mut out).expect("write to string cannot fail");
            writeln!(&mut out, "| Case | Reason |").expect("write to string cannot fail");
            writeln!(&mut out, "| --- | --- |").expect("write to string cannot fail");
            for case in &self.unsupported {
                writeln!(&mut out, "| `{}` | {} |", case.name, case.reason)
                    .expect("write to string cannot fail");
            }
        }

        Ok(out)
    }
}

impl SourcePin {
    pub fn new(id: impl Into<String>, url: impl Into<String>, sha256: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            sha256: sha256.into(),
        }
    }
}

impl CaseEvidence {
    pub fn timed(
        name: impl Into<String>,
        source_id: impl Into<String>,
        candidate_ms: f64,
        baseline_ms: f64,
    ) -> Self {
        Self {
            name: name.into(),
            source_id: source_id.into(),
            candidate_ms,
            baseline_ms,
            checksum: ChecksumStatus::Passed,
        }
    }

    pub fn with_checksum(mut self, checksum: ChecksumStatus) -> Self {
        self.checksum = checksum;
        self
    }

    pub fn speedup(&self) -> f64 {
        self.baseline_ms / self.candidate_ms
    }

    pub fn winner(&self) -> Winner {
        let speedup = self.speedup();
        if (speedup - 1.0).abs() <= f64::EPSILON {
            Winner::Tie
        } else if speedup > 1.0 {
            Winner::Candidate
        } else {
            Winner::Baseline
        }
    }
}

impl UnsupportedCase {
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyClaim => write!(f, "claim cannot be empty"),
            ValidationError::EmptySourceId => write!(f, "source id cannot be empty"),
            ValidationError::DuplicateSource(id) => write!(f, "duplicate source id `{id}`"),
            ValidationError::DuplicateCase(name) => write!(f, "duplicate case `{name}`"),
            ValidationError::MissingCaseSource { case, source_id } => {
                write!(f, "case `{case}` references missing source `{source_id}`")
            }
            ValidationError::InvalidTiming(name) => {
                write!(f, "case `{name}` has invalid timing")
            }
            ValidationError::ChecksumFailed(name) => {
                write!(f, "case `{name}` has a checksum failure")
            }
            ValidationError::EmptyUnsupportedReason(name) => {
                write!(f, "unsupported case `{name}` needs a reason")
            }
        }
    }
}

impl std::error::Error for ValidationError {}
