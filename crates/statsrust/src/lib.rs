//! `statsrust` is the StatsModels-inspired FromPythonToRust library slice.
//!
//! It starts a StatsModels-style Rust crate with ordinary least squares,
//! logistic regression, model diagnostics, and prediction over the
//! `numrs-core` array foundation.

mod logit;
mod ols;

pub use logit::{BinaryClassificationMetrics, LogitFit, LogitModel, LogitOptions};
pub use ols::{OlsFit, OlsModel, RegressionMetrics, WlsFit};
