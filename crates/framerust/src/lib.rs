use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum FrameError {
    EmptyFrame,
    DuplicateColumn(String),
    LengthMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },
    MissingColumn(String),
    UnsupportedKeyColumn(String),
    NonNumericColumn(String),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "frame requires at least one column"),
            Self::DuplicateColumn(name) => write!(f, "duplicate column: {name}"),
            Self::LengthMismatch {
                column,
                expected,
                actual,
            } => write!(
                f,
                "column {column} has length {actual}, expected {expected}"
            ),
            Self::MissingColumn(name) => write!(f, "missing column: {name}"),
            Self::UnsupportedKeyColumn(name) => {
                write!(f, "column {name} cannot be used as a group key")
            }
            Self::NonNumericColumn(name) => write!(f, "column {name} is not numeric"),
        }
    }
}

impl Error for FrameError {}

pub type Result<T> = std::result::Result<T, FrameError>;

#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    Text(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Text(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            Self::I64(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&[String]> {
        match self {
            Self::Text(values) => Some(values),
            _ => None,
        }
    }

    fn key_at(&self, row: usize, name: &str) -> Result<Key> {
        match self {
            Self::I64(values) => Ok(Key::I64(values[row])),
            Self::Bool(values) => Ok(Key::Bool(values[row])),
            Self::Text(values) => Ok(Key::Text(values[row].clone())),
            Self::F64(_) => Err(FrameError::UnsupportedKeyColumn(name.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    I64(i64),
    Bool(bool),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationKind {
    Count,
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aggregation {
    pub column: Option<String>,
    pub kind: AggregationKind,
    pub output: String,
}

impl Aggregation {
    pub fn count(output: impl Into<String>) -> Self {
        Self {
            column: None,
            kind: AggregationKind::Count,
            output: output.into(),
        }
    }

    pub fn sum(column: impl Into<String>, output: impl Into<String>) -> Self {
        Self::numeric(column, AggregationKind::Sum, output)
    }

    pub fn mean(column: impl Into<String>, output: impl Into<String>) -> Self {
        Self::numeric(column, AggregationKind::Mean, output)
    }

    pub fn min(column: impl Into<String>, output: impl Into<String>) -> Self {
        Self::numeric(column, AggregationKind::Min, output)
    }

    pub fn max(column: impl Into<String>, output: impl Into<String>) -> Self {
        Self::numeric(column, AggregationKind::Max, output)
    }

    fn numeric(
        column: impl Into<String>,
        kind: AggregationKind,
        output: impl Into<String>,
    ) -> Self {
        Self {
            column: Some(column.into()),
            kind,
            output: output.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    columns: BTreeMap<String, Column>,
    nrows: usize,
}

impl Frame {
    pub fn from_columns<I, S>(columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Column)>,
        S: Into<String>,
    {
        let mut out = BTreeMap::new();
        let mut nrows = None;
        for (name, column) in columns {
            let name = name.into();
            if out.contains_key(&name) {
                return Err(FrameError::DuplicateColumn(name));
            }
            match nrows {
                Some(expected) if column.len() != expected => {
                    return Err(FrameError::LengthMismatch {
                        column: name,
                        expected,
                        actual: column.len(),
                    });
                }
                None => nrows = Some(column.len()),
                _ => {}
            }
            out.insert(name, column);
        }

        let nrows = nrows.ok_or(FrameError::EmptyFrame)?;
        Ok(Self {
            columns: out,
            nrows,
        })
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .get(name)
            .ok_or_else(|| FrameError::MissingColumn(name.to_string()))
    }

    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.keys().map(String::as_str)
    }

    pub fn groupby(&self, key: impl Into<String>) -> GroupBy<'_> {
        GroupBy {
            frame: self,
            key: key.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AggState {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl AggState {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn count_row(&mut self) {
        self.count += 1;
    }

    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn finish(self, kind: AggregationKind) -> f64 {
        match kind {
            AggregationKind::Count => self.count as f64,
            AggregationKind::Sum => self.sum,
            AggregationKind::Mean => self.sum / self.count as f64,
            AggregationKind::Min => self.min,
            AggregationKind::Max => self.max,
        }
    }
}

pub struct GroupBy<'a> {
    frame: &'a Frame,
    key: String,
}

#[derive(Debug, Clone, Copy)]
enum NumericColumn<'a> {
    F64(&'a [f64]),
    I64(&'a [i64]),
}

impl NumericColumn<'_> {
    fn get(self, row: usize) -> f64 {
        match self {
            Self::F64(values) => values[row],
            Self::I64(values) => values[row] as f64,
        }
    }
}

#[derive(Debug)]
struct ResolvedAggregation<'a> {
    kind: AggregationKind,
    output: String,
    numeric: Option<NumericColumn<'a>>,
}

impl GroupBy<'_> {
    pub fn agg(&self, aggs: &[Aggregation]) -> Result<Frame> {
        let key_column = self.frame.column(&self.key)?;
        let resolved = self.resolve_aggs(aggs)?;
        if let Column::I64(keys) = key_column {
            if let Some(frame) = self.agg_dense_i64(keys, &resolved)? {
                return Ok(frame);
            }
        }
        self.agg_generic(key_column, &resolved)
    }

    fn resolve_aggs<'a>(&'a self, aggs: &[Aggregation]) -> Result<Vec<ResolvedAggregation<'a>>> {
        aggs.iter()
            .map(|agg| {
                let numeric = match agg.kind {
                    AggregationKind::Count => None,
                    _ => {
                        let name = agg.column.as_ref().expect("numeric aggregation column");
                        let column = self.frame.column(name)?;
                        Some(match column {
                            Column::F64(values) => NumericColumn::F64(values),
                            Column::I64(values) => NumericColumn::I64(values),
                            Column::Bool(_) | Column::Text(_) => {
                                return Err(FrameError::NonNumericColumn(name.clone()));
                            }
                        })
                    }
                };
                Ok(ResolvedAggregation {
                    kind: agg.kind,
                    output: agg.output.clone(),
                    numeric,
                })
            })
            .collect()
    }

    fn agg_dense_i64(
        &self,
        key_values: &[i64],
        aggs: &[ResolvedAggregation<'_>],
    ) -> Result<Option<Frame>> {
        if key_values.is_empty() {
            return self
                .finish_i64_keys(Vec::new(), vec![Vec::new(); aggs.len()], aggs)
                .map(Some);
        }
        let min = *key_values.iter().min().expect("non-empty keys");
        let max = *key_values.iter().max().expect("non-empty keys");
        let range = (max as i128) - (min as i128) + 1;
        if range <= 0 || range as usize > self.frame.nrows.max(1) * 4 {
            return Ok(None);
        }

        let mut group_index = vec![usize::MAX; range as usize];
        let mut keys = Vec::new();
        let mut states: Vec<Vec<AggState>> = vec![Vec::new(); aggs.len()];

        for (row, key) in key_values.iter().enumerate() {
            let offset = (*key - min) as usize;
            let idx = if group_index[offset] == usize::MAX {
                let idx = keys.len();
                group_index[offset] = idx;
                keys.push(*key);
                for state_column in &mut states {
                    state_column.push(AggState::new());
                }
                idx
            } else {
                group_index[offset]
            };

            update_states(row, idx, aggs, &mut states);
        }

        self.finish_i64_keys(keys, states, aggs).map(Some)
    }

    fn agg_generic(&self, key_column: &Column, aggs: &[ResolvedAggregation<'_>]) -> Result<Frame> {
        let mut group_index = HashMap::new();
        let mut keys = Vec::new();
        let mut states: Vec<Vec<AggState>> = vec![Vec::new(); aggs.len()];

        for row in 0..self.frame.nrows {
            let key = key_column.key_at(row, &self.key)?;
            let idx = match group_index.entry(key.clone()) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let idx = keys.len();
                    keys.push(key);
                    for state_column in &mut states {
                        state_column.push(AggState::new());
                    }
                    entry.insert(idx);
                    idx
                }
            };

            update_states(row, idx, aggs, &mut states);
        }

        self.finish_keys(keys, states, aggs)
    }

    fn finish_i64_keys(
        &self,
        keys: Vec<i64>,
        states: Vec<Vec<AggState>>,
        aggs: &[ResolvedAggregation<'_>],
    ) -> Result<Frame> {
        self.finish_columns(Column::I64(keys), states, aggs)
    }

    fn finish_keys(
        &self,
        keys: Vec<Key>,
        states: Vec<Vec<AggState>>,
        aggs: &[ResolvedAggregation<'_>],
    ) -> Result<Frame> {
        self.finish_columns(keys_to_column(keys), states, aggs)
    }

    fn finish_columns(
        &self,
        key_column: Column,
        states: Vec<Vec<AggState>>,
        aggs: &[ResolvedAggregation<'_>],
    ) -> Result<Frame> {
        let mut columns = vec![(self.key.clone(), key_column)];
        for (agg_idx, agg) in aggs.iter().enumerate() {
            let values = states[agg_idx]
                .iter()
                .map(|state| state.finish(agg.kind))
                .collect::<Vec<_>>();
            let column = match agg.kind {
                AggregationKind::Count => {
                    Column::I64(values.into_iter().map(|v| v as i64).collect())
                }
                _ => Column::F64(values),
            };
            columns.push((agg.output.clone(), column));
        }
        Frame::from_columns(columns)
    }
}

fn update_states(
    row: usize,
    group_idx: usize,
    aggs: &[ResolvedAggregation<'_>],
    states: &mut [Vec<AggState>],
) {
    for (agg_idx, agg) in aggs.iter().enumerate() {
        match agg.kind {
            AggregationKind::Count => states[agg_idx][group_idx].count_row(),
            _ => {
                let value = agg.numeric.expect("resolved numeric aggregation").get(row);
                states[agg_idx][group_idx].push(value);
            }
        }
    }
}

fn keys_to_column(keys: Vec<Key>) -> Column {
    match keys.first() {
        Some(Key::I64(_)) | None => Column::I64(
            keys.into_iter()
                .map(|key| match key {
                    Key::I64(value) => value,
                    _ => unreachable!("group keys come from one source column"),
                })
                .collect(),
        ),
        Some(Key::Bool(_)) => Column::Bool(
            keys.into_iter()
                .map(|key| match key {
                    Key::Bool(value) => value,
                    _ => unreachable!("group keys come from one source column"),
                })
                .collect(),
        ),
        Some(Key::Text(_)) => Column::Text(
            keys.into_iter()
                .map(|key| match key {
                    Key::Text(value) => value,
                    _ => unreachable!("group keys come from one source column"),
                })
                .collect(),
        ),
    }
}

pub fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

pub fn bench_median_ms<F>(rounds: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    median_duration(samples).as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groupby_i64_sum_mean_min_max_and_count() {
        let frame = Frame::from_columns([
            ("store", Column::I64(vec![2, 1, 2, 1, 2])),
            ("sales", Column::F64(vec![10.0, 4.0, 7.0, 6.0, 3.0])),
        ])
        .unwrap();
        let grouped = frame
            .groupby("store")
            .agg(&[
                Aggregation::count("rows"),
                Aggregation::sum("sales", "sales_sum"),
                Aggregation::mean("sales", "sales_mean"),
                Aggregation::min("sales", "sales_min"),
                Aggregation::max("sales", "sales_max"),
            ])
            .unwrap();

        assert_eq!(grouped.column("store").unwrap().as_i64().unwrap(), &[2, 1]);
        assert_eq!(grouped.column("rows").unwrap().as_i64().unwrap(), &[3, 2]);
        assert_eq!(
            grouped.column("sales_sum").unwrap().as_f64().unwrap(),
            &[20.0, 10.0]
        );
        assert_eq!(
            grouped.column("sales_mean").unwrap().as_f64().unwrap(),
            &[20.0 / 3.0, 5.0]
        );
        assert_eq!(
            grouped.column("sales_min").unwrap().as_f64().unwrap(),
            &[3.0, 4.0]
        );
        assert_eq!(
            grouped.column("sales_max").unwrap().as_f64().unwrap(),
            &[10.0, 6.0]
        );
    }

    #[test]
    fn groupby_text_preserves_first_seen_order() {
        let frame = Frame::from_columns([
            (
                "region",
                Column::Text(vec![
                    "west".to_string(),
                    "east".to_string(),
                    "west".to_string(),
                ]),
            ),
            ("sales", Column::I64(vec![5, 9, 7])),
        ])
        .unwrap();
        let grouped = frame
            .groupby("region")
            .agg(&[Aggregation::sum("sales", "sales_sum")])
            .unwrap();

        assert_eq!(
            grouped.column("region").unwrap().as_text().unwrap(),
            &["west".to_string(), "east".to_string()]
        );
        assert_eq!(
            grouped.column("sales_sum").unwrap().as_f64().unwrap(),
            &[12.0, 9.0]
        );
    }

    #[test]
    fn rejects_mismatched_column_lengths() {
        let err = Frame::from_columns([
            ("a", Column::I64(vec![1, 2])),
            ("b", Column::F64(vec![1.0])),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            FrameError::LengthMismatch {
                column: "b".to_string(),
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn rejects_non_numeric_aggregation_columns() {
        let frame = Frame::from_columns([
            ("key", Column::I64(vec![1, 1])),
            ("name", Column::Text(vec!["a".to_string(), "b".to_string()])),
        ])
        .unwrap();
        let err = frame
            .groupby("key")
            .agg(&[Aggregation::sum("name", "name_sum")])
            .unwrap_err();
        assert_eq!(err, FrameError::NonNumericColumn("name".to_string()));
    }
}
