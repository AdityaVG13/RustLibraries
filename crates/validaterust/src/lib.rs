use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    List(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn object(fields: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Self {
        Self::Object(
            fields
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
        )
    }

    pub fn list(values: impl IntoIterator<Item = Value>) -> Self {
        Self::List(values.into_iter().collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    Bool,
    Int,
    Float,
    Text,
    List(Box<ValueType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub value_type: ValueType,
    pub required: bool,
    pub min_len: Option<usize>,
    pub min_number: Option<i64>,
    pub max_number: Option<i64>,
}

impl Field {
    pub fn required(name: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            name: name.into(),
            value_type,
            required: true,
            min_len: None,
            min_number: None,
            max_number: None,
        }
    }

    pub fn optional(name: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            required: false,
            ..Self::required(name, value_type)
        }
    }

    pub fn min_len(mut self, min_len: usize) -> Self {
        self.min_len = Some(min_len);
        self
    }

    pub fn number_range(mut self, min: i64, max: i64) -> Self {
        self.min_number = Some(min);
        self.max_number = Some(max);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: impl IntoIterator<Item = Field>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    pub fn validate<'a>(&self, value: &'a Value) -> Result<ValidatedObject<'a>, ValidationError> {
        let object = match value {
            Value::Object(object) => object,
            _ => return Err(ValidationError::RootNotObject),
        };
        for field in &self.fields {
            let Some(field_value) = object.get(&field.name) else {
                if field.required {
                    return Err(ValidationError::MissingField(field.name.clone()));
                }
                continue;
            };
            if matches!(field_value, Value::Null) && !field.required {
                continue;
            }
            validate_type(&field.value_type, field_value, &field.name)?;
            validate_constraints(field, field_value)?;
        }
        Ok(ValidatedObject { object })
    }

    pub fn validate_many<'a>(
        &self,
        values: &'a [Value],
    ) -> Result<Vec<ValidatedObject<'a>>, ValidationError> {
        values.iter().map(|value| self.validate(value)).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ValidatedObject<'a> {
    object: &'a BTreeMap<String, Value>,
}

impl ValidatedObject<'_> {
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.object.get(name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    RootNotObject,
    MissingField(String),
    TypeMismatch {
        field: String,
        expected: ValueType,
        actual: &'static str,
    },
    MinLength {
        field: String,
        min: usize,
        actual: usize,
    },
    NumberTooSmall {
        field: String,
        min: i64,
        actual: i64,
    },
    NumberTooLarge {
        field: String,
        max: i64,
        actual: i64,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootNotObject => write!(f, "root value must be an object"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::TypeMismatch {
                field,
                expected,
                actual,
            } => write!(f, "field {field} expected {expected:?}, received {actual}"),
            Self::MinLength { field, min, actual } => {
                write!(f, "field {field} length {actual} is below minimum {min}")
            }
            Self::NumberTooSmall { field, min, actual } => {
                write!(f, "field {field} value {actual} is below minimum {min}")
            }
            Self::NumberTooLarge { field, max, actual } => {
                write!(f, "field {field} value {actual} is above maximum {max}")
            }
        }
    }
}

impl Error for ValidationError {}

fn validate_type(expected: &ValueType, value: &Value, field: &str) -> Result<(), ValidationError> {
    match (expected, value) {
        (ValueType::Bool, Value::Bool(_))
        | (ValueType::Int, Value::Int(_))
        | (ValueType::Float, Value::Float(_))
        | (ValueType::Float, Value::Int(_))
        | (ValueType::Text, Value::Text(_)) => Ok(()),
        (ValueType::List(inner), Value::List(values)) => {
            for value in values {
                validate_type(inner, value, field)?;
            }
            Ok(())
        }
        _ => Err(ValidationError::TypeMismatch {
            field: field.to_string(),
            expected: expected.clone(),
            actual: type_name(value),
        }),
    }
}

fn validate_constraints(field: &Field, value: &Value) -> Result<(), ValidationError> {
    if let (Some(min), Value::Text(text)) = (field.min_len, value) {
        if text.len() < min {
            return Err(ValidationError::MinLength {
                field: field.name.clone(),
                min,
                actual: text.len(),
            });
        }
    }
    if let Value::Int(value) = value {
        if let Some(min) = field.min_number {
            if *value < min {
                return Err(ValidationError::NumberTooSmall {
                    field: field.name.clone(),
                    min,
                    actual: *value,
                });
            }
        }
        if let Some(max) = field.max_number {
            if *value > max {
                return Err(ValidationError::NumberTooLarge {
                    field: field.name.clone(),
                    max,
                    actual: *value,
                });
            }
        }
    }
    Ok(())
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Text(_) => "text",
        Value::List(_) => "list",
        Value::Object(_) => "object",
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

    fn user_schema() -> Schema {
        Schema::new([
            Field::required("id", ValueType::Int).number_range(0, 1_000_000),
            Field::required("email", ValueType::Text).min_len(5),
            Field::required("score", ValueType::Float),
            Field::optional("active", ValueType::Bool),
            Field::required("tags", ValueType::List(Box::new(ValueType::Text))),
        ])
    }

    #[test]
    fn validates_object_schema() {
        let value = Value::object([
            ("id", Value::Int(7)),
            ("email", Value::Text("user@example.test".to_string())),
            ("score", Value::Float(91.5)),
            ("active", Value::Bool(true)),
            (
                "tags",
                Value::list([
                    Value::Text("alpha".to_string()),
                    Value::Text("beta".to_string()),
                ]),
            ),
        ]);
        let validated = user_schema().validate(&value).unwrap();
        assert_eq!(validated.get("id"), Some(&Value::Int(7)));
    }

    #[test]
    fn reports_missing_fields() {
        let value = Value::object([("id", Value::Int(7))]);
        assert_eq!(
            user_schema().validate(&value).unwrap_err(),
            ValidationError::MissingField("email".to_string())
        );
    }

    #[test]
    fn reports_nested_list_type_mismatch() {
        let value = Value::object([
            ("id", Value::Int(7)),
            ("email", Value::Text("user@example.test".to_string())),
            ("score", Value::Float(91.5)),
            (
                "tags",
                Value::list([Value::Text("ok".to_string()), Value::Int(3)]),
            ),
        ]);
        assert_eq!(
            user_schema().validate(&value).unwrap_err(),
            ValidationError::TypeMismatch {
                field: "tags".to_string(),
                expected: ValueType::Text,
                actual: "int",
            }
        );
    }

    #[test]
    fn reports_string_constraints() {
        let value = Value::object([
            ("id", Value::Int(7)),
            ("email", Value::Text("x".to_string())),
            ("score", Value::Float(91.5)),
            ("tags", Value::list([Value::Text("ok".to_string())])),
        ]);
        assert_eq!(
            user_schema().validate(&value).unwrap_err(),
            ValidationError::MinLength {
                field: "email".to_string(),
                min: 5,
                actual: 1,
            }
        );
    }
}
