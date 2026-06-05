//! Output schema validation — a lightweight JSON Schema implementation.
//!
//! This module provides a simplified schema validation system inspired by
//! JSON Schema. It supports type checking, required fields, and basic
//! constraints — enough to validate that action outputs conform to the
//! shape declared in an [`IntentContract`](super::contract::IntentContract).
//!
//! # Supported Checks
//!
//! - **Type validation**: string, number, integer, boolean, array, object, null
//! - **Required fields**: ensure specific keys are present in objects
//! - **Minimum/maximum**: numeric range constraints
//! - **Min/max length**: string length constraints
//! - **Min/max items**: array size constraints
//! - **Nested objects**: recursive validation of object properties
//!
//! # Example
//!
//! ```
//! use pincher_core::intent::schema::{OutputSchema, SchemaField, SchemaValidator};
//!
//! let schema = OutputSchema::object()
//!     .with_required("status", SchemaField::string().with_min_length(1))
//!     .with_required("code", SchemaField::integer().with_minimum(0.0))
//!     .with_optional("message", SchemaField::string());
//!
//! let output = serde_json::json!({
//!     "status": "ok",
//!     "code": 200,
//!     "message": "success"
//! });
//!
//! let validator = SchemaValidator::new();
//! assert!(validator.validate(&output, &schema).is_ok());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, instrument, warn};

/// Schema validation errors.
#[derive(Debug, Error)]
pub enum SchemaValidationError {
    /// A required field is missing from the object.
    #[error("Missing required field: {field}")]
    MissingRequiredField {
        /// The name of the missing field.
        field: String,
    },

    /// A field has the wrong type.
    #[error("Type mismatch for field '{field}': expected {expected}, got {actual}")]
    TypeMismatch {
        /// The field path.
        field: String,
        /// The expected type name.
        expected: String,
        /// The actual type name.
        actual: String,
    },

    /// A numeric value is below the minimum.
    #[error("Value for field '{field}' is below minimum: {value} < {minimum}")]
    BelowMinimum {
        /// The field path.
        field: String,
        /// The actual value.
        value: f64,
        /// The minimum allowed value.
        minimum: f64,
    },

    /// A numeric value exceeds the maximum.
    #[error("Value for field '{field}' exceeds maximum: {value} > {maximum}")]
    AboveMaximum {
        /// The field path.
        field: String,
        /// The actual value.
        value: f64,
        /// The maximum allowed value.
        maximum: f64,
    },

    /// A string is shorter than the minimum length.
    #[error("String for field '{field}' is too short: length {length} < {min_length}")]
    StringTooShort {
        /// The field path.
        field: String,
        /// The actual length.
        length: usize,
        /// The minimum allowed length.
        min_length: usize,
    },

    /// A string exceeds the maximum length.
    #[error("String for field '{field}' is too long: length {length} > {max_length}")]
    StringTooLong {
        /// The field path.
        field: String,
        /// The actual length.
        length: usize,
        /// The maximum allowed length.
        max_length: usize,
    },

    /// An array has fewer items than the minimum.
    #[error("Array for field '{field}' has too few items: {count} < {min_items}")]
    ArrayTooShort {
        /// The field path.
        field: String,
        /// The actual item count.
        count: usize,
        /// The minimum required items.
        min_items: usize,
    },

    /// An array has more items than the maximum.
    #[error("Array for field '{field}' has too many items: {count} > {max_items}")]
    ArrayTooLong {
        /// The field path.
        field: String,
        /// The actual item count.
        count: usize,
        /// The maximum allowed items.
        max_items: usize,
    },

    /// An array item failed validation.
    #[error("Array item {index} in field '{field}' failed validation: {reason}")]
    ArrayItemInvalid {
        /// The field path.
        field: String,
        /// The index of the invalid item.
        index: usize,
        /// The reason for failure.
        reason: String,
    },

    /// A nested object property failed validation.
    #[error("Nested validation error in field '{field}': {inner}")]
    Nested {
        /// The field path.
        field: String,
        /// The inner error.
        inner: Box<SchemaValidationError>,
    },

    /// The root value has the wrong type.
    #[error("Root type mismatch: expected {expected}, got {actual}")]
    RootTypeMismatch {
        /// The expected type name.
        expected: String,
        /// The actual type name.
        actual: String,
    },
}

/// Result type for schema validation operations.
pub type SchemaValidationResult<T> = Result<T, SchemaValidationError>;

/// The type of a schema field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// A JSON string.
    String,
    /// A JSON number (floating point).
    Number,
    /// A JSON integer (number without fractional part).
    Integer,
    /// A JSON boolean.
    Boolean,
    /// A JSON array.
    Array,
    /// A JSON object.
    Object,
    /// A JSON null value.
    Null,
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldType::String => write!(f, "string"),
            FieldType::Number => write!(f, "number"),
            FieldType::Integer => write!(f, "integer"),
            FieldType::Boolean => write!(f, "boolean"),
            FieldType::Array => write!(f, "array"),
            FieldType::Object => write!(f, "object"),
            FieldType::Null => write!(f, "null"),
        }
    }
}

/// A schema field definition describing the expected shape of a value.
///
/// This is a simplified JSON Schema field — it captures type, constraints,
/// and nested structure without the full complexity of JSON Schema draft-07.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    /// The expected type of this field.
    pub field_type: FieldType,

    /// Minimum value for number/integer fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,

    /// Maximum value for number/integer fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,

    /// Minimum string length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,

    /// Maximum string length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,

    /// Minimum number of array items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_items: Option<usize>,

    /// Maximum number of array items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_items: Option<usize>,

    /// Schema for array items (if this field is an array type).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<SchemaField>>,

    /// Properties of an object (if this field is an object type).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, SchemaField>>,

    /// Which properties are required (for object types).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl SchemaField {
    /// Create a string-type schema field.
    pub fn string() -> Self {
        Self {
            field_type: FieldType::String,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a number-type schema field.
    pub fn number() -> Self {
        Self {
            field_type: FieldType::Number,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an integer-type schema field.
    pub fn integer() -> Self {
        Self {
            field_type: FieldType::Integer,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a boolean-type schema field.
    pub fn boolean() -> Self {
        Self {
            field_type: FieldType::Boolean,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an array-type schema field with the given item schema.
    pub fn array(items: SchemaField) -> Self {
        Self {
            field_type: FieldType::Array,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: Some(Box::new(items)),
            properties: None,
            required: None,
        }
    }

    /// Create a null-type schema field.
    pub fn null() -> Self {
        Self {
            field_type: FieldType::Null,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Set the minimum value (for number/integer fields).
    pub fn with_minimum(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    /// Set the maximum value (for number/integer fields).
    pub fn with_maximum(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    /// Set the minimum string length.
    pub fn with_min_length(mut self, min: usize) -> Self {
        self.min_length = Some(min);
        self
    }

    /// Set the maximum string length.
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    /// Set the minimum number of array items.
    pub fn with_min_items(mut self, min: usize) -> Self {
        self.min_items = Some(min);
        self
    }

    /// Set the maximum number of array items.
    pub fn with_max_items(mut self, max: usize) -> Self {
        self.max_items = Some(max);
        self
    }
}

/// An output schema that describes the expected shape of an action's output.
///
/// This is essentially a named [`SchemaField`] of object type, with
/// convenience methods for constructing object schemas with required
/// and optional fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    /// The root schema field (always an object type).
    pub root: SchemaField,
}

impl OutputSchema {
    /// Create a new object-type output schema with no fields.
    pub fn object() -> Self {
        Self {
            root: SchemaField {
                field_type: FieldType::Object,
                minimum: None,
                maximum: None,
                min_length: None,
                max_length: None,
                min_items: None,
                max_items: None,
                items: None,
                properties: Some(HashMap::new()),
                required: Some(Vec::new()),
            },
        }
    }

    /// Add a required field to the object schema.
    pub fn with_required(mut self, name: &str, field: SchemaField) -> Self {
        if let Some(props) = &mut self.root.properties {
            props.insert(name.to_string(), field);
        } else {
            self.root.properties = Some({
                let mut map = HashMap::new();
                map.insert(name.to_string(), field);
                map
            });
        }
        if let Some(req) = &mut self.root.required {
            if !req.contains(&name.to_string()) {
                req.push(name.to_string());
            }
        } else {
            self.root.required = Some(vec![name.to_string()]);
        }
        self
    }

    /// Add an optional field to the object schema.
    pub fn with_optional(mut self, name: &str, field: SchemaField) -> Self {
        if let Some(props) = &mut self.root.properties {
            props.insert(name.to_string(), field);
        } else {
            self.root.properties = Some({
                let mut map = HashMap::new();
                map.insert(name.to_string(), field);
                map
            });
        }
        self
    }

    /// Create an output schema from a raw [`SchemaField`].
    ///
    /// This is useful when the root type is not an object (e.g., for
    /// actions that return plain strings or arrays).
    pub fn from_field(field: SchemaField) -> Self {
        Self { root: field }
    }
}

/// Validates action output against an [`OutputSchema`].
///
/// The validator walks the schema recursively and checks that the
/// output value conforms to the declared types and constraints.
pub struct SchemaValidator;

impl SchemaValidator {
    /// Create a new schema validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate a JSON value against an output schema.
    ///
    /// Returns `Ok(())` if the value conforms to the schema, or
    /// a [`SchemaValidationError`] describing the first violation.
    #[instrument(skip(self, value, schema))]
    pub fn validate(
        &self,
        value: &serde_json::Value,
        schema: &OutputSchema,
    ) -> SchemaValidationResult<()> {
        debug!("Validating output against schema");
        self.validate_field(value, &schema.root, "$")
    }

    /// Validate a JSON value against a single schema field.
    fn validate_field(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        // Check null against Null type
        if value.is_null() && field.field_type == FieldType::Null {
            return Ok(());
        }

        // Null values are allowed for optional fields (not in required list)
        if value.is_null() {
            // Null is acceptable only if the field is not a required Null type
            debug!(path = path, "Null value encountered at non-Null type field");
            return Ok(());
        }

        match field.field_type {
            FieldType::String => self.validate_string(value, field, path),
            FieldType::Number => self.validate_number(value, field, path),
            FieldType::Integer => self.validate_integer(value, field, path),
            FieldType::Boolean => self.validate_boolean(value, path),
            FieldType::Array => self.validate_array(value, field, path),
            FieldType::Object => self.validate_object(value, field, path),
            FieldType::Null => {
                // Non-null value where null was expected
                Err(SchemaValidationError::TypeMismatch {
                    field: path.to_string(),
                    expected: "null".to_string(),
                    actual: json_type_name(value),
                })
            }
        }
    }

    /// Validate a string value.
    fn validate_string(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        let s = value
            .as_str()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "string".to_string(),
                actual: json_type_name(value),
            })?;

        let len = s.len();

        if let Some(min) = field.min_length {
            if len < min {
                return Err(SchemaValidationError::StringTooShort {
                    field: path.to_string(),
                    length: len,
                    min_length: min,
                });
            }
        }

        if let Some(max) = field.max_length {
            if len > max {
                return Err(SchemaValidationError::StringTooLong {
                    field: path.to_string(),
                    length: len,
                    max_length: max,
                });
            }
        }

        Ok(())
    }

    /// Validate a number value.
    fn validate_number(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        let n = value
            .as_f64()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "number".to_string(),
                actual: json_type_name(value),
            })?;

        self.check_numeric_range(n, field, path)
    }

    /// Validate an integer value.
    fn validate_integer(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        // Accept both integers and floats that happen to be whole numbers
        let n = value
            .as_f64()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "integer".to_string(),
                actual: json_type_name(value),
            })?;

        if n.fract() != 0.0 {
            warn!(
                path = path,
                value = n,
                "Expected integer but got fractional number"
            );
            return Err(SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "integer".to_string(),
                actual: "number (fractional)".to_string(),
            });
        }

        self.check_numeric_range(n, field, path)
    }

    /// Check numeric range constraints.
    fn check_numeric_range(
        &self,
        value: f64,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        if let Some(min) = field.minimum {
            if value < min {
                return Err(SchemaValidationError::BelowMinimum {
                    field: path.to_string(),
                    value,
                    minimum: min,
                });
            }
        }

        if let Some(max) = field.maximum {
            if value > max {
                return Err(SchemaValidationError::AboveMaximum {
                    field: path.to_string(),
                    value,
                    maximum: max,
                });
            }
        }

        Ok(())
    }

    /// Validate a boolean value.
    fn validate_boolean(
        &self,
        value: &serde_json::Value,
        path: &str,
    ) -> SchemaValidationResult<()> {
        if !value.is_boolean() {
            return Err(SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "boolean".to_string(),
                actual: json_type_name(value),
            });
        }
        Ok(())
    }

    /// Validate an array value.
    fn validate_array(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        let arr = value
            .as_array()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "array".to_string(),
                actual: json_type_name(value),
            })?;

        let count = arr.len();

        if let Some(min) = field.min_items {
            if count < min {
                return Err(SchemaValidationError::ArrayTooShort {
                    field: path.to_string(),
                    count,
                    min_items: min,
                });
            }
        }

        if let Some(max) = field.max_items {
            if count > max {
                return Err(SchemaValidationError::ArrayTooLong {
                    field: path.to_string(),
                    count,
                    max_items: max,
                });
            }
        }

        // Validate each array item against the items schema
        if let Some(items_schema) = &field.items {
            for (i, item) in arr.iter().enumerate() {
                let item_path = format!("{}[{}]", path, i);
                if let Err(e) = self.validate_field(item, items_schema, &item_path) {
                    return Err(SchemaValidationError::ArrayItemInvalid {
                        field: path.to_string(),
                        index: i,
                        reason: e.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate an object value.
    fn validate_object(
        &self,
        value: &serde_json::Value,
        field: &SchemaField,
        path: &str,
    ) -> SchemaValidationResult<()> {
        let obj = value
            .as_object()
            .ok_or_else(|| SchemaValidationError::TypeMismatch {
                field: path.to_string(),
                expected: "object".to_string(),
                actual: json_type_name(value),
            })?;

        // Check required fields
        if let Some(required) = &field.required {
            for req_name in required {
                if !obj.contains_key(req_name) {
                    return Err(SchemaValidationError::MissingRequiredField {
                        field: format!("{}.{}", path, req_name),
                    });
                }
            }
        }

        // Validate each property that has a schema definition
        if let Some(properties) = &field.properties {
            for (prop_name, prop_schema) in properties {
                if let Some(prop_value) = obj.get(prop_name) {
                    let prop_path = format!("{}.{}", path, prop_name);
                    if let Err(e) = self.validate_field(prop_value, prop_schema, &prop_path) {
                        return Err(SchemaValidationError::Nested {
                            field: prop_path,
                            inner: Box::new(e),
                        });
                    }
                }
                // If the property is not present and not required, that's fine
                // (required fields are already checked above)
            }
        }

        Ok(())
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine the human-readable type name of a JSON value.
fn json_type_name(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(_) => "boolean".to_string(),
        serde_json::Value::Number(n) => {
            if n.is_f64() && n.as_f64().is_some_and(|f| f.fract() != 0.0) {
                "number".to_string()
            } else {
                "integer".to_string()
            }
        }
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "array".to_string(),
        serde_json::Value::Object(_) => "object".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_string_basic() {
        let schema = OutputSchema::from_field(SchemaField::string());
        let validator = SchemaValidator::new();

        let value = serde_json::json!("hello");
        assert!(validator.validate(&value, &schema).is_ok());

        let bad_value = serde_json::json!(42);
        assert!(validator.validate(&bad_value, &schema).is_err());
    }

    #[test]
    fn test_validate_string_length() {
        let schema =
            OutputSchema::from_field(SchemaField::string().with_min_length(2).with_max_length(5));
        let validator = SchemaValidator::new();

        assert!(validator
            .validate(&serde_json::json!("ab"), &schema)
            .is_ok());
        assert!(validator
            .validate(&serde_json::json!("abcde"), &schema)
            .is_ok());
        assert!(validator
            .validate(&serde_json::json!("a"), &schema)
            .is_err());
        assert!(validator
            .validate(&serde_json::json!("abcdef"), &schema)
            .is_err());
    }

    #[test]
    fn test_validate_integer() {
        let schema =
            OutputSchema::from_field(SchemaField::integer().with_minimum(0.0).with_maximum(100.0));
        let validator = SchemaValidator::new();

        assert!(validator.validate(&serde_json::json!(0), &schema).is_ok());
        assert!(validator.validate(&serde_json::json!(50), &schema).is_ok());
        assert!(validator.validate(&serde_json::json!(100), &schema).is_ok());
        assert!(validator.validate(&serde_json::json!(-1), &schema).is_err());
        assert!(validator
            .validate(&serde_json::json!(101), &schema)
            .is_err());
        assert!(validator
            .validate(&serde_json::json!(3.15), &schema)
            .is_err());
    }

    #[test]
    fn test_validate_object_with_required_fields() {
        let schema = OutputSchema::object()
            .with_required("name", SchemaField::string().with_min_length(1))
            .with_required("age", SchemaField::integer().with_minimum(0.0))
            .with_optional("email", SchemaField::string());

        let validator = SchemaValidator::new();

        let valid = serde_json::json!({
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com"
        });
        assert!(validator.validate(&valid, &schema).is_ok());

        let missing_required = serde_json::json!({
            "name": "Bob"
        });
        assert!(validator.validate(&missing_required, &schema).is_err());

        let wrong_type = serde_json::json!({
            "name": "Charlie",
            "age": "thirty"
        });
        assert!(validator.validate(&wrong_type, &schema).is_err());

        let without_optional = serde_json::json!({
            "name": "Diana",
            "age": 25
        });
        assert!(validator.validate(&without_optional, &schema).is_ok());
    }

    #[test]
    fn test_validate_array() {
        let schema = OutputSchema::from_field(
            SchemaField::array(SchemaField::integer())
                .with_min_items(1)
                .with_max_items(3),
        );
        let validator = SchemaValidator::new();

        assert!(validator
            .validate(&serde_json::json!([1, 2, 3]), &schema)
            .is_ok());
        assert!(validator.validate(&serde_json::json!([1]), &schema).is_ok());
        assert!(validator.validate(&serde_json::json!([]), &schema).is_err());
        assert!(validator
            .validate(&serde_json::json!([1, 2, 3, 4]), &schema)
            .is_err());
        assert!(validator
            .validate(&serde_json::json!(["not int"]), &schema)
            .is_err());
    }

    #[test]
    fn test_validate_nested_object() {
        let address_schema = SchemaField {
            field_type: FieldType::Object,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
            items: None,
            properties: Some({
                let mut m = HashMap::new();
                m.insert("city".to_string(), SchemaField::string().with_min_length(1));
                m.insert("zip".to_string(), SchemaField::string().with_min_length(1));
                m
            }),
            required: Some(vec!["city".to_string(), "zip".to_string()]),
        };

        let schema = OutputSchema::object()
            .with_required("name", SchemaField::string())
            .with_required("address", address_schema);

        let validator = SchemaValidator::new();

        let valid = serde_json::json!({
            "name": "Eve",
            "address": {
                "city": "NYC",
                "zip": "10001"
            }
        });
        assert!(validator.validate(&valid, &schema).is_ok());

        let missing_nested = serde_json::json!({
            "name": "Eve",
            "address": {
                "city": "NYC"
            }
        });
        assert!(validator.validate(&missing_nested, &schema).is_err());
    }

    #[test]
    fn test_null_type() {
        let schema = OutputSchema::from_field(SchemaField::null());
        let validator = SchemaValidator::new();

        assert!(validator
            .validate(&serde_json::Value::Null, &schema)
            .is_ok());
        assert!(validator
            .validate(&serde_json::json!("not null"), &schema)
            .is_err());
    }

    #[test]
    fn test_boolean_type() {
        let schema = OutputSchema::from_field(SchemaField::boolean());
        let validator = SchemaValidator::new();

        assert!(validator
            .validate(&serde_json::json!(true), &schema)
            .is_ok());
        assert!(validator
            .validate(&serde_json::json!(false), &schema)
            .is_ok());
        assert!(validator
            .validate(&serde_json::json!("true"), &schema)
            .is_err());
    }
}
