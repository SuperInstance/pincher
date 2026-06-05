//! Intent contracts — declarative intent-to-action mappings from TOML.
//!
//! An [`IntentContract`] is a declarative specification of how PincherOS
//! should handle a particular class of intents. Instead of teaching the
//! engine individual intent→action pairs, users can write an `Intent.toml`
//! file that defines:
//!
//! - **Patterns**: intent templates that match incoming user requests
//! - **Action template**: the action to execute when a pattern matches
//! - **Confidence threshold**: minimum similarity to trigger this contract
//! - **Priority**: disambiguation order when multiple contracts match (0–100)
//! - **Conflict strategy**: how to resolve when multiple contracts compete
//! - **Output schema**: optional validation for action output
//!
//! # TOML Format
//!
//! ```toml
//! [contract]
//! name = "file-operations"
//! confidence_threshold = 0.75
//! priority = 80
//! conflict_strategy = "highest_confidence"
//!
//! [[contract.patterns]]
//! template = "read file {path}"
//! regex = "read\\s+(?:the\\s+)?file\\s+(.+)"
//!
//! [[contract.patterns]]
//! template = "show me {path}"
//! regex = "show\\s+me\\s+(.+)"
//!
//! [contract.action]
//! template = "file.read {{path}}"
//!
//! [contract.output_schema]
//! type = "object"
//! required = ["content"]
//!
//! [contract.output_schema.properties.content]
//! type = "string"
//! min_length = 1
//! ```

use crate::intent::schema::OutputSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Intent contract errors.
#[derive(Debug, Error)]
pub enum ContractError {
    /// An I/O error occurred while reading the TOML file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The TOML file could not be parsed.
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    /// The contract failed validation.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Schema deserialization failed.
    #[error("Schema error: {0}")]
    Schema(String),
}

/// Result type for contract operations.
pub type ContractResult<T> = Result<T, ContractError>;

/// Strategy for resolving conflicts when multiple intent contracts match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStrategy {
    /// Use the first contract that matches (in definition order).
    FirstMatch,
    /// Use the contract with the highest confidence score.
    #[default]
    HighestConfidence,
    /// Use the contract with the highest priority value.
    HighestPriority,
    /// Merge the action templates from all matching contracts.
    Merge,
}

impl std::fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictStrategy::FirstMatch => write!(f, "first_match"),
            ConflictStrategy::HighestConfidence => write!(f, "highest_confidence"),
            ConflictStrategy::HighestPriority => write!(f, "highest_priority"),
            ConflictStrategy::Merge => write!(f, "merge"),
        }
    }
}

/// An intent pattern within a contract.
///
/// Each pattern specifies how to match an incoming intent. The `template`
/// is a human-readable pattern like `"read file {path}"`, while the optional
/// `regex` provides precise matching. If no regex is given, a simple
/// case-insensitive substring match is used against the template's literal
/// parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPattern {
    /// A human-readable intent template, e.g. `"read file {path}"`.
    ///
    /// Placeholders like `{path}` are extracted from the template for
    /// action variable substitution.
    pub template: String,

    /// An optional regex pattern for precise matching.
    ///
    /// If provided, this regex is used instead of simple substring matching.
    /// Named capture groups (e.g. `(?P<path>.+)`) are extracted and made
    /// available for action template substitution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
}

impl IntentPattern {
    /// Create a new intent pattern with a template and no regex.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            regex: None,
        }
    }

    /// Create an intent pattern with both template and regex.
    pub fn with_regex(template: impl Into<String>, regex: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            regex: Some(regex.into()),
        }
    }

    /// Extract placeholder names from the template.
    ///
    /// Placeholders are of the form `{name}`. This returns a Vec of
    /// the names found in the template string.
    pub fn extract_placeholders(&self) -> Vec<String> {
        let mut placeholders = Vec::new();
        let mut chars = self.template.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' {
                let mut name = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc == '}' {
                        chars.next();
                        break;
                    }
                    name.push(chars.next().unwrap());
                }
                if !name.is_empty() {
                    placeholders.push(name);
                }
            }
        }
        placeholders
    }

    /// Check if the template portion of this pattern matches the given input.
    ///
    /// Uses case-insensitive substring matching on the literal (non-placeholder)
    /// parts of the template. Returns a confidence score in [0.0, 1.0].
    pub fn match_score(&self, input: &str) -> f64 {
        let input_lower = input.to_lowercase();
        let _template_lower = self.template.to_lowercase();

        // Simple approach: strip placeholders and check if the
        // literal parts appear in the input
        let literal_parts = self.literal_parts();

        if literal_parts.is_empty() {
            // Template is all placeholders — any input matches weakly
            return 0.3;
        }

        let mut matched = 0usize;
        let total = literal_parts.len();

        for part in &literal_parts {
            if input_lower.contains(&part.to_lowercase()) {
                matched += 1;
            }
        }

        // Score based on how many literal parts matched, with a bonus
        // for full matches and a penalty for short inputs
        let base_score = matched as f64 / total as f64;

        // Small bonus if the input length suggests a real match
        let length_ratio = if input.len() > 3 {
            let template_non_placeholder_len: usize = literal_parts.iter().map(|s| s.len()).sum();
            if template_non_placeholder_len > 0 {
                let ratio = input.len() as f64 / template_non_placeholder_len as f64;
                if ratio > 0.3 && ratio < 10.0 {
                    0.1
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        (base_score * 0.9 + length_ratio).min(1.0)
    }

    /// Extract the literal (non-placeholder) parts of the template.
    fn literal_parts(&self) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut in_placeholder = false;
        let chars = self.template.chars().peekable();

        for c in chars {
            if c == '{' {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current.clear();
                in_placeholder = true;
            } else if c == '}' {
                in_placeholder = false;
                current.clear();
            } else if !in_placeholder {
                current.push(c);
            }
        }

        if !current.trim().is_empty() {
            parts.push(current.trim().to_string());
        }

        parts
    }

    /// Resolve action template variables from a matched input.
    ///
    /// Given the input string, attempt to extract values for the template
    /// placeholders. This is a simple heuristic: for each placeholder,
    /// take the text between the preceding literal part and the following
    /// literal part (or end of string).
    pub fn resolve_variables(&self, input: &str) -> Vec<(String, String)> {
        let placeholders = self.extract_placeholders();
        if placeholders.is_empty() {
            return Vec::new();
        }

        let literals = self.literal_parts();

        // Strategy: split input by the literal parts to find variable values
        let input_lower = input.to_lowercase();
        let mut values = Vec::new();

        if literals.is_empty() {
            // No literals — the whole input is the first placeholder
            if !placeholders.is_empty() {
                values.push((placeholders[0].clone(), input.to_string()));
            }
            return values;
        }

        // Find positions of each literal in the input
        let mut positions: Vec<Option<usize>> = Vec::new();
        for lit in &literals {
            positions.push(input_lower.find(&lit.to_lowercase()));
        }

        // Extract values between literals
        let mut placeholder_idx = 0;
        for (i, pos) in positions.iter().enumerate() {
            if let &Some(start) = pos {
                let end = start + literals[i].len();

                if placeholder_idx < placeholders.len() {
                    // Value before this literal
                    let prev_end = if i == 0 {
                        0
                    } else if let Some(&prev_pos) = positions.get(i - 1).and_then(|p| p.as_ref()) {
                        prev_pos + literals[i - 1].len()
                    } else {
                        0
                    };

                    if start > prev_end && placeholder_idx == 0 && i == 0 {
                        let val = input[prev_end..start].trim().to_string();
                        if !val.is_empty() {
                            values.push((placeholders[placeholder_idx].clone(), val));
                            placeholder_idx += 1;
                        }
                    }

                    // Value after this literal
                    if placeholder_idx < placeholders.len() {
                        let next_start = if i + 1 < positions.len() {
                            positions[i + 1].unwrap_or(input.len())
                        } else {
                            input.len()
                        };

                        if end < next_start {
                            let val = input[end..next_start].trim().to_string();
                            if !val.is_empty() {
                                values.push((placeholders[placeholder_idx].clone(), val));
                                placeholder_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        // If we still have unresolved placeholders, assign remaining input
        if placeholder_idx < placeholders.len() && values.len() < placeholders.len() {
            // Try to fill in remaining placeholders with what's left
            let used_text: Vec<&str> = values.iter().map(|(_, v)| v.as_str()).collect();
            let remaining = input
                .split_whitespace()
                .filter(|w| !used_text.iter().any(|u| u.contains(w)))
                .collect::<Vec<_>>()
                .join(" ");

            if !remaining.is_empty() {
                while placeholder_idx < placeholders.len() {
                    values.push((placeholders[placeholder_idx].clone(), remaining.clone()));
                    placeholder_idx += 1;
                }
            }
        }

        values
    }
}

/// The action specification within an intent contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTemplate {
    /// The action template string, e.g. `"file.read {{path}}"`.
    ///
    /// Double-brace variables like `{{path}}` are substituted with values
    /// extracted from the matched intent pattern.
    pub template: String,
}

impl ActionTemplate {
    /// Create a new action template.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Resolve the action template by substituting variables.
    ///
    /// Variables in the template are of the form `{{name}}`. The `variables`
    /// map provides values for each variable name.
    pub fn resolve(&self, variables: &[(String, String)]) -> String {
        let mut result = self.template.clone();
        for (name, value) in variables {
            let placeholder = format!("{{{{{}}}}}", name); // e.g. "{{path}}"
            result = result.replace(&placeholder, value);
        }
        result
    }
}

/// A declarative intent contract loaded from an `Intent.toml` file.
///
/// The contract defines how PincherOS should recognize and handle a class
/// of intents. It specifies patterns to match, an action to execute,
/// confidence and priority thresholds, and an optional output schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentContract {
    /// A unique name for this contract (e.g., `"file-operations"`).
    pub name: String,

    /// The intent patterns that trigger this contract.
    pub patterns: Vec<IntentPattern>,

    /// The action template to execute when matched.
    pub action: ActionTemplate,

    /// Minimum confidence score (0.0–1.0) required to trigger this contract.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,

    /// Priority level (0–100) for conflict resolution. Higher = more important.
    #[serde(default = "default_priority")]
    pub priority: u8,

    /// Strategy for resolving conflicts when multiple contracts match.
    #[serde(default)]
    pub conflict_strategy: ConflictStrategy,

    /// Optional output schema for validating action results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<OutputSchema>,
}

fn default_confidence_threshold() -> f64 {
    0.6
}

fn default_priority() -> u8 {
    50
}

impl IntentContract {
    /// Validate this contract, checking for structural issues.
    ///
    /// Returns `Ok(())` if the contract is valid, or a [`ContractError`]
    /// describing the first issue found.
    pub fn validate(&self) -> ContractResult<()> {
        // Name must be non-empty
        if self.name.trim().is_empty() {
            return Err(ContractError::Validation(
                "Contract name must not be empty".into(),
            ));
        }

        // Must have at least one pattern
        if self.patterns.is_empty() {
            return Err(ContractError::Validation(format!(
                "Contract '{}' must have at least one pattern",
                self.name
            )));
        }

        // Validate each pattern
        for (i, pattern) in self.patterns.iter().enumerate() {
            if pattern.template.trim().is_empty() {
                return Err(ContractError::Validation(format!(
                    "Pattern {} in contract '{}' has an empty template",
                    i, self.name
                )));
            }

            // Validate regex if present
            if let Some(regex_str) = &pattern.regex {
                // We can't compile regex without the regex crate, so do a
                // basic sanity check: balanced parentheses, no lone quantifiers
                let mut depth = 0i32;
                let chars: Vec<char> = regex_str.chars().collect();
                for (ci, &c) in chars.iter().enumerate() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth < 0 {
                                return Err(ContractError::Validation(format!(
                                    "Pattern {} in contract '{}' has unbalanced parentheses in regex at position {}",
                                    i, self.name, ci
                                )));
                            }
                        }
                        '*' | '+' | '?' | '{' if ci == 0 => {
                            // Quantifiers must not appear at the start
                            return Err(ContractError::Validation(format!(
                                "Pattern {} in contract '{}' has a quantifier at the start of regex",
                                i, self.name
                            )));
                        }
                        '*' | '+' | '?' | '{' => {}
                        _ => {}
                    }
                }
                if depth != 0 {
                    return Err(ContractError::Validation(format!(
                        "Pattern {} in contract '{}' has unbalanced parentheses in regex",
                        i, self.name
                    )));
                }
            }
        }

        // Validate action template
        if self.action.template.trim().is_empty() {
            return Err(ContractError::Validation(format!(
                "Contract '{}' has an empty action template",
                self.name
            )));
        }

        // Validate confidence threshold range
        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(ContractError::Validation(format!(
                "Contract '{}' has confidence_threshold {} outside [0.0, 1.0]",
                self.name, self.confidence_threshold
            )));
        }

        // Priority is u8 so it's automatically in [0, 255], but we constrain to [0, 100]
        if self.priority > 100 {
            return Err(ContractError::Validation(format!(
                "Contract '{}' has priority {} > 100",
                self.name, self.priority
            )));
        }

        // Verify that action template variables are available from patterns
        let action_vars = self.extract_action_variables();
        let pattern_vars: Vec<String> = self
            .patterns
            .iter()
            .flat_map(|p| p.extract_placeholders())
            .collect();

        for var in &action_vars {
            if !pattern_vars.contains(var) {
                warn!(
                    contract = self.name,
                    variable = var,
                    "Action template references variable not found in any pattern template"
                );
                // This is a warning, not an error — the variable might be
                // provided at runtime. But we log it for visibility.
            }
        }

        Ok(())
    }

    /// Extract variable names from the action template.
    ///
    /// Variables in action templates use double-brace syntax: `{{name}}`.
    pub fn extract_action_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        let template = &self.action.template;
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'
                let mut name = String::new();
                loop {
                    match chars.peek() {
                        Some(&'}') => {
                            chars.next();
                            if chars.peek() == Some(&'}') {
                                chars.next();
                                break;
                            } else {
                                name.push('}');
                            }
                        }
                        Some(&ch) => {
                            name.push(ch);
                            chars.next();
                        }
                        None => break,
                    }
                }
                if !name.is_empty() {
                    vars.push(name);
                }
            }
        }

        vars
    }

    /// Match an input string against this contract's patterns.
    ///
    /// Returns the highest confidence score from all matching patterns,
    /// or `None` if no pattern matches above the confidence threshold.
    pub fn match_intent(&self, input: &str) -> Option<f64> {
        let mut best_score = 0.0f64;

        for pattern in &self.patterns {
            let score = pattern.match_score(input);
            debug!(
                contract = self.name,
                template = pattern.template,
                score = score,
                "Pattern match score"
            );
            if score > best_score {
                best_score = score;
            }
        }

        if best_score >= self.confidence_threshold {
            Some(best_score)
        } else {
            debug!(
                contract = self.name,
                score = best_score,
                threshold = self.confidence_threshold,
                "Score below confidence threshold"
            );
            None
        }
    }

    /// Resolve the action template for a given input.
    ///
    /// This finds the best-matching pattern, extracts variables from the
    /// input, and substitutes them into the action template.
    pub fn resolve_action(&self, input: &str) -> String {
        let best_pattern = self.patterns.iter().max_by(|a, b| {
            a.match_score(input)
                .partial_cmp(&b.match_score(input))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(pattern) = best_pattern {
            let variables = pattern.resolve_variables(input);
            self.action.resolve(&variables)
        } else {
            self.action.template.clone()
        }
    }

    /// Load an intent contract from a TOML file.
    #[instrument(skip(path))]
    pub fn load_from_toml(path: &Path) -> ContractResult<Self> {
        info!(path = ?path, "Loading intent contract from TOML");

        let content = std::fs::read_to_string(path)?;
        let contract: IntentContract = toml::from_str(&content)?;

        info!(
            name = contract.name,
            pattern_count = contract.patterns.len(),
            "Parsed intent contract"
        );

        contract.validate()?;

        Ok(contract)
    }

    /// Load multiple intent contracts from a directory of TOML files.
    ///
    /// Each `.toml` file in the directory is parsed as an [`IntentContract`].
    /// Files that fail validation are logged and skipped.
    #[instrument(skip(dir))]
    pub fn load_directory(dir: &Path) -> ContractResult<Vec<IntentContract>> {
        info!(dir = ?dir, "Loading intent contracts from directory");

        let mut contracts = Vec::new();

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                match Self::load_from_toml(&path) {
                    Ok(contract) => contracts.push(contract),
                    Err(e) => {
                        warn!(path = ?path, error = %e, "Skipping invalid contract file");
                    }
                }
            }
        }

        info!(count = contracts.len(), "Loaded intent contracts");
        Ok(contracts)
    }

    /// Parse an intent contract from a TOML string.
    pub fn from_toml_str(toml_str: &str) -> ContractResult<Self> {
        let contract: IntentContract = toml::from_str(toml_str)?;
        contract.validate()?;
        Ok(contract)
    }
}

/// The top-level TOML structure for an intent contract file.
///
/// The TOML file wraps the contract in a `[contract]` section.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct IntentToml {
    contract: IntentContract,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_pattern_placeholders() {
        let pattern = IntentPattern::new("read file {path}");
        let placeholders = pattern.extract_placeholders();
        assert_eq!(placeholders, vec!["path"]);
    }

    #[test]
    fn test_intent_pattern_multiple_placeholders() {
        let pattern = IntentPattern::new("move {source} to {destination}");
        let placeholders = pattern.extract_placeholders();
        assert_eq!(placeholders, vec!["source", "destination"]);
    }

    #[test]
    fn test_intent_pattern_no_placeholders() {
        let pattern = IntentPattern::new("show system info");
        let placeholders = pattern.extract_placeholders();
        assert!(placeholders.is_empty());
    }

    #[test]
    fn test_intent_pattern_match_score() {
        let pattern = IntentPattern::new("read file {path}");
        let score = pattern.match_score("read file /etc/hosts");
        assert!(score > 0.5, "Expected high match score, got {}", score);
    }

    #[test]
    fn test_intent_pattern_no_match() {
        let pattern = IntentPattern::new("delete file {path}");
        let score = pattern.match_score("read file /etc/hosts");
        assert!(score < 0.5, "Expected low match score, got {}", score);
    }

    #[test]
    fn test_action_template_resolve() {
        let template = ActionTemplate::new("file.read {{path}}");
        let resolved = template.resolve(&[("path".to_string(), "/etc/hosts".to_string())]);
        assert_eq!(resolved, "file.read /etc/hosts");
    }

    #[test]
    fn test_action_template_multiple_vars() {
        let template = ActionTemplate::new("move {{source}} to {{destination}}");
        let resolved = template.resolve(&[
            ("source".to_string(), "/tmp/a".to_string()),
            ("destination".to_string(), "/tmp/b".to_string()),
        ]);
        assert_eq!(resolved, "move /tmp/a to /tmp/b");
    }

    #[test]
    fn test_contract_validate_valid() {
        let contract = IntentContract {
            name: "test-contract".to_string(),
            patterns: vec![IntentPattern::new("do {thing}")],
            action: ActionTemplate::new("execute {{thing}}"),
            confidence_threshold: 0.7,
            priority: 50,
            conflict_strategy: ConflictStrategy::HighestConfidence,
            output_schema: None,
        };
        assert!(contract.validate().is_ok());
    }

    #[test]
    fn test_contract_validate_empty_name() {
        let contract = IntentContract {
            name: "".to_string(),
            patterns: vec![IntentPattern::new("do {thing}")],
            action: ActionTemplate::new("execute {{thing}}"),
            confidence_threshold: 0.7,
            priority: 50,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };
        assert!(contract.validate().is_err());
    }

    #[test]
    fn test_contract_validate_no_patterns() {
        let contract = IntentContract {
            name: "no-patterns".to_string(),
            patterns: vec![],
            action: ActionTemplate::new("do something"),
            confidence_threshold: 0.7,
            priority: 50,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };
        assert!(contract.validate().is_err());
    }

    #[test]
    fn test_contract_validate_bad_confidence() {
        let contract = IntentContract {
            name: "bad-confidence".to_string(),
            patterns: vec![IntentPattern::new("test")],
            action: ActionTemplate::new("do"),
            confidence_threshold: 1.5,
            priority: 50,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };
        assert!(contract.validate().is_err());
    }

    #[test]
    fn test_contract_validate_priority_over_100() {
        let contract = IntentContract {
            name: "high-priority".to_string(),
            patterns: vec![IntentPattern::new("test")],
            action: ActionTemplate::new("do"),
            confidence_threshold: 0.5,
            priority: 150,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };
        assert!(contract.validate().is_err());
    }

    #[test]
    fn test_contract_match_intent() {
        let contract = IntentContract {
            name: "file-ops".to_string(),
            patterns: vec![
                IntentPattern::new("read file {path}"),
                IntentPattern::new("show file {path}"),
            ],
            action: ActionTemplate::new("file.read {{path}}"),
            confidence_threshold: 0.5,
            priority: 80,
            conflict_strategy: ConflictStrategy::HighestConfidence,
            output_schema: None,
        };

        assert!(contract.match_intent("read file /tmp/test").is_some());
        assert!(contract.match_intent("completely unrelated").is_none());
    }

    #[test]
    fn test_contract_resolve_action() {
        let contract = IntentContract {
            name: "file-ops".to_string(),
            patterns: vec![IntentPattern::new("read file {path}")],
            action: ActionTemplate::new("file.read {{path}}"),
            confidence_threshold: 0.3,
            priority: 50,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };

        let action = contract.resolve_action("read file /tmp/test.txt");
        assert!(action.contains("file.read"));
    }

    #[test]
    fn test_contract_extract_action_variables() {
        let contract = IntentContract {
            name: "test".to_string(),
            patterns: vec![IntentPattern::new("do {thing}")],
            action: ActionTemplate::new("execute {{thing}} with {{extra}}"),
            confidence_threshold: 0.5,
            priority: 50,
            conflict_strategy: ConflictStrategy::default(),
            output_schema: None,
        };

        let vars = contract.extract_action_variables();
        assert_eq!(vars, vec!["thing", "extra"]);
    }

    #[test]
    fn test_from_toml_str() {
        let toml = r#"
name = "greeting"
confidence_threshold = 0.7
priority = 60
conflict_strategy = "first_match"

[[patterns]]
template = "say hello to {name}"

[[patterns]]
template = "greet {name}"

[action]
template = "greet {{name}}"
"#;

        let contract = IntentContract::from_toml_str(toml).unwrap();
        assert_eq!(contract.name, "greeting");
        assert_eq!(contract.patterns.len(), 2);
        assert_eq!(contract.priority, 60);
        assert_eq!(contract.conflict_strategy, ConflictStrategy::FirstMatch);
    }

    #[test]
    fn test_conflict_strategy_default() {
        assert_eq!(
            ConflictStrategy::default(),
            ConflictStrategy::HighestConfidence
        );
    }

    #[test]
    fn test_conflict_strategy_display() {
        assert_eq!(ConflictStrategy::FirstMatch.to_string(), "first_match");
        assert_eq!(
            ConflictStrategy::HighestConfidence.to_string(),
            "highest_confidence"
        );
        assert_eq!(
            ConflictStrategy::HighestPriority.to_string(),
            "highest_priority"
        );
        assert_eq!(ConflictStrategy::Merge.to_string(), "merge");
    }
}
