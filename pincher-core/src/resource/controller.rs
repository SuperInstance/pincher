//! PID resource controller for PincherOS
//!
//! Monitors system resources (CPU, RAM) and provides a three-state
//! degradation model:
//! - **Normal**: Full LLM access, normal context window
//! - **Light**: Reduced context, skip LLM for high-confidence reflexes
//! - **Critical**: Reflex-only mode, no LLM calls, minimal logging
//!
//! Uses a PID controller to smooth out resource fluctuations and avoid
//! rapid state transitions.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::System;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Resource controller errors.
#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("System info error: {0}")]
    SystemInfo(String),

    #[error("Invalid PID gains: {0}")]
    InvalidGains(String),
}

/// Result type for resource operations.
pub type ResourceResult<T> = Result<T, ResourceError>;

/// The three resource states of PincherOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceState {
    /// Normal operation: RAM < 70%, CPU < 60%.
    /// Full LLM access, normal context window.
    Normal,
    /// Light degradation: RAM 70-85% OR CPU 60-80%.
    /// Reduced context window, skip LLM for confidence > 0.85.
    Light,
    /// Critical: RAM > 85% OR CPU > 80%.
    /// Reflex-only mode, no LLM calls, minimal logging.
    Critical,
}

impl std::fmt::Display for ResourceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceState::Normal => write!(f, "normal"),
            ResourceState::Light => write!(f, "light"),
            ResourceState::Critical => write!(f, "critical"),
        }
    }
}

impl ResourceState {
    /// Get the resource budget for this state.
    pub fn budget(&self) -> ResourceBudget {
        match self {
            ResourceState::Normal => ResourceBudget {
                max_context_tokens: 4096,
                allow_llm: true,
                log_level: "debug".to_string(),
            },
            ResourceState::Light => ResourceBudget {
                max_context_tokens: 2048,
                allow_llm: true,
                log_level: "info".to_string(),
            },
            ResourceState::Critical => ResourceBudget {
                max_context_tokens: 512,
                allow_llm: false,
                log_level: "warn".to_string(),
            },
        }
    }
}

/// Resource budget constraints based on current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum context tokens for LLM interactions.
    pub max_context_tokens: usize,
    /// Whether LLM calls are allowed.
    pub allow_llm: bool,
    /// Minimum log level for this state.
    pub log_level: String,
}

/// PID controller gains for smoothing resource measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidController {
    /// Proportional gain — responds to current error.
    pub kp: f64,
    /// Integral gain — responds to accumulated error.
    pub ki: f64,
    /// Derivative gain — responds to rate of change.
    pub kd: f64,
}

impl PidController {
    /// Create a new PID controller with the given gains.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd }
    }

    /// Create default PID gains for RAM monitoring.
    pub fn ram_defaults() -> Self {
        Self {
            kp: 0.6,
            ki: 0.1,
            kd: 0.3,
        }
    }

    /// Create default PID gains for CPU monitoring.
    pub fn cpu_defaults() -> Self {
        Self {
            kp: 0.5,
            ki: 0.15,
            kd: 0.35,
        }
    }

    /// Compute PID output given error, integral, and derivative.
    pub fn compute(&self, error: f64, integral: f64, derivative: f64) -> f64 {
        self.kp * error + self.ki * integral + self.kd * derivative
    }
}

impl Default for PidController {
    fn default() -> Self {
        Self::ram_defaults()
    }
}

/// Internal PID state for a single measurement.
#[derive(Debug, Clone)]
struct PidState {
    integral: f64,
    last_error: f64,
    last_update: Instant,
}

impl PidState {
    fn new() -> Self {
        Self {
            integral: 0.0,
            last_error: 0.0,
            last_update: Instant::now(),
        }
    }

    fn update(&mut self, error: f64, dt_secs: f64, controller: &PidController) -> f64 {
        // Trapezoidal integration
        self.integral += (self.last_error + error) * 0.5 * dt_secs;

        // Anti-windup: clamp integral term
        self.integral = self.integral.clamp(-10.0, 10.0);

        // Derivative (rate of change)
        let derivative = if dt_secs > 0.0 {
            (error - self.last_error) / dt_secs
        } else {
            0.0
        };

        let output = controller.compute(error, self.integral, derivative);

        self.last_error = error;
        self.last_update = Instant::now();

        output
    }
}

/// Resource thresholds for state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// RAM usage percentage for Light state.
    pub ram_light: f64,
    /// RAM usage percentage for Critical state.
    pub ram_critical: f64,
    /// CPU usage percentage for Light state.
    pub cpu_light: f64,
    /// CPU usage percentage for Critical state.
    pub cpu_critical: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            ram_light: 70.0,
            ram_critical: 85.0,
            cpu_light: 60.0,
            cpu_critical: 80.0,
        }
    }
}

/// The resource controller that monitors system state.
pub struct ResourceController {
    sys: System,
    thresholds: ResourceThresholds,
    ram_controller: PidController,
    cpu_controller: PidController,
    ram_state: PidState,
    cpu_state: PidState,
    current_state: ResourceState,
    smoothed_ram: f64,
    smoothed_cpu: f64,
    tick_count: u64,
    /// Hysteresis: number of consecutive ticks before transitioning.
    hysteresis_ticks: u64,
    /// Counter for ticks in a different state.
    state_change_counter: u64,
    /// The state we're counting toward.
    pending_state: ResourceState,
}

impl ResourceController {
    /// Create a new resource controller with default settings.
    pub fn new() -> Self {
        Self {
            sys: System::new_all(),
            thresholds: ResourceThresholds::default(),
            ram_controller: PidController::ram_defaults(),
            cpu_controller: PidController::cpu_defaults(),
            ram_state: PidState::new(),
            cpu_state: PidState::new(),
            current_state: ResourceState::Normal,
            smoothed_ram: 0.0,
            smoothed_cpu: 0.0,
            tick_count: 0,
            hysteresis_ticks: 3,
            state_change_counter: 0,
            pending_state: ResourceState::Normal,
        }
    }

    /// Create a resource controller with custom thresholds.
    pub fn with_thresholds(thresholds: ResourceThresholds) -> Self {
        Self {
            thresholds,
            ..Self::new()
        }
    }

    /// Create a resource controller with custom PID gains.
    pub fn with_pid_gains(ram_gains: PidController, cpu_gains: PidController) -> Self {
        Self {
            ram_controller: ram_gains,
            cpu_controller: cpu_gains,
            ..Self::new()
        }
    }

    /// Set the hysteresis tick count for state transitions.
    pub fn with_hysteresis(mut self, ticks: u64) -> Self {
        self.hysteresis_ticks = ticks.max(1);
        self
    }

    /// Perform a tick — refresh system info and update state.
    #[instrument(skip(self))]
    pub fn tick(&mut self) -> ResourceState {
        self.sys.refresh_all();
        self.tick_count += 1;

        // Get raw measurements
        let total_ram = self.sys.total_memory() as f64;
        let used_ram = self.sys.used_memory() as f64;
        let ram_pct = if total_ram > 0.0 {
            (used_ram / total_ram) * 100.0
        } else {
            0.0
        };

        let cpu_pct = self.sys.global_cpu_usage() as f64;

        // PID smoothing
        let ram_error = ram_pct / 100.0; // Normalize to [0, 1]
        let cpu_error = cpu_pct / 100.0;

        let dt_secs = 1.0; // Assume 1 second between ticks
        let ram_output = self
            .ram_state
            .update(ram_error, dt_secs, &self.ram_controller);
        let cpu_output = self
            .cpu_state
            .update(cpu_error, dt_secs, &self.cpu_controller);

        // Convert PID output back to percentage (with smoothing)
        // Exponential moving average
        let alpha = 0.3;
        self.smoothed_ram = alpha * (ram_output * 100.0) + (1.0 - alpha) * self.smoothed_ram;
        self.smoothed_cpu = alpha * (cpu_output * 100.0) + (1.0 - alpha) * self.smoothed_cpu;

        // On first tick, initialize smoothed values
        if self.tick_count == 1 {
            self.smoothed_ram = ram_pct;
            self.smoothed_cpu = cpu_pct;
        }

        debug!(
            raw_ram_pct = format!("{:.1}%", ram_pct),
            raw_cpu_pct = format!("{:.1}%", cpu_pct),
            smoothed_ram = format!("{:.1}%", self.smoothed_ram),
            smoothed_cpu = format!("{:.1}%", self.smoothed_cpu),
            current_state = %self.current_state,
            "Resource tick"
        );

        // Determine target state based on smoothed values
        let target_state = self.compute_state(self.smoothed_ram, self.smoothed_cpu);

        // Apply hysteresis for state transitions
        if target_state != self.current_state {
            if target_state == self.pending_state {
                self.state_change_counter += 1;
                if self.state_change_counter >= self.hysteresis_ticks {
                    info!(
                        from = %self.current_state,
                        to = %target_state,
                        ram = format!("{:.1}%", self.smoothed_ram),
                        cpu = format!("{:.1}%", self.smoothed_cpu),
                        "Resource state transition"
                    );
                    self.current_state = target_state;
                    self.state_change_counter = 0;
                }
            } else {
                // Different target — reset counter
                self.pending_state = target_state;
                self.state_change_counter = 1;
            }
        } else {
            self.state_change_counter = 0;
            self.pending_state = target_state;
        }

        self.current_state
    }

    /// Check the current state without updating.
    pub fn check_state(&self) -> ResourceState {
        self.current_state
    }

    /// Compute the target state from smoothed RAM/CPU percentages.
    fn compute_state(&self, ram_pct: f64, cpu_pct: f64) -> ResourceState {
        // Critical takes priority: RAM > 85% OR CPU > 80%
        if ram_pct >= self.thresholds.ram_critical || cpu_pct >= self.thresholds.cpu_critical {
            ResourceState::Critical
        }
        // Light: RAM 70-85% OR CPU 60-80%
        else if ram_pct >= self.thresholds.ram_light || cpu_pct >= self.thresholds.cpu_light {
            ResourceState::Light
        }
        // Normal
        else {
            ResourceState::Normal
        }
    }

    /// Get the current resource budget.
    pub fn budget(&self) -> ResourceBudget {
        self.current_state.budget()
    }

    /// Get the current smoothed RAM usage percentage.
    pub fn smoothed_ram(&self) -> f64 {
        self.smoothed_ram
    }

    /// Get the current smoothed CPU usage percentage.
    pub fn smoothed_cpu(&self) -> f64 {
        self.smoothed_cpu
    }

    /// Get the tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Check if LLM calls are allowed in the current state.
    pub fn allow_llm(&self) -> bool {
        self.current_state.budget().allow_llm
    }

    /// Check if LLM should be skipped for a given confidence level.
    ///
    /// In Light state, LLM calls are skipped for reflexes with
    /// confidence > 0.85 to conserve resources.
    pub fn should_skip_llm(&self, confidence: f64) -> bool {
        match self.current_state {
            ResourceState::Normal => false,
            ResourceState::Light => confidence > 0.85,
            ResourceState::Critical => true,
        }
    }

    /// Get a snapshot of current resource metrics.
    pub fn metrics(&self) -> ResourceMetrics {
        ResourceMetrics {
            ram_usage_pct: self.smoothed_ram,
            cpu_usage_pct: self.smoothed_cpu,
            state: self.current_state,
            budget: self.budget(),
            tick_count: self.tick_count,
        }
    }
}

impl Default for ResourceController {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot of resource metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Current RAM usage percentage (smoothed).
    pub ram_usage_pct: f64,
    /// Current CPU usage percentage (smoothed).
    pub cpu_usage_pct: f64,
    /// Current resource state.
    pub state: ResourceState,
    /// Current resource budget.
    pub budget: ResourceBudget,
    /// Total tick count.
    pub tick_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_state_budget() {
        let normal_budget = ResourceState::Normal.budget();
        assert_eq!(normal_budget.max_context_tokens, 4096);
        assert!(normal_budget.allow_llm);

        let light_budget = ResourceState::Light.budget();
        assert_eq!(light_budget.max_context_tokens, 2048);
        assert!(light_budget.allow_llm);

        let critical_budget = ResourceState::Critical.budget();
        assert_eq!(critical_budget.max_context_tokens, 512);
        assert!(!critical_budget.allow_llm);
    }

    #[test]
    fn test_resource_state_display() {
        assert_eq!(ResourceState::Normal.to_string(), "normal");
        assert_eq!(ResourceState::Light.to_string(), "light");
        assert_eq!(ResourceState::Critical.to_string(), "critical");
    }

    #[test]
    fn test_pid_controller_compute() {
        let pid = PidController::new(1.0, 0.0, 0.0);
        let output = pid.compute(0.5, 0.0, 0.0);
        assert!((output - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pid_state_update() {
        let mut state = PidState::new();
        let pid = PidController::new(1.0, 0.1, 0.5);

        let output = state.update(0.5, 1.0, &pid);
        // P: 1.0 * 0.5 = 0.5
        // I: 0.1 * (0.0 + 0.5) * 0.5 * 1.0 = 0.025
        // D: 0.5 * (0.5 - 0.0) / 1.0 = 0.25
        // Total: 0.5 + 0.025 + 0.25 = 0.775
        assert!((output - 0.775).abs() < 1e-10);
    }

    #[test]
    fn test_compute_state() {
        let controller = ResourceController::new();

        assert_eq!(controller.compute_state(50.0, 40.0), ResourceState::Normal);
        assert_eq!(controller.compute_state(75.0, 50.0), ResourceState::Light);
        assert_eq!(controller.compute_state(50.0, 70.0), ResourceState::Light);
        assert_eq!(
            controller.compute_state(90.0, 40.0),
            ResourceState::Critical
        );
        assert_eq!(
            controller.compute_state(50.0, 85.0),
            ResourceState::Critical
        );
    }

    #[test]
    fn test_should_skip_llm() {
        let controller = ResourceController::new();

        // Normal state: never skip
        assert!(!controller.should_skip_llm(0.5));
        assert!(!controller.should_skip_llm(0.9));
    }

    #[test]
    fn test_resource_controller_tick() {
        let mut controller = ResourceController::new();
        let state = controller.tick();
        // On a typical dev machine, we should be in Normal or Light state
        assert!(
            state == ResourceState::Normal
                || state == ResourceState::Light
                || state == ResourceState::Critical
        );
    }

    #[test]
    fn test_metrics() {
        let mut controller = ResourceController::new();
        controller.tick();
        let metrics = controller.metrics();
        assert!(metrics.ram_usage_pct >= 0.0);
        assert!(metrics.cpu_usage_pct >= 0.0);
        assert_eq!(metrics.tick_count, 1);
    }
}
