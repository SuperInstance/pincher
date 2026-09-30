//! # Market Data Feed Adapter
//!
//! Ingests real or historical market data into the Matrix Engine feature tensor.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐    poll()     ┌──────────────┐  ingest()   ┌──────────────┐
//! │  Data Source  │ ──────────►  │  Data Feed    │ ──────────► │    Matrix    │
//! │  (CSV, WS…)   │              │  Adapter      │             │    Engine    │
//! └──────────────┘              └──────────────┘             └──────────────┘
//! ```
//!
//! - `MarketDataFeed` trait: the abstraction for any source
//! - `StockTick`: canonical tick representation
//! - `CsvFileFeed`: CSV-backed implementation for backtesting & replay
//! - Configurable replay speed for time-accelerated simulation

use crate::engine::MatrixEngine;
use crate::error::{HybridError, HybridResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────
// Core Types
// ─────────────────────────────────────────────────────────────────────

/// A single stock market tick — the atomic unit of market data.
///
/// Carries all fields needed for the Matrix Engine's feature tensor,
/// including price, volume, VWAP, and bid-ask spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockTick {
    /// Ticker symbol (e.g. "AAPL", "MSFT")
    pub ticker: String,
    /// Last trade price
    pub price: f64,
    /// Volume of the trade (shares)
    pub volume: f64,
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// Volume-weighted average price over the aggregation window
    pub vwap: f64,
    /// Bid-ask spread (ask - bid) in dollars
    pub spread: f64,
}

impl StockTick {
    /// Convert this tick into a feature vector for matrix ingestion.
    ///
    /// Feature order: `[price, volume, vwap, spread]`
    /// These map to the first 4 positions in the feature tensor `X[n_stocks, n_features, n_history]`.
    pub fn to_features(&self) -> Vec<f64> {
        vec![self.price, self.volume, self.vwap, self.spread]
    }

    /// Validate that all numeric fields are finite (no NaN/inf).
    pub fn is_valid(&self) -> bool {
        self.price.is_finite()
            && self.volume.is_finite()
            && self.vwap.is_finite()
            && self.spread.is_finite()
            && self.volume >= 0.0
            && self.spread >= 0.0
            && self.price > 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// MarketDataFeed Trait
// ─────────────────────────────────────────────────────────────────────

/// Abstraction over any market data source.
///
/// Implementations handle the source-specific polling logic (reading a CSV,
/// connecting to a WebSocket, pulling from an API) and return normalized
/// `StockTick` vectors.
pub trait MarketDataFeed: Send + Sync {
    /// Poll for the latest batch of stock ticks.
    ///
    /// Returns a `Vec<StockTick>` representing all ticks accumulated since
    /// the last poll. Returns an empty vec if no new data is available.
    fn poll(&self) -> Vec<StockTick>;

    /// Total number of ticks available in this feed (if known).
    fn len(&self) -> usize;

    /// Whether this feed has no ticks.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this feed has been fully consumed.
    fn is_exhausted(&self) -> bool;

    /// Reset the feed back to the beginning.
    fn reset(&self);
}

// ─────────────────────────────────────────────────────────────────────
// CsvFileFeed — CSV-backed feed for testing & backtesting
// ─────────────────────────────────────────────────────────────────────

/// A market data feed that reads ticks from a CSV file.
///
/// Supports:
/// - Historical data replay at configurable speed factors
/// - Non-destructive multiple reads (via interior mutability)
/// - Error-resilient CSV parsing with clear diagnostics
///
/// ## Replay Speed
///
/// - `speed = 0.0` — no waiting, replay as fast as possible
/// - `speed = 1.0` — wall-clock speed matching original timestamps
/// - `speed = N` — N× acceleration (e.g. 60.0 = one minute of data per wall-clock second)
///
/// ## CSV Format
///
/// ```csv
/// ticker,price,volume,timestamp,vwap,spread
/// AAPL,175.34,100000,1700000000000,174.89,0.02
/// MSFT,378.12,85000,1700000000000,377.45,0.01
/// ```
pub struct CsvFileFeed {
    /// All ticks loaded from the CSV (sorted by timestamp for replay).
    ticks: Vec<StockTick>,

    /// Current read position (interior mutability for &self poll).
    position: Mutex<usize>,

    /// Replay speed multiplier: 0 = fastest, 1 = real-time, N = N×.
    speed: f64,

    /// Wall-clock time of the last `poll()` call (for speed-governed replay).
    last_poll_time: Mutex<Option<Instant>>,

    /// Timestamp of the last tick emitted (for inter-tick timing).
    last_tick_ts: Mutex<Option<u64>>,

    /// Whether to validate ticks for finite values when reading.
    validate: bool,
}

impl CsvFileFeed {
    /// Number of feature fields extracted per tick for the tensor.
    pub const FEATURES_PER_TICK: usize = 4;

    /// Create a new CSV file feed from a path.
    ///
    /// # Errors
    ///
    /// Returns `HybridError::Io` if the file cannot be opened or read.
    /// Returns `HybridError::Internal` with details if CSV parsing fails
    /// on any row.
    pub fn new(path: impl AsRef<Path>) -> HybridResult<Self> {
        let path = path.as_ref();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .trim(csv::Trim::All)
            .from_path(path)
            .map_err(|e| {
                HybridError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to open CSV {}: {}", path.display(), e),
                ))
            })?;

        let headers = reader.headers().map_err(|e| {
            HybridError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read CSV headers from {}: {}", path.display(), e),
            ))
        })?;

        // Validate expected columns.
        const EXPECTED: [&str; 6] = ["ticker", "price", "volume", "timestamp", "vwap", "spread"];
        for (i, expected) in EXPECTED.iter().enumerate() {
            let actual = headers.get(i).unwrap_or("");
            if actual != *expected {
                return Err(HybridError::Internal(format!(
                    "CSV header mismatch at column {i}: expected '{expected}', got '{actual}' in {}",
                    path.display()
                )));
            }
        }

        let mut ticks: Vec<StockTick> = Vec::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record = result.map_err(|e| {
                HybridError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "CSV parse error at row {} in {}: {}",
                        row_idx + 2,
                        path.display(),
                        e
                    ),
                ))
            })?;

            let tick = Self::parse_record(&record, row_idx + 2, path)?;
            ticks.push(tick);
        }

        if ticks.is_empty() {
            return Err(HybridError::Internal(format!(
                "CSV file {} contains no data rows",
                path.display()
            )));
        }

        // Ensure ticks are sorted by timestamp for correct replay timing.
        ticks.sort_by_key(|t| t.timestamp);

        Ok(Self {
            ticks,
            position: Mutex::new(0),
            speed: 0.0,
            last_poll_time: Mutex::new(None),
            last_tick_ts: Mutex::new(None),
            validate: true,
        })
    }

    /// Parse a single CSV record into a `StockTick`.
    fn parse_record(
        record: &csv::StringRecord,
        row_num: usize,
        path: &Path,
    ) -> HybridResult<StockTick> {
        let get_field = |idx: usize| -> HybridResult<&str> {
            record.get(idx).ok_or_else(|| {
                HybridError::Internal(format!(
                    "Missing field at column {idx} in row {row_num} of {}",
                    path.display()
                ))
            })
        };

        let parse_f64 = |s: &str, field: &str| -> HybridResult<f64> {
            s.parse::<f64>().map_err(|e| {
                HybridError::Internal(format!(
                    "Cannot parse {field} '{}' as f64 in row {row_num} of {}: {e}",
                    s,
                    path.display()
                ))
            })
        };

        let ticker = get_field(0)?.trim().to_string();
        if ticker.is_empty() {
            return Err(HybridError::Internal(format!(
                "Empty ticker at row {row_num} of {}",
                path.display()
            )));
        }

        let price = parse_f64(get_field(1)?, "price")?;
        let volume = parse_f64(get_field(2)?, "volume")?;
        let ts_str = get_field(3)?;
        let timestamp: u64 = ts_str.parse::<u64>().map_err(|e| {
            HybridError::Internal(format!(
                "Cannot parse timestamp '{ts_str}' as u64 in row {row_num} of {}: {e}",
                path.display()
            ))
        })?;
        let vwap = parse_f64(get_field(4)?, "vwap")?;
        let spread = parse_f64(get_field(5)?, "spread")?;

        let tick = StockTick {
            ticker,
            price,
            volume,
            timestamp,
            vwap,
            spread,
        };

        Ok(tick)
    }

    /// Set the replay speed multiplier.
    ///
    /// - `0.0` = as fast as possible (no delays)
    /// - `1.0` = wall-clock speed matching source timestamps
    /// - `> 1.0` = accelerated replay
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.0);
    }

    /// Get the current replay speed.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Enable or disable tick validation (NaN/Inf checks).
    pub fn set_validate(&mut self, validate: bool) {
        self.validate = validate;
    }

    /// Number of ticks loaded from the CSV.
    pub fn total_ticks(&self) -> usize {
        self.ticks.len()
    }

    /// Peek at the next tick without consuming it.
    pub fn peek(&self) -> Option<&StockTick> {
        let pos = *self.position.lock().expect("position lock");
        self.ticks.get(pos)
    }

    /// Read a batch of ticks and feed them into a Matrix Engine's feature tensor.
    ///
    /// This is the primary integration point: it calls `poll()` to get the
    /// latest ticks, converts each to features, and calls `matrix.ingest()`
    /// for each.
    ///
    /// If replay speed > 0, it enforces wall-clock delays between batches
    /// to match the original tick timing.
    ///
    /// # Returns
    ///
    /// The number of ticks ingested into the tensor this cycle.
    pub async fn feed_to_tensor<M: MatrixEngine>(
        &self,
        matrix: &M,
        tick: u64,
    ) -> HybridResult<usize> {
        // Apply replay-speed timing gate before polling.
        self.enforce_speed_gate();

        let ticks = self.poll();
        let count = ticks.len();

        if count == 0 {
            return Ok(0);
        }

        for t in &ticks {
            // Validate tick data if enabled.
            if self.validate && !t.is_valid() {
                return Err(HybridError::Internal(format!(
                    "Invalid tick data for {} at ts={}: \
                     non-finite or non-positive value detected",
                    t.ticker, t.timestamp
                )));
            }

            let features = t.to_features();
            matrix.ingest(&t.ticker, &features, tick).await;
        }

        // Update last tick timestamp for inter-batch timing.
        if let Some(last) = ticks.last() {
            let mut last_ts = self.last_tick_ts.lock().expect("last_tick_ts lock");
            *last_ts = Some(last.timestamp);
        }

        Ok(count)
    }

    /// Enforce replay-speed wall-clock gating between poll calls.
    ///
    /// When `speed > 0`, calculates the expected delay from the original
    /// timestamps and sleeps for the appropriate wall-clock duration.
    /// When `speed == 0`, no delay is applied.
    fn enforce_speed_gate(&self) {
        let speed = self.speed;
        if speed <= 0.0 {
            return;
        }

        let last_ts = *self.last_tick_ts.lock().expect("last_tick_ts lock");
        let Some(prev_ts) = last_ts else {
            // First poll — record the baseline timestamp and return.
            return;
        };

        // Get the next tick's timestamp to calculate the inter-tick interval.
        let pos = *self.position.lock().expect("position lock");
        let Some(next_tick) = self.ticks.get(pos) else {
            return; // Exhausted.
        };

        let dt_ms = (next_tick.timestamp.saturating_sub(prev_ts)) as f64 / speed;
        if dt_ms <= 0.0 {
            return;
        }

        let delay = Duration::from_secs_f64(dt_ms / 1000.0);
        std::thread::sleep(delay);
    }

    /// Return all loaded ticks (for inspection / debugging).
    pub fn all_ticks(&self) -> &[StockTick] {
        &self.ticks
    }
}

impl MarketDataFeed for CsvFileFeed {
    /// Poll the next batch of ticks from the CSV.
    ///
    /// A "batch" is a single tick for precise replay. Multiple ticks
    /// with the same timestamp are returned as one batch.
    fn poll(&self) -> Vec<StockTick> {
        let mut pos = self.position.lock().expect("position lock");
        if *pos >= self.ticks.len() {
            return Vec::new();
        }

        let current_ts = self.ticks[*pos].timestamp;

        // Collect all ticks sharing the same timestamp (a single batch).
        let batch_start = *pos;
        let mut batch_end = *pos;
        while batch_end < self.ticks.len() && self.ticks[batch_end].timestamp == current_ts {
            batch_end += 1;
        }

        let batch: Vec<StockTick> = self.ticks[batch_start..batch_end].to_vec();
        *pos = batch_end;

        // Record this poll's wall-clock time for speed gating.
        let mut last_poll = self.last_poll_time.lock().expect("last_poll_time lock");
        *last_poll = Some(Instant::now());

        batch
    }

    fn len(&self) -> usize {
        self.ticks.len()
    }

    fn is_exhausted(&self) -> bool {
        let pos = self.position.lock().expect("position lock");
        *pos >= self.ticks.len()
    }

    fn reset(&self) {
        let mut pos = self.position.lock().expect("position lock");
        *pos = 0;
        let mut last_ts = self.last_tick_ts.lock().expect("last_tick_ts lock");
        *last_ts = None;
        let mut last_poll = self.last_poll_time.lock().expect("last_poll_time lock");
        *last_poll = None;
    }
}

impl std::fmt::Debug for CsvFileFeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsvFileFeed")
            .field("total_ticks", &self.ticks.len())
            .field("position", &self.position.lock().map(|p| *p).unwrap_or(0))
            .field("speed", &self.speed)
            .field("validate", &self.validate)
            .field("exhausted", &self.is_exhausted())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::MatrixEngine;
    use crate::types::{MatrixMetadata, MatrixSnapshot, PartialSnapshot};
    use async_trait::async_trait;
    use ndarray::Array1;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ── Spy Matrix Engine for testing ingestion ─────────────────────

    #[derive(Default)]
    struct SpyMatrixEngine {
        ingest_count: AtomicU64,
        last_ticker: std::sync::Mutex<Option<String>>,
        last_features: std::sync::Mutex<Option<Vec<f64>>>,
        last_tick: AtomicU64,
    }

    #[async_trait]
    impl MatrixEngine for SpyMatrixEngine {
        async fn ingest(&self, ticker: &str, features: &[f64], tick: u64) {
            self.ingest_count.fetch_add(1, Ordering::SeqCst);
            *self.last_ticker.lock().unwrap() = Some(ticker.to_string());
            *self.last_features.lock().unwrap() = Some(features.to_vec());
            self.last_tick.store(tick, Ordering::SeqCst);
        }

        async fn fast_cycle(&self, _tick: u64) -> MatrixMetadata {
            MatrixMetadata {
                tick: 0,
                n_stocks: 0,
                n_features: 0,
                n_history: 0,
                mean_correlation: 0.0,
                timestamp_ms: 0,
            }
        }

        async fn medium_cycle(&self, _tick: u64) -> PartialSnapshot {
            PartialSnapshot {
                tick: 0,
                n_stocks: 0,
                correlation_matrix_cond: 0.0,
                top_eigenvalues: vec![],
                regime: "unknown".into(),
                timestamp_ms: 0,
            }
        }

        async fn full_cycle(&self, tick: u64) -> MatrixSnapshot {
            MatrixSnapshot {
                tick,
                n_stocks: 5,
                eigenvalues: vec![],
                eigenvectors: ndarray::Array2::from_shape_vec((0, 0), vec![]).unwrap(),
                topologies: vec![],
                universe_betti: [0, 0, 0],
                regime: "unknown".into(),
                condition_number: 0.0,
            }
        }

        async fn get_slice(&self, _ticker: &str) -> Option<ndarray::Array2<f32>> {
            None
        }

        async fn get_cross_section(&self, _feature: &str, _time_idx: usize) -> Option<Array1<f32>> {
            None
        }

        async fn add_ticker(&self, _ticker: &str, _initial_features: Option<&[f64]>) {}

        async fn remove_ticker(&self, _ticker: &str) {}
    }

    // ── Fixture path helper ─────────────────────────────────────────

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    // ── Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_csv_new_loads_ticks() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");
        assert_eq!(feed.total_ticks(), 100);
        assert!(!feed.is_exhausted());
    }

    #[test]
    fn test_csv_poll_returns_tick_batches() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        let batch = feed.poll();
        assert_eq!(batch.len(), 5); // 5 tickers, same timestamp
        assert_eq!(batch[0].ticker, "AAPL");
        assert!(!feed.is_exhausted());
    }

    #[test]
    fn test_csv_poll_all_100_ticks() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        let mut total = 0;
        loop {
            let batch = feed.poll();
            if batch.is_empty() {
                break;
            }
            total += batch.len();
        }

        assert_eq!(total, 100);
        assert!(feed.is_exhausted());
    }

    #[test]
    fn test_csv_reset() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        // Consume half.
        for _ in 0..5 {
            feed.poll();
        }
        assert!(!feed.is_exhausted());

        feed.reset();
        assert!(!feed.is_exhausted());

        // First batch after reset should be back at the beginning.
        let batch = feed.poll();
        assert_eq!(batch.len(), 5);
        assert_eq!(batch[0].ticker, "AAPL");
    }

    #[test]
    fn test_peek() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        // Initial peek at first tick.
        let peeked = feed.peek().expect("Should have peekable tick");
        assert_eq!(peeked.ticker, "AAPL");
        assert_eq!(peeked.price, 174.82);

        // Peek should NOT advance position — calling poll returns the same batch.
        // After poll, the cursor moves past the first batch (5 ticks).
        // The next unpolled tick is AAPL again (batch 2, timestamp 1704115805000),
        // but with a different price.
        feed.poll();
        let next = feed.peek().expect("Should peek at next batch");
        assert_eq!(next.ticker, "AAPL");
        // The second batch should have a different price (drifted).
        assert_ne!(next.price, 174.82);
    }

    #[test]
    fn test_tick_validation() {
        let valid = StockTick {
            ticker: "AAPL".into(),
            price: 150.0,
            volume: 1000.0,
            timestamp: 1,
            vwap: 149.5,
            spread: 0.02,
        };
        assert!(valid.is_valid());

        let nan_price = StockTick {
            price: f64::NAN,
            ..valid.clone()
        };
        assert!(!nan_price.is_valid());

        let negative_vol = StockTick {
            volume: -100.0,
            ..valid.clone()
        };
        assert!(!negative_vol.is_valid());

        let zero_price = StockTick {
            price: 0.0,
            ..valid.clone()
        };
        assert!(!zero_price.is_valid());
    }

    #[test]
    fn test_to_features() {
        let tick = StockTick {
            ticker: "AAPL".into(),
            price: 150.0,
            volume: 1000.0,
            timestamp: 1,
            vwap: 149.5,
            spread: 0.02,
        };

        let features = tick.to_features();
        assert_eq!(features.len(), 4);
        assert!((features[0] - 150.0).abs() < 1e-10);
        assert!((features[1] - 1000.0).abs() < 1e-10);
        assert!((features[2] - 149.5).abs() < 1e-10);
        assert!((features[3] - 0.02).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_feed_to_tensor_ingests_ticks() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        let matrix = SpyMatrixEngine::default();

        // Feed first batch (5 ticks) at tick=1.
        let count = feed
            .feed_to_tensor(&matrix, 1)
            .await
            .expect("feed_to_tensor failed");
        assert_eq!(count, 5);

        // Verify ingest was called 5 times.
        assert_eq!(matrix.ingest_count.load(Ordering::SeqCst), 5);

        // Verify last ingested ticker.
        assert_eq!(matrix.last_ticker.lock().unwrap().as_deref(), Some("TSLA")); // 5th ticker sorted alphabetically

        // Verify features length.
        assert_eq!(
            matrix
                .last_features
                .lock()
                .unwrap()
                .as_ref()
                .map(|v| v.len()),
            Some(4)
        );

        // Verify tick was forwarded.
        assert_eq!(matrix.last_tick.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_feed_to_tensor_multiple_batches() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        let matrix = SpyMatrixEngine::default();

        // Feed 3 batches.
        for tick in 1..=3u64 {
            let count = feed.feed_to_tensor(&matrix, tick).await.unwrap();
            assert_eq!(count, 5, "batch {tick} should have 5 ticks");
        }

        assert_eq!(matrix.ingest_count.load(Ordering::SeqCst), 15);
    }

    #[tokio::test]
    async fn test_feed_to_tensor_exhaustion() {
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");

        let matrix = SpyMatrixEngine::default();

        // Consume all 20 batches.
        for tick in 1..=20u64 {
            let count = feed.feed_to_tensor(&matrix, tick).await.unwrap();
            assert_eq!(count, 5, "batch {tick}");
        }

        // Next call should return 0.
        let count = feed.feed_to_tensor(&matrix, 21).await.unwrap();
        assert_eq!(count, 0);
        assert!(feed.is_exhausted());
    }

    #[test]
    fn test_custom_csv_parse() {
        // Build CSV data in memory and write to a temp file.
        let dir = std::env::temp_dir();
        let path = dir.join("test_custom_datafeed.csv");
        let csv_data = "\
ticker,price,volume,timestamp,vwap,spread
AAPL,150.0,10000,1000,149.5,0.02
MSFT,350.0,5000,1000,348.2,0.01
GOOG,2800.0,2000,1001,2775.0,0.05
";
        std::fs::write(&path, csv_data).unwrap();

        let feed = CsvFileFeed::new(&path).expect("Failed to parse custom CSV");
        assert_eq!(feed.total_ticks(), 3);

        // First batch: AAPL + MSFT (same timestamp 1000).
        let batch1 = feed.poll();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0].ticker, "AAPL");
        assert!((batch1[0].price - 150.0).abs() < 1e-10);
        assert_eq!(batch1[1].ticker, "MSFT");

        // Second batch: GOOG (timestamp 1001).
        let batch2 = feed.poll();
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0].ticker, "GOOG");

        // Exhausted.
        assert!(feed.is_exhausted());

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_set_speed() {
        let mut feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");
        assert!((feed.speed() - 0.0).abs() < 1e-10);

        feed.set_speed(60.0);
        assert!((feed.speed() - 60.0).abs() < 1e-10);

        // Negative should clamp to 0.
        feed.set_speed(-1.0);
        assert!((feed.speed() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_missing_csv_returns_error() {
        let result = CsvFileFeed::new("/tmp/nonexistent_file_xyz.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_and_poll_known_tickers() {
        // Re-load and verify we see all 5 tickers in the first batch.
        let feed = CsvFileFeed::new(fixture_path("sample_5stocks.csv"))
            .expect("Failed to load sample CSV");
        let batch = feed.poll();
        let tickers: Vec<&str> = batch.iter().map(|t| t.ticker.as_str()).collect();
        assert_eq!(tickers, vec!["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]);
    }
}
