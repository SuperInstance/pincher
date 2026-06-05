//! .nail format pack/unpack for PincherOS
//!
//! The .nail format is PincherOS's portable shell format — a tar.zst
//! archive containing the agent's complete state: reflexes, identity,
//! configuration, and integrity checksums. Like a hermit crab's shell,
//! it can be packed up and moved to a new environment.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Pack/unpack errors.
#[derive(Debug, Error)]
pub enum PackError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Fingerprint error: {0}")]
    Fingerprint(#[from] crate::migration::fingerprint::FingerprintError),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("Missing file in nail archive: {0}")]
    MissingFile(String),

    #[error("Database error: {0}")]
    Database(String),
}

/// Result type for pack operations.
pub type PackResult<T> = Result<T, PackError>;

/// Manifest file contained within a .nail archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NailManifest {
    /// Format version.
    pub version: String,
    /// Hardware fingerprint of the source shell.
    pub fingerprint: String,
    /// Timestamp when the nail was packed.
    pub timestamp: String,
    /// Number of reflexes in the archive.
    pub reflex_count: u64,
    /// BLAKE3 checksums of all contained files.
    pub checksums: NailChecksums,
}

/// BLAKE3 checksums for files in the .nail archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NailChecksums {
    /// Checksum of reflexes.db.
    pub reflexes_db: String,
    /// Checksum of identity.json.
    pub identity_json: String,
    /// Checksum of config.toml.
    pub config_toml: String,
}

/// Agent identity contained within a .nail archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// The agent's name.
    pub name: String,
    /// Agent preferences.
    pub preferences: AgentPreferences,
    /// Creation timestamp.
    pub created_at: String,
}

/// Agent preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPreferences {
    /// Preferred shell.
    pub preferred_shell: String,
    /// Preferred editor.
    pub preferred_editor: String,
    /// Preferred language for responses.
    pub language: String,
    /// Verbosity level (0-2).
    pub verbosity: u8,
}

impl Default for AgentPreferences {
    fn default() -> Self {
        Self {
            preferred_shell: "bash".to_string(),
            preferred_editor: "vim".to_string(),
            language: "en".to_string(),
            verbosity: 1,
        }
    }
}

impl Default for AgentIdentity {
    fn default() -> Self {
        Self {
            name: "Pinchy".to_string(),
            preferences: AgentPreferences::default(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Agent configuration contained within a .nail archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Resource thresholds.
    pub resource_thresholds: ResourceThresholdsConfig,
    /// Veto rules file path.
    pub veto_rules_path: String,
    /// Embedding model path.
    pub model_path: String,
    /// RPC socket path.
    pub rpc_socket_path: String,
}

/// Resource thresholds configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholdsConfig {
    pub ram_light: f64,
    pub ram_critical: f64,
    pub cpu_light: f64,
    pub cpu_critical: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            resource_thresholds: ResourceThresholdsConfig {
                ram_light: 70.0,
                ram_critical: 85.0,
                cpu_light: 60.0,
                cpu_critical: 80.0,
            },
            veto_rules_path: ".pincher/veto_rules.toml".to_string(),
            model_path: ".pincher/models/all-MiniLM-L6-v2-int8.onnx".to_string(),
            rpc_socket_path: "/tmp/pincher.sock".to_string(),
        }
    }
}

/// Compute the BLAKE3 hash of a file.
fn blake3_hash_file(path: &Path) -> PackResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hasher.finalize().to_hex().to_string())
}

/// Compute the BLAKE3 hash of bytes.
///
/// TODO: Will be used for inline checksum verification during pack/unpack
/// operations where file-level hashing isn't suitable (e.g., verifying
/// in-memory data before writing to disk).
#[allow(dead_code)]
pub fn blake3_hash_bytes(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

/// Pack a database into a .nail file (tar.zst archive).
///
/// The .nail archive contains:
/// - `manifest.json`: Version, fingerprint, timestamp, reflex count, checksums
/// - `reflexes.db`: SQLite database dump
/// - `identity.json`: Agent name and preferences
/// - `config.toml`: Agent configuration
#[instrument(skip(db_path, output))]
pub fn pack_nail(db_path: &Path, output: &Path) -> PackResult<()> {
    info!(
        db_path = ?db_path,
        output = ?output,
        "Packing .nail archive"
    );

    if !db_path.exists() {
        return Err(PackError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Database not found: {:?}", db_path),
        )));
    }

    // Create a temporary directory for staging
    let temp_dir = tempfile::tempdir().map_err(PackError::Io)?;
    let temp_path = temp_dir.path();

    // Stage 1: Copy the database
    let db_copy = temp_path.join("reflexes.db");
    fs::copy(db_path, &db_copy)?;

    // Stage 2: Create identity.json
    let identity = AgentIdentity::default();
    let identity_json = serde_json::to_string_pretty(&identity)?;
    let identity_path = temp_path.join("identity.json");
    fs::write(&identity_path, &identity_json)?;

    // Stage 3: Create config.toml
    let config = AgentConfig::default();
    let config_toml = toml::to_string_pretty(&config)
        .map_err(|e| PackError::Compression(format!("Failed to serialize config: {}", e)))?;
    let config_path = temp_path.join("config.toml");
    fs::write(&config_path, &config_toml)?;

    // Stage 4: Compute checksums
    let checksums = NailChecksums {
        reflexes_db: blake3_hash_file(&db_copy)?,
        identity_json: blake3_hash_file(&identity_path)?,
        config_toml: blake3_hash_file(&config_path)?,
    };

    // Stage 5: Count reflexes
    let reflex_count = count_reflexes_in_db(&db_copy)?;

    // Stage 6: Create manifest
    let manifest = NailManifest {
        version: "0.1.0".to_string(),
        fingerprint: crate::migration::fingerprint::fingerprint_hash(
            &crate::migration::fingerprint::fingerprint()?,
        ),
        timestamp: chrono::Utc::now().to_rfc3339(),
        reflex_count,
        checksums,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    let manifest_path = temp_path.join("manifest.json");
    fs::write(&manifest_path, &manifest_json)?;

    // Stage 7: Create tar.zst archive
    create_tar_zst(temp_path, output)?;

    info!(output = ?output, "Nail archive packed successfully");
    Ok(())
}

/// Unpack a .nail file to an output directory.
#[instrument(skip(nail_path, output_dir))]
pub fn unpack_nail(nail_path: &Path, output_dir: &Path) -> PackResult<()> {
    info!(
        nail_path = ?nail_path,
        output_dir = ?output_dir,
        "Unpacking .nail archive"
    );

    if !nail_path.exists() {
        return Err(PackError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Nail file not found: {:?}", nail_path),
        )));
    }

    fs::create_dir_all(output_dir)?;

    // Extract tar.zst archive
    extract_tar_zst(nail_path, output_dir)?;

    // Verify checksums
    let manifest_path = output_dir.join("manifest.json");
    if manifest_path.exists() {
        let manifest: NailManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;

        // Verify reflexes.db
        let db_path = output_dir.join("reflexes.db");
        if db_path.exists() {
            let actual = blake3_hash_file(&db_path)?;
            if actual != manifest.checksums.reflexes_db {
                return Err(PackError::ChecksumMismatch {
                    file: "reflexes.db".to_string(),
                    expected: manifest.checksums.reflexes_db,
                    actual,
                });
            }
        }

        // Verify identity.json
        let identity_path = output_dir.join("identity.json");
        if identity_path.exists() {
            let actual = blake3_hash_file(&identity_path)?;
            if actual != manifest.checksums.identity_json {
                return Err(PackError::ChecksumMismatch {
                    file: "identity.json".to_string(),
                    expected: manifest.checksums.identity_json,
                    actual,
                });
            }
        }

        // Verify config.toml
        let config_path = output_dir.join("config.toml");
        if config_path.exists() {
            let actual = blake3_hash_file(&config_path)?;
            if actual != manifest.checksums.config_toml {
                return Err(PackError::ChecksumMismatch {
                    file: "config.toml".to_string(),
                    expected: manifest.checksums.config_toml,
                    actual,
                });
            }
        }

        info!(
            reflex_count = manifest.reflex_count,
            "Nail archive verified and unpacked"
        );
    } else {
        warn!("No manifest.json found in nail archive — skipping verification");
    }

    Ok(())
}

/// Verify a .nail archive without extracting it.
#[instrument(skip(nail_path))]
pub fn verify_nail(nail_path: &Path) -> PackResult<bool> {
    info!(nail_path = ?nail_path, "Verifying .nail archive");

    let temp_dir = tempfile::tempdir().map_err(PackError::Io)?;
    let temp_path = temp_dir.path();

    // Extract to temp directory
    extract_tar_zst(nail_path, temp_path)?;

    // Check for required files
    let manifest_path = temp_path.join("manifest.json");
    if !manifest_path.exists() {
        warn!("No manifest.json in archive");
        return Ok(false);
    }

    let manifest: NailManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;

    // Verify all files exist
    let required_files = ["reflexes.db", "identity.json", "config.toml"];
    for file in &required_files {
        if !temp_path.join(file).exists() {
            warn!(file = file, "Required file missing from archive");
            return Ok(false);
        }
    }

    // Verify checksums
    let db_path = temp_path.join("reflexes.db");
    let actual_db_hash = blake3_hash_file(&db_path)?;
    if actual_db_hash != manifest.checksums.reflexes_db {
        warn!("reflexes.db checksum mismatch");
        return Ok(false);
    }

    let identity_path = temp_path.join("identity.json");
    let actual_identity_hash = blake3_hash_file(&identity_path)?;
    if actual_identity_hash != manifest.checksums.identity_json {
        warn!("identity.json checksum mismatch");
        return Ok(false);
    }

    let config_path = temp_path.join("config.toml");
    let actual_config_hash = blake3_hash_file(&config_path)?;
    if actual_config_hash != manifest.checksums.config_toml {
        warn!("config.toml checksum mismatch");
        return Ok(false);
    }

    info!("Nail archive verification passed");
    Ok(true)
}

/// Count reflexes in a SQLite database.
fn count_reflexes_in_db(db_path: &Path) -> PackResult<u64> {
    let conn = rusqlite::Connection::open(db_path)
        .map_err(|e| PackError::Database(format!("Failed to open database: {}", e)))?;

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM reflexes", [], |row| row.get(0))
        .map_err(|e| PackError::Database(format!("Failed to count reflexes: {}", e)))?;

    Ok(count as u64)
}

/// Create a tar.zst archive from a directory.
fn create_tar_zst(source_dir: &Path, output_path: &Path) -> PackResult<()> {
    debug!(source = ?source_dir, output = ?output_path, "Creating tar.zst archive");

    let output_file = File::create(output_path)?;
    let buf_writer = BufWriter::new(output_file);

    // Create zstd encoder
    let encoder = zstd::Encoder::new(buf_writer, 3)
        .map_err(|e| PackError::Compression(format!("Failed to create zstd encoder: {}", e)))?;

    // Create tar archive
    let mut tar_builder = tar::Builder::new(encoder);

    // Add all files in the source directory
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy();
            debug!(file = %file_name, "Adding file to archive");
            tar_builder.append_path_with_name(&path, &*file_name)?;
        }
    }

    // Finish the tar archive
    let encoder = tar_builder
        .into_inner()
        .map_err(|e| PackError::Compression(format!("Failed to finalize tar: {}", e)))?;

    // Finish the zstd compression
    encoder
        .finish()
        .map_err(|e| PackError::Compression(format!("Failed to finalize zstd: {}", e)))?;

    debug!("tar.zst archive created successfully");
    Ok(())
}

/// Extract a tar.zst archive to a directory.
fn extract_tar_zst(archive_path: &Path, output_dir: &Path) -> PackResult<()> {
    debug!(archive = ?archive_path, output = ?output_dir, "Extracting tar.zst archive");

    let input_file = File::open(archive_path)?;
    let buf_reader = BufReader::new(input_file);

    // Create zstd decoder
    let decoder = zstd::Decoder::new(buf_reader)
        .map_err(|e| PackError::Compression(format!("Failed to create zstd decoder: {}", e)))?;

    // Extract tar archive
    let mut tar_archive = tar::Archive::new(decoder);
    tar_archive
        .unpack(output_dir)
        .map_err(|e| PackError::Compression(format!("Failed to extract tar: {}", e)))?;

    debug!("tar.zst archive extracted successfully");
    Ok(())
}

/// Read the manifest from a .nail archive without full extraction.
#[instrument(skip(nail_path))]
pub fn read_manifest(nail_path: &Path) -> PackResult<NailManifest> {
    let temp_dir = tempfile::tempdir().map_err(PackError::Io)?;
    extract_tar_zst(nail_path, temp_dir.path())?;

    let manifest_path = temp_dir.path().join("manifest.json");
    if !manifest_path.exists() {
        return Err(PackError::MissingFile("manifest.json".to_string()));
    }

    let manifest: NailManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    Ok(manifest)
}

/// Read the identity from a .nail archive without full extraction.
#[instrument(skip(nail_path))]
pub fn read_identity(nail_path: &Path) -> PackResult<AgentIdentity> {
    let temp_dir = tempfile::tempdir().map_err(PackError::Io)?;
    extract_tar_zst(nail_path, temp_dir.path())?;

    let identity_path = temp_dir.path().join("identity.json");
    if !identity_path.exists() {
        return Err(PackError::MissingFile("identity.json".to_string()));
    }

    let identity: AgentIdentity = serde_json::from_str(&fs::read_to_string(&identity_path)?)?;
    Ok(identity)
}

/// Minimal tempfile module — we use a simple approach since tempfile may not be in deps.
mod tempfile {
    use std::path::{Path, PathBuf};

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    pub fn tempdir() -> Result<TempDir, std::io::Error> {
        let id = uuid::Uuid::new_v4().to_string();
        let path = std::env::temp_dir().join(format!("pincher-nail-{}", id));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash_bytes() {
        let hash1 = blake3_hash_bytes(b"hello");
        let hash2 = blake3_hash_bytes(b"hello");
        let hash3 = blake3_hash_bytes(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // BLAKE3 hex length
    }

    #[test]
    fn test_default_identity() {
        let identity = AgentIdentity::default();
        assert_eq!(identity.name, "Pinchy");
        assert_eq!(identity.preferences.preferred_shell, "bash");
    }

    #[test]
    fn test_default_config() {
        let config = AgentConfig::default();
        assert_eq!(config.resource_thresholds.ram_light, 70.0);
        assert_eq!(config.resource_thresholds.ram_critical, 85.0);
    }

    #[test]
    fn test_manifest_serialization() {
        let manifest = NailManifest {
            version: "0.1.0".to_string(),
            fingerprint: "abc123".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            reflex_count: 42,
            checksums: NailChecksums {
                reflexes_db: "hash1".to_string(),
                identity_json: "hash2".to_string(),
                config_toml: "hash3".to_string(),
            },
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let restored: NailManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.version, restored.version);
        assert_eq!(manifest.reflex_count, restored.reflex_count);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let temp_dir = std::env::temp_dir().join("pincher_pack_test");
        // Clean up from previous runs
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        std::fs::create_dir_all(&temp_dir).ok();

        // Create a simple SQLite database
        let db_path = temp_dir.join("test_reflexes.db");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS reflexes (
                    id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    action_sql TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    invoke_count INTEGER NOT NULL DEFAULT 0,
                    last_invoked TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                INSERT INTO reflexes (id, intent, action_sql, embedding, confidence)
                VALUES ('test-1', 'test.intent', 'SELECT 1', X'00000000', 0.9);",
            )
            .unwrap();
        }

        // Pack
        let nail_path = temp_dir.join("test.nail");
        pack_nail(&db_path, &nail_path).unwrap();
        assert!(nail_path.exists());

        // Unpack
        let unpack_dir = temp_dir.join("unpacked");
        unpack_nail(&nail_path, &unpack_dir).unwrap();

        // Verify files exist
        assert!(unpack_dir.join("reflexes.db").exists());
        assert!(unpack_dir.join("identity.json").exists());
        assert!(unpack_dir.join("config.toml").exists());
        assert!(unpack_dir.join("manifest.json").exists());

        // Verify the database content
        let unpacked_db = unpack_dir.join("reflexes.db");
        let conn = rusqlite::Connection::open(&unpacked_db).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM reflexes", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_verify_nail() {
        let temp_dir = std::env::temp_dir().join("pincher_verify_test");
        std::fs::create_dir_all(&temp_dir).ok();

        let db_path = temp_dir.join("verify_reflexes.db");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch("CREATE TABLE IF NOT EXISTS reflexes (id TEXT PRIMARY KEY, intent TEXT, action_sql TEXT, embedding BLOB, confidence REAL, invoke_count INTEGER, last_invoked TEXT, created_at TEXT);")
                .unwrap();
        }

        let nail_path = temp_dir.join("verify.nail");
        pack_nail(&db_path, &nail_path).unwrap();

        let valid = verify_nail(&nail_path).unwrap();
        assert!(valid);
    }
}
