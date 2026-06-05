//! Example: teach a reflex and then execute it.
//!
//! Run with: cargo run --example teach_and_do

use pincher_core::ReflexEngine;

fn main() -> anyhow::Result<()> {
    // Create an in-memory PincherOS instance for the demo.
    let mut engine = ReflexEngine::open(std::path::Path::new(":memory:"), None)?;

    // Teach a custom reflex.
    let reflex = engine.teach("show system info", "system.info")?;
    println!(
        "[TAUGHT] reflex {} — intent: \"{}\", action: \"{}\"",
        reflex.id, reflex.intent, reflex.action
    );

    // Do a command that should match.
    let execution = engine.do_command("show system info")?;
    println!(
        "[DO] confidence: {:.2}, match_type: {:?}",
        execution.confidence, execution.match_type
    );
    if let Some(reflex_id) = execution.reflex_id {
        println!("  matched reflex: {}", reflex_id);
    }
    println!(
        "  output: {}...",
        &execution.output[..execution.output.len().min(80)]
    );

    // Check status.
    let status = engine.get_status()?;
    println!(
        "[STATUS] reflexes={}, embedder_loaded={}",
        status.reflex_count, status.embedder_loaded
    );

    Ok(())
}
