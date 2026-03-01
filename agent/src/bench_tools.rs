// Benchmark, profiling, and correctness test tools for the Agent framework.

use crate::tools::{BenchmarkComparison, BenchmarkResult, FunctionProfile, ToolResult};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio::time::{timeout, Duration};

/// bench_tools configuration for external paths.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Absolute path to the benchmark binary.
    pub benchmark_bin: PathBuf,
    /// Absolute path to the data directory (base_vectors, query_vectors, ground_truth).
    pub data_dir: PathBuf,
    /// CPU core list for taskset (e.g. "0-3", "4-7", "0,2,4,6"). None means no pinning.
    pub cpu_cores: Option<String>,
}

/// Default timeout for benchmark execution (10 minutes).
const BENCHMARK_TIMEOUT_SECS: u64 = 600;

/// Default recall threshold for correctness tests.
const RECALL_THRESHOLD: f64 = 0.95;

fn anti_cheat_default_passed() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
struct AntiCheatOutput {
    #[serde(default = "anti_cheat_default_passed")]
    passed: bool,
    #[serde(default)]
    message: String,
}

impl Default for AntiCheatOutput {
    fn default() -> Self {
        Self {
            passed: true,
            message: String::new(),
        }
    }
}

/// Wrapper for the benchmark binary's JSON output: `{"benchmark": ..., "anti_cheat": ...}`.
#[derive(Debug, Deserialize)]
struct BenchmarkOutput {
    benchmark: BenchmarkResult,
    #[serde(default)]
    anti_cheat: AntiCheatOutput,
}

/// Timeout (seconds) for the server to become ready after startup.
const SERVER_READY_TIMEOUT_SECS: u64 = 30;

/// Polling interval (milliseconds) for server readiness checks.
const SERVER_POLL_INTERVAL_MS: u64 = 200;

/// Timeout (seconds) for `cargo build --release`.
const BUILD_TIMEOUT_SECS: u64 = 300;

// ─── Port Allocation ─────────────────────────────────────────────────────────

// ─── History Helpers ─────────────────────────────────────────────────────────

/// Get the next round number by scanning existing files in a directory.
fn next_round_number(dir: &Path, prefix: &str) -> u32 {
    let mut max = 0u32;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Parse "prefix_NNN.ext" �?NNN
            if name.starts_with(prefix) {
                if let Some(num_part) = name.strip_prefix(prefix).and_then(|s| s.split('.').next())
                {
                    if let Ok(n) = num_part.parse::<u32>() {
                        max = max.max(n);
                    }
                }
            }
        }
    }
    max + 1
}

/// Save a benchmark result to benchmarks/benchmark_NNN.json and return the round number.
fn save_benchmark_result(work_dir: &Path, result: &BenchmarkResult) -> u32 {
    let dir = work_dir.join("benchmarks");
    let _ = std::fs::create_dir_all(&dir);
    let round = next_round_number(&dir, "benchmark_");
    let path = dir.join(format!("benchmark_{:03}.json", round));
    if let Ok(json) = serde_json::to_string_pretty(result) {
        let _ = std::fs::write(&path, json);
    }
    round
}

/// Load the most recent benchmark result from benchmarks/ directory.
fn load_previous_benchmark(work_dir: &Path) -> Option<BenchmarkResult> {
    let dir = work_dir.join("benchmarks");
    if !dir.exists() {
        return None;
    }
    let current_max = next_round_number(&dir, "benchmark_").saturating_sub(1);
    if current_max == 0 {
        return None;
    }
    let path = dir.join(format!("benchmark_{:03}.json", current_max));
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Build a BenchmarkComparison from previous and current results.
fn build_comparison(prev: &BenchmarkResult, curr: &BenchmarkResult) -> BenchmarkComparison {
    let qps_change = if prev.qps > 0.0 {
        (curr.qps - prev.qps) / prev.qps * 100.0
    } else {
        0.0
    };
    let recall_change = if prev.recall > 0.0 {
        (curr.recall - prev.recall) / prev.recall * 100.0
    } else {
        0.0
    };
    BenchmarkComparison {
        previous_qps: prev.qps,
        qps_change_pct: (qps_change * 100.0).round() / 100.0,
        previous_recall: prev.recall,
        recall_change_pct: (recall_change * 100.0).round() / 100.0,
    }
}

fn apply_anti_cheat_guard(benchmark: &mut BenchmarkResult, anti_cheat: &AntiCheatOutput) {
    if anti_cheat.passed {
        return;
    }
    benchmark.qps = 0.0;
    benchmark.recall_passed = false;
}

/// Save profiling results (flamegraph + report) to profiling/ with round numbers.
/// Returns (round_number, flamegraph_path, report_path).
fn save_profiling_results(
    work_dir: &Path,
    flamegraph_svg: &Path,
    top_functions: &[FunctionProfile],
) -> (u32, String, String) {
    let dir = work_dir.join("profiling");
    let _ = std::fs::create_dir_all(&dir);
    let round = next_round_number(&dir, "flamegraph_");

    // Copy flamegraph SVG if it exists
    let fg_dest = dir.join(format!("flamegraph_{:03}.svg", round));
    let fg_path = if flamegraph_svg.exists() {
        let _ = std::fs::copy(flamegraph_svg, &fg_dest);
        format!("profiling/flamegraph_{:03}.svg", round)
    } else {
        String::new()
    };

    // Save text report
    let report_path = dir.join(format!("report_{:03}.txt", round));
    let mut report = String::new();
    report.push_str(&format!("Profiling Report #{:03}\n", round));
    report.push_str(&format!(
        "Date: {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    report.push_str("Top Functions:\n");
    for f in top_functions {
        report.push_str(&format!("  {:>6.2}%  {}\n", f.percentage, f.function));
    }
    let _ = std::fs::write(&report_path, &report);

    (round, fg_path, format!("profiling/report_{:03}.txt", round))
}

/// Allocate a random available port by binding to port 0 and reading the OS-assigned port.
fn allocate_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to allocate port: {}", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Failed to get local addr: {}", e))?
        .port();
    // Drop the listener so the port is free for the server to use.
    drop(listener);
    Ok(port)
}

// ─── Server Lifecycle Management ─────────────────────────────────────────────

/// Build the project with `cargo build --release`.
///
/// Tries an incremental build first. If it fails (e.g. stale/permission-broken
/// artifacts), runs `cargo clean --release` and retries once.
/// Returns `Ok(())` on success, or `Err` with the stderr content on failure or timeout.
async fn build_project(work_dir: &Path) -> Result<(), String> {
    build_project_inner(work_dir, false).await
}

/// Build with profiling-friendly settings (debug symbols, no LTO).
async fn build_project_for_profiling(work_dir: &Path) -> Result<(), String> {
    build_project_inner(work_dir, true).await
}

async fn build_project_inner(work_dir: &Path, profiling: bool) -> Result<(), String> {
    // Pre-flight check: Cargo.toml must exist
    if !work_dir.join("Cargo.toml").exists() {
        return Err(format!(
            "Cargo.toml not found in '{}'. The project directory may be corrupted.",
            work_dir.display()
        ));
    }

    let first_result = if profiling {
        try_cargo_build_profile(work_dir).await
    } else {
        try_cargo_build(work_dir).await
    };

    match first_result {
        Ok(()) => return Ok(()),
        Err(first_err) => {
            // First build failed �?clean and retry
            let _ = Command::new("cargo")
                .args(["clean", "--release"])
                .current_dir(work_dir)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .await;

            let retry_result = if profiling {
                try_cargo_build_profile(work_dir).await
            } else {
                try_cargo_build(work_dir).await
            };

            match retry_result {
                Ok(()) => Ok(()),
                Err(_retry_err) => {
                    // Return the original error �?it's usually more informative
                    Err(first_err)
                }
            }
        }
    }
}

/// Execute `cargo build --release` once with a timeout.
async fn try_cargo_build(work_dir: &Path) -> Result<(), String> {
    try_cargo_build_inner(work_dir, false).await
}

/// Build with profiling-friendly settings: debug symbols ON, LTO OFF.
/// This produces a binary where perf can resolve individual function names
/// instead of collapsing everything into the axum handler due to LTO inlining.
async fn try_cargo_build_profile(work_dir: &Path) -> Result<(), String> {
    try_cargo_build_inner(work_dir, true).await
}

async fn try_cargo_build_inner(work_dir: &Path, profiling: bool) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"])
        .current_dir(work_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    if profiling {
        // Debug symbols so perf can resolve function names.
        cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "2");
        // Disable LTO �?with LTO enabled, all functions get inlined into the
        // top-level handler and perf cannot distinguish them.
        cmd.env("CARGO_PROFILE_RELEASE_LTO", "false");
        // Use more codegen units to reduce inlining across compilation units.
        cmd.env("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "16");
    }

    let result = timeout(Duration::from_secs(BUILD_TIMEOUT_SECS), cmd.output()).await;

    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!(
                    "cargo build --release failed (exit code {}):\n{}",
                    output.status.code().unwrap_or(-1),
                    stderr
                ))
            }
        }
        Ok(Err(e)) => Err(format!("Failed to execute cargo build: {}", e)),
        Err(_) => Err(format!(
            "cargo build --release timed out after {} seconds",
            BUILD_TIMEOUT_SECS
        )),
    }
}

/// Public tool: build the project and return success/failure with error messages.
pub async fn build_project_tool(work_dir: &Path) -> ToolResult {
    match build_project(work_dir).await {
        Ok(()) => ToolResult::BuildProject {
            success: true,
            message: "Build succeeded.".to_string(),
        },
        Err(e) => ToolResult::BuildProject {
            success: false,
            message: format!("Build failed: {}", e),
        },
    }
}

/// Start the skeleton server binary as a child process.
///
/// Launches `target/release/<binary>` in `work_dir` with the `PORT` environment
/// variable set to the given port.
async fn start_server(
    work_dir: &Path,
    port: u16,
    cpu_cores: Option<&str>,
) -> Result<tokio::process::Child, String> {
    // Dynamically detect binary name from Cargo.toml
    let binary_name = detect_binary_name(work_dir)?;
    let binary = work_dir.join(format!("target/release/{}", binary_name));
    if !binary.exists() {
        return Err(format!(
            "Server binary not found at '{}'. Build the project first.",
            binary.display()
        ));
    }

    // Canonicalize to absolute path to avoid relative-path resolution issues
    let binary = binary.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve binary path '{}': {}",
            binary.display(),
            e
        )
    })?;

    let abs_work_dir = work_dir
        .canonicalize()
        .map_err(|e| format!("Failed to resolve work_dir '{}': {}", work_dir.display(), e))?;

    // Use taskset to pin the server to specific CPU cores.
    // This prevents the model's code from consuming all CPUs on the machine.
    // Example: taskset -c 0-3 <binary> pins to cores 0,1,2,3.
    if let Some(cores) = cpu_cores {
        Command::new("taskset")
            .arg("-c")
            .arg(cores)
            .arg(&binary)
            .current_dir(&abs_work_dir)
            .env("PORT", port.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start server with taskset: {}", e))
    } else {
        Command::new(&binary)
            .current_dir(&abs_work_dir)
            .env("PORT", port.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start server '{}': {}", binary.display(), e))
    }
}

/// Read the `[package] name` from Cargo.toml in `work_dir` to determine the binary name.
fn detect_binary_name(work_dir: &Path) -> Result<String, String> {
    let cargo_toml = work_dir.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!(
            "Cargo.toml not found in '{}'. Is this a valid Rust project?",
            work_dir.display()
        ));
    }
    let content = std::fs::read_to_string(&cargo_toml)
        .map_err(|e| format!("Failed to read Cargo.toml: {}", e))?;

    // Simple TOML parsing: find name = "..." in [package] section
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = trimmed.strip_prefix("name") {
                let rest = rest.trim();
                if let Some(rest) = rest.strip_prefix('=') {
                    let rest = rest.trim().trim_matches('"').trim_matches('\'');
                    if !rest.is_empty() {
                        return Ok(rest.to_string());
                    }
                }
            }
        }
    }
    Err(format!(
        "Could not find package name in '{}'",
        cargo_toml.display()
    ))
}

/// Wait for a TCP port to become reachable.
///
/// Polls with `TcpStream::connect` every `poll_interval_ms` milliseconds.
/// Returns `Ok(())` once the port accepts a connection, or `Err` if `timeout_secs` elapses.
async fn wait_for_server_ready(
    port: u16,
    timeout_secs: u64,
    poll_interval_ms: u64,
) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let addr = format!("127.0.0.1:{}", port);

    loop {
        match tokio::net::TcpStream::connect(&addr).await {
            Ok(_) => return Ok(()),
            Err(_) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(format!(
                        "Server on port {} not ready after {} seconds",
                        port, timeout_secs
                    ));
                }
                tokio::time::sleep(Duration::from_millis(poll_interval_ms)).await;
            }
        }
    }
}

/// Kill a server child process and wait for it to exit.
///
/// Attempts `child.kill()` followed by `child.wait()` to ensure cleanup.
async fn kill_server(child: &mut tokio::process::Child) {
    let _ = child.kill().await;
    let _ = child.wait().await;
}

// ─── Data File Discovery ─────────────────────────────────────────────────────

/// Discover the base vectors file, handling both single-file and sharded layouts.
///
/// - If `{data_dir}/base_vectors.json` exists, returns its path directly.
/// - If sharded files `base_vectors_N.json` exist, merges them into
///   `{work_dir}/base_vectors_merged.json` and returns that path.
fn find_base_vectors(data_dir: &Path, work_dir: &Path) -> Result<PathBuf, String> {
    // Check for single file first
    let single = data_dir.join("base_vectors.json");
    if single.exists() {
        return Ok(single);
    }

    // Look for sharded files (base_vectors_0.json, base_vectors_1.json, ...)
    let mut shards: Vec<PathBuf> = Vec::new();
    for i in 0.. {
        let shard = data_dir.join(format!("base_vectors_{}.json", i));
        if shard.exists() {
            shards.push(shard);
        } else {
            break;
        }
    }

    if shards.is_empty() {
        return Err(format!(
            "No base_vectors.json or base_vectors_N.json files found in {}",
            data_dir.display()
        ));
    }

    // Merge shards into a single file
    let merged_path = work_dir.join("base_vectors_merged.json");
    let mut all_vectors: Vec<serde_json::Value> = Vec::new();
    for shard in &shards {
        let content = std::fs::read_to_string(shard)
            .map_err(|e| format!("Failed to read {}: {}", shard.display(), e))?;
        let vectors: Vec<serde_json::Value> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", shard.display(), e))?;
        all_vectors.extend(vectors);
    }
    let merged_json = serde_json::to_string(&all_vectors)
        .map_err(|e| format!("Failed to serialize merged vectors: {}", e))?;
    std::fs::write(&merged_path, merged_json)
        .map_err(|e| format!("Failed to write merged file: {}", e))?;

    Ok(merged_path)
}

/// Run the benchmark client against the vector database service.
///
/// Launches the benchmark binary with the given concurrency and warmup settings,
/// captures its JSON stdout, and parses it into a `BenchmarkResult`.
/// Run the benchmark client against the vector database service.
///
/// Manages the full server lifecycle: kill leftover processes on the port,
/// build the project, start the server, wait for readiness, execute the
/// benchmark binary, and finally kill the server (regardless of outcome).
///
/// Uses `config` for the benchmark binary and data file paths, and `work_dir`
/// for building and running the skeleton server.
pub async fn run_benchmark(
    work_dir: &Path,
    config: &BenchConfig,
    concurrency: Option<usize>,
    warmup: Option<usize>,
    max_queries: Option<usize>,
) -> ToolResult {
    let concurrency = concurrency.unwrap_or(4);
    let warmup = warmup.unwrap_or(100);
    let max_queries = max_queries.unwrap_or(1000);

    // 1. Allocate a random port for this run
    let port = match allocate_port() {
        Ok(p) => p,
        Err(e) => return ToolResult::Error { message: e },
    };

    // 2. Build the project
    if let Err(e) = build_project(work_dir).await {
        return ToolResult::Error {
            message: format!("Build failed: {}", e),
        };
    }

    // 3. Start the server
    let mut child = match start_server(work_dir, port, config.cpu_cores.as_deref()).await {
        Ok(child) => child,
        Err(e) => {
            return ToolResult::Error {
                message: format!("Failed to start server: {}", e),
            };
        }
    };

    // 4. Wait for server readiness
    if let Err(e) =
        wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS, SERVER_POLL_INTERVAL_MS).await
    {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!("Server not ready: {}", e),
        };
    }

    // 5. Locate data files
    let base_vectors = match find_base_vectors(&config.data_dir, work_dir) {
        Ok(p) => p,
        Err(e) => {
            kill_server(&mut child).await;
            return ToolResult::Error { message: e };
        }
    };
    let query_vectors = config.data_dir.join("query_vectors.json");
    let ground_truth = config.data_dir.join("ground_truth.json");

    for (label, path) in [
        ("query vectors", &query_vectors),
        ("ground truth", &ground_truth),
    ] {
        if !path.exists() {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Data file not found: {} ({})", path.display(), label),
            };
        }
    }

    // 6. Execute the benchmark binary
    let benchmark_bin = &config.benchmark_bin;
    if !benchmark_bin.exists() {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!(
                "Benchmark binary not found at '{}'.",
                benchmark_bin.display()
            ),
        };
    }

    let result = match timeout(
        Duration::from_secs(BENCHMARK_TIMEOUT_SECS),
        Command::new(benchmark_bin.to_str().unwrap_or_default())
            .arg("--server-url")
            .arg(format!("http://127.0.0.1:{}", port))
            .arg("--concurrency")
            .arg(concurrency.to_string())
            .arg("--warmup")
            .arg(warmup.to_string())
            .arg("--max-queries")
            .arg(max_queries.to_string())
            .arg("--base-vectors")
            .arg(base_vectors.to_str().unwrap_or_default())
            .arg("--query-vectors")
            .arg(query_vectors.to_str().unwrap_or_default())
            .arg("--ground-truth")
            .arg(ground_truth.to_str().unwrap_or_default())
            .current_dir(work_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Failed to execute benchmark binary: {}", e),
            };
        }
        Err(_) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!(
                    "Benchmark timed out after {} seconds",
                    BENCHMARK_TIMEOUT_SECS
                ),
            };
        }
    };

    // 7. Always kill the server
    kill_server(&mut child).await;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return ToolResult::Error {
            message: format!(
                "Benchmark exited with code {}: {}",
                result.status.code().unwrap_or(-1),
                stderr.chars().take(2000).collect::<String>()
            ),
        };
    }

    let stdout = String::from_utf8_lossy(&result.stdout);
    match serde_json::from_str::<BenchmarkOutput>(&stdout) {
        Ok(output) => {
            let mut bench = output.benchmark;
            if !output.anti_cheat.passed {
                eprintln!(
                    "[benchmark] Anti-cheat failed. Invalidating score (QPS=0). Detail: {}",
                    if output.anti_cheat.message.is_empty() {
                        "SUSPICIOUS benchmark output"
                    } else {
                        output.anti_cheat.message.as_str()
                    }
                );
            }
            apply_anti_cheat_guard(&mut bench, &output.anti_cheat);
            // Add comparison with previous run
            if let Some(prev) = load_previous_benchmark(work_dir) {
                bench.comparison = Some(build_comparison(&prev, &bench));
            }
            // Save this result for future comparisons
            save_benchmark_result(work_dir, &bench);
            ToolResult::RunBenchmark(bench)
        }
        Err(e) => ToolResult::Error {
            message: format!(
                "Failed to parse benchmark JSON output: {}. Output: {}",
                e,
                stdout.chars().take(500).collect::<String>()
            ),
        },
    }
}

/// Run performance profiling on the skeleton server process.
///
/// **Unlike `run_benchmark` and `run_correctness_test`, this function does NOT
/// manage the server lifecycle.** The server must already be running on port 8080
/// before calling this function. Start the server manually (e.g. via `run_command`)
/// Run performance profiling on the skeleton server.
///
/// Manages the full server lifecycle: kill leftover processes on the port,
/// build the project, start the server, wait for readiness, run `perf record`,
/// generate flamegraph, extract top functions, and finally kill the server.
pub async fn run_profiling(
    work_dir: &Path,
    config: &BenchConfig,
    _duration: Option<u64>,
) -> ToolResult {
    let perf_data = work_dir.join("perf.data");
    let flamegraph_svg = work_dir.join("flamegraph.svg");

    // 1. Allocate a random port for this run
    let port = match allocate_port() {
        Ok(p) => p,
        Err(e) => return ToolResult::Error { message: e },
    };

    // 2. Build with profiling-friendly settings (debug symbols, no LTO)
    //    so perf can resolve individual function names in the flamegraph.
    if let Err(e) = build_project_for_profiling(work_dir).await {
        return ToolResult::Error {
            message: format!("Build failed: {}", e),
        };
    }

    // 3. Start the server
    let mut child = match start_server(work_dir, port, config.cpu_cores.as_deref()).await {
        Ok(child) => child,
        Err(e) => {
            return ToolResult::Error {
                message: format!("Failed to start server: {}", e),
            };
        }
    };

    // 4. Wait for server readiness
    if let Err(e) =
        wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS, SERVER_POLL_INTERVAL_MS).await
    {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!("Server not ready: {}", e),
        };
    }

    // 5. Find the server PID for perf
    let pid = match child.id() {
        Some(pid) => pid,
        None => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: "Failed to get server process ID for profiling.".to_string(),
            };
        }
    };

    // 6. Locate data files (same as run_benchmark) for real workload
    let base_vectors = match find_base_vectors(&config.data_dir, work_dir) {
        Ok(p) => p,
        Err(e) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Profiling needs real data: {}", e),
            };
        }
    };
    let query_vectors = config.data_dir.join("query_vectors.json");
    let ground_truth = config.data_dir.join("ground_truth.json");
    for (label, path) in [
        ("query vectors", &query_vectors),
        ("ground truth", &ground_truth),
    ] {
        if !path.exists() {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Data file not found: {} ({})", path.display(), label),
            };
        }
    }

    let benchmark_bin = &config.benchmark_bin;
    if !benchmark_bin.exists() {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!("Benchmark binary not found at '{}'. Profiling needs the benchmark client to generate real load.", benchmark_bin.display()),
        };
    }

    // 7. Launch perf record and benchmark client in parallel.
    //    perf records CPU samples while the benchmark client drives real traffic.

    // Spawn perf record as a child process (not .output() �?we need it running in background)
    let mut perf_child = match Command::new("perf")
        .args([
            "record",
            "-F",
            "99",
            "-p",
            &pid.to_string(),
            "-g",
            "-o",
            perf_data.to_str().unwrap_or("perf.data"),
        ])
        .current_dir(work_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Failed to start perf record: {}", e),
            };
        }
    };

    // Run benchmark client with real data (use max_queries=1000, concurrency=4)
    let bench_result = timeout(
        Duration::from_secs(BENCHMARK_TIMEOUT_SECS),
        Command::new(benchmark_bin.to_str().unwrap_or_default())
            .arg("--server-url")
            .arg(format!("http://127.0.0.1:{}", port))
            .arg("--concurrency")
            .arg("4")
            .arg("--warmup")
            .arg("100")
            .arg("--max-queries")
            .arg("1000")
            .arg("--base-vectors")
            .arg(base_vectors.to_str().unwrap_or_default())
            .arg("--query-vectors")
            .arg(query_vectors.to_str().unwrap_or_default())
            .arg("--ground-truth")
            .arg(ground_truth.to_str().unwrap_or_default())
            .current_dir(work_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output(),
    )
    .await;

    // Benchmark finished (or timed out). Stop perf by sending SIGINT.
    #[cfg(unix)]
    {
        if let Some(perf_pid) = perf_child.id() {
            // SIGINT (2) tells perf to flush and exit cleanly
            unsafe {
                libc::kill(perf_pid as i32, libc::SIGINT);
            }
        }
    }
    // Wait for perf to finish writing
    let perf_wait = timeout(Duration::from_secs(10), perf_child.wait()).await;
    match perf_wait {
        Ok(Ok(_)) => {} // perf exited
        _ => {
            let _ = perf_child.kill().await;
        } // force kill if stuck
    }

    // Log benchmark result (informational, not the main output)
    match bench_result {
        Ok(Ok(output)) => {
            if output.status.success() {
                eprintln!("[profiling] Benchmark client completed successfully during profiling.");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!(
                    "[profiling] Benchmark client exited with code {}: {}",
                    output.status.code().unwrap_or(-1),
                    &stderr[..stderr.len().min(500)]
                );
            }
        }
        Ok(Err(e)) => eprintln!("[profiling] Benchmark client failed to execute: {}", e),
        Err(_) => eprintln!(
            "[profiling] Benchmark client timed out (profiling data should still be valid)."
        ),
    }

    // Check perf.data was produced
    if !perf_data.exists() {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: "perf record produced no data file.".to_string(),
        };
    }

    // 8. Kill the server (profiling is done)
    kill_server(&mut child).await;

    // 8. Generate flamegraph SVG
    let flamegraph_result = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "perf script -i {} | stackcollapse-perf.pl | flamegraph.pl > {}",
            perf_data.display(),
            flamegraph_svg.display()
        ))
        .current_dir(work_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await;

    let flamegraph_path = if let Ok(output) = flamegraph_result {
        if output.status.success() && flamegraph_svg.exists() {
            flamegraph_svg.to_string_lossy().to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // 9. Extract top functions from perf report
    let top_functions = extract_top_functions(work_dir, &perf_data).await;
    let total_samples = top_functions
        .iter()
        .map(|f| (f.percentage * 100.0) as u64)
        .sum::<u64>()
        .max(1);

    // 10. Save profiling results to profiling/ directory with round numbers
    let (_round, saved_fg_path, _report_path) =
        save_profiling_results(work_dir, &flamegraph_svg, &top_functions);

    // Use the saved path (relative to work_dir) if flamegraph was generated
    let final_fg_path = if !saved_fg_path.is_empty() {
        saved_fg_path
    } else {
        flamegraph_path
    };

    ToolResult::RunProfiling {
        flamegraph_svg_path: final_fg_path,
        top_functions,
        total_samples,
    }
}

/// Extract top functions from perf report output.
async fn extract_top_functions(base_dir: &Path, perf_data: &Path) -> Vec<FunctionProfile> {
    let output = Command::new("perf")
        .args([
            "report",
            "-i",
            perf_data.to_str().unwrap_or("perf.data"),
            "--stdio",
            "--no-children",
            "-n",
            "--percent-limit",
            "1.0",
        ])
        .current_dir(base_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_perf_report(&stdout)
}

/// Parse perf report --stdio output to extract function names and percentages.
///
/// Lines look like: "  42.50%  1234  binary_name  [.] function_name"
fn parse_perf_report(report: &str) -> Vec<FunctionProfile> {
    let mut functions = Vec::new();

    for line in report.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Look for lines starting with a percentage
        if let Some(pct_end) = trimmed.find('%') {
            let pct_str = trimmed[..pct_end].trim();
            if let Ok(percentage) = pct_str.parse::<f64>() {
                // Extract function name after [.] or [k] marker
                if let Some(bracket_pos) = trimmed.find("[.] ").or_else(|| trimmed.find("[k] ")) {
                    let func_name = trimmed[bracket_pos + 4..].trim().to_string();
                    if !func_name.is_empty() {
                        functions.push(FunctionProfile {
                            function: func_name,
                            percentage,
                        });
                    }
                }
            }
        }
    }

    // Sort by percentage descending, take top 10
    functions.sort_by(|a, b| {
        b.percentage
            .partial_cmp(&a.percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    functions.truncate(10);
    functions
}

/// Run a correctness test by executing the benchmark with a small query subset.
///
/// Runs the benchmark client in a lightweight mode and checks whether the recall
/// meets the threshold (0.95). Returns pass/fail status and details about any
/// failing queries.
/// Run a correctness test by executing the benchmark with conservative settings.
///
/// Manages the full server lifecycle (same as `run_benchmark`): kill leftover
/// processes, build, start, wait, execute, kill. Runs the benchmark binary with
/// `--concurrency 1 --warmup 100 --recall-threshold 0.95` and checks whether
/// the recall meets the threshold.
pub async fn run_correctness_test(work_dir: &Path, config: &BenchConfig) -> ToolResult {
    // 1. Allocate a random port for this run
    let port = match allocate_port() {
        Ok(p) => p,
        Err(e) => return ToolResult::Error { message: e },
    };

    // 2. Build the project
    if let Err(e) = build_project(work_dir).await {
        return ToolResult::Error {
            message: format!("Build failed: {}", e),
        };
    }

    // 3. Start the server
    let mut child = match start_server(work_dir, port, config.cpu_cores.as_deref()).await {
        Ok(child) => child,
        Err(e) => {
            return ToolResult::Error {
                message: format!("Failed to start server: {}", e),
            };
        }
    };

    // 4. Wait for server readiness
    if let Err(e) =
        wait_for_server_ready(port, SERVER_READY_TIMEOUT_SECS, SERVER_POLL_INTERVAL_MS).await
    {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!("Server not ready: {}", e),
        };
    }

    // 5. Locate data files
    let base_vectors = match find_base_vectors(&config.data_dir, work_dir) {
        Ok(p) => p,
        Err(e) => {
            kill_server(&mut child).await;
            return ToolResult::Error { message: e };
        }
    };
    let query_vectors = config.data_dir.join("query_vectors.json");
    let ground_truth = config.data_dir.join("ground_truth.json");

    for (label, path) in [
        ("query vectors", &query_vectors),
        ("ground truth", &ground_truth),
    ] {
        if !path.exists() {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Data file not found: {} ({})", path.display(), label),
            };
        }
    }

    // 6. Execute the benchmark binary in correctness mode
    let benchmark_bin = &config.benchmark_bin;
    if !benchmark_bin.exists() {
        kill_server(&mut child).await;
        return ToolResult::Error {
            message: format!(
                "Benchmark binary not found at '{}'.",
                benchmark_bin.display()
            ),
        };
    }

    let result = match timeout(
        Duration::from_secs(BENCHMARK_TIMEOUT_SECS),
        Command::new(benchmark_bin.to_str().unwrap_or_default())
            .arg("--server-url")
            .arg(format!("http://127.0.0.1:{}", port))
            .arg("--concurrency")
            .arg("1")
            .arg("--warmup")
            .arg("0")
            .arg("--max-queries")
            .arg("100")
            .arg("--base-vectors")
            .arg(base_vectors.to_str().unwrap_or_default())
            .arg("--query-vectors")
            .arg(query_vectors.to_str().unwrap_or_default())
            .arg("--ground-truth")
            .arg(ground_truth.to_str().unwrap_or_default())
            .arg("--recall-threshold")
            .arg(RECALL_THRESHOLD.to_string())
            .current_dir(work_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!("Failed to execute correctness test: {}", e),
            };
        }
        Err(_) => {
            kill_server(&mut child).await;
            return ToolResult::Error {
                message: format!(
                    "Correctness test timed out after {} seconds",
                    BENCHMARK_TIMEOUT_SECS
                ),
            };
        }
    };

    // 7. Always kill the server
    kill_server(&mut child).await;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return ToolResult::Error {
            message: format!(
                "Correctness test exited with code {}: {}",
                result.status.code().unwrap_or(-1),
                stderr.chars().take(2000).collect::<String>()
            ),
        };
    }

    let stdout = String::from_utf8_lossy(&result.stdout);
    match serde_json::from_str::<BenchmarkOutput>(&stdout) {
        Ok(output) => {
            let bench = output.benchmark;
            let passed = bench.recall >= RECALL_THRESHOLD;
            let message = if passed {
                format!(
                    "Correctness test PASSED: recall {:.4} >= threshold {:.4}",
                    bench.recall, RECALL_THRESHOLD
                )
            } else {
                format!(
                    "Correctness test FAILED: recall {:.4} < threshold {:.4}",
                    bench.recall, RECALL_THRESHOLD
                )
            };

            ToolResult::RunCorrectnessTest {
                passed,
                total_queries: bench.total_queries,
                recall: bench.recall,
                recall_threshold: RECALL_THRESHOLD,
                failed_queries: Vec::new(),
                message,
            }
        }
        Err(e) => ToolResult::Error {
            message: format!(
                "Failed to parse correctness test output: {}. Output: {}",
                e,
                stdout.chars().take(500).collect::<String>()
            ),
        },
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── parse_perf_report tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_perf_report_basic() {
        let report = r#"
# Overhead  Samples  Command          Shared Object        Symbol
# ........  .......  ...............  ...................  ......
#
    42.50%     1234  skeleton         [.] l2_distance
    25.30%      800  skeleton         [.] search
    10.00%      300  skeleton         [.] insert
     5.20%      150  libc.so          [.] memcpy
"#;
        let functions = parse_perf_report(report);
        assert_eq!(functions.len(), 4);
        assert_eq!(functions[0].function, "l2_distance");
        assert!((functions[0].percentage - 42.50).abs() < f64::EPSILON);
        assert_eq!(functions[1].function, "search");
        assert!((functions[1].percentage - 25.30).abs() < f64::EPSILON);
        assert_eq!(functions[2].function, "insert");
        assert_eq!(functions[3].function, "memcpy");
    }

    #[test]
    fn test_parse_perf_report_empty() {
        let functions = parse_perf_report("");
        assert!(functions.is_empty());
    }

    #[test]
    fn test_parse_perf_report_comments_only() {
        let report = "# This is a comment\n# Another comment\n";
        let functions = parse_perf_report(report);
        assert!(functions.is_empty());
    }

    #[test]
    fn test_parse_perf_report_kernel_symbols() {
        let report = r#"
    30.00%      500  skeleton         [k] __do_page_fault
    20.00%      300  skeleton         [.] user_function
"#;
        let functions = parse_perf_report(report);
        assert_eq!(functions.len(), 2);
        assert_eq!(functions[0].function, "__do_page_fault");
        assert_eq!(functions[1].function, "user_function");
    }

    #[test]
    fn test_parse_perf_report_truncates_to_10() {
        let mut report = String::new();
        for i in 0..15 {
            report.push_str(&format!(
                "    {:.2}%      100  skeleton         [.] func_{}\n",
                50.0 - i as f64,
                i
            ));
        }
        let functions = parse_perf_report(&report);
        assert_eq!(functions.len(), 10);
        // Should be sorted by percentage descending
        assert_eq!(functions[0].function, "func_0");
        assert_eq!(functions[9].function, "func_9");
    }

    #[test]
    fn test_parse_perf_report_sorted_descending() {
        let report = r#"
     5.00%      100  skeleton         [.] low_func
    50.00%      500  skeleton         [.] high_func
    20.00%      200  skeleton         [.] mid_func
"#;
        let functions = parse_perf_report(report);
        assert_eq!(functions.len(), 3);
        assert_eq!(functions[0].function, "high_func");
        assert_eq!(functions[1].function, "mid_func");
        assert_eq!(functions[2].function, "low_func");
    }

    // ─── run_benchmark tests ────────────────────────────────────────────────
    //
    // Note: run_benchmark now manages the full server lifecycle (build �?start
    // �?wait �?benchmark �?kill). In a temp dir with no Cargo project, the
    // build step will fail first, so we test that the build-failure path
    // returns an error.

    #[tokio::test]
    async fn test_run_benchmark_build_fails_in_empty_dir() {
        let dir = tempdir();
        let config = BenchConfig {
            benchmark_bin: dir.join("nonexistent_binary"),
            data_dir: dir.join("nonexistent_data"),
            cpu_cores: None,
        };
        match run_benchmark(&dir, &config, None, None, None).await {
            ToolResult::Error { message } => {
                assert!(
                    message.contains("Build failed"),
                    "Expected build failure message, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    // ─── run_correctness_test tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_run_correctness_test_build_fails_in_empty_dir() {
        let dir = tempdir();
        let config = BenchConfig {
            benchmark_bin: dir.join("nonexistent_binary"),
            data_dir: dir.join("nonexistent_data"),
            cpu_cores: None,
        };
        match run_correctness_test(&dir, &config).await {
            ToolResult::Error { message } => {
                assert!(
                    message.contains("Build failed"),
                    "Expected build failure message, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    // ─── run_profiling tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_profiling_build_fails_in_empty_dir() {
        let dir = tempdir();
        let config = BenchConfig {
            benchmark_bin: dir.join("nonexistent_binary"),
            data_dir: dir.join("nonexistent_data"),
            cpu_cores: None,
        };
        match run_profiling(&dir, &config, Some(1)).await {
            ToolResult::Error { message } => {
                assert!(
                    message.contains("Build failed"),
                    "Expected build failure message, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for build failure, got {:?}", other),
        }
    }

    // ─── BenchmarkResult JSON parsing test ───────────────────────────────────

    #[test]
    fn test_benchmark_result_json_parsing() {
        // Test parsing the wrapped format that the real benchmark binary outputs
        let json = r#"{
            "benchmark": {
                "qps": 1500.5,
                "total_queries": 10000,
                "duration_secs": 6.66,
                "avg_latency_ms": 2.5,
                "p50_latency_ms": 2.0,
                "p95_latency_ms": 5.0,
                "p99_latency_ms": 10.0,
                "recall": 0.98,
                "recall_threshold": 0.95,
                "recall_passed": true,
                "concurrency": 4
            },
            "anti_cheat": {
                "passed": true,
                "message": "OK"
            }
        }"#;
        let output: BenchmarkOutput = serde_json::from_str(json).unwrap();
        assert!((output.benchmark.qps - 1500.5).abs() < f64::EPSILON);
        assert_eq!(output.benchmark.total_queries, 10000);
        assert!(output.benchmark.recall_passed);
        assert!(output.anti_cheat.passed);
    }

    #[test]
    fn test_apply_anti_cheat_guard_invalidates_score() {
        let mut benchmark = BenchmarkResult {
            qps: 1500.5,
            total_queries: 10000,
            duration_secs: 6.66,
            avg_latency_ms: 2.5,
            p50_latency_ms: 2.0,
            p95_latency_ms: 5.0,
            p99_latency_ms: 10.0,
            recall: 0.98,
            recall_threshold: 0.95,
            recall_passed: true,
            concurrency: 4,
            comparison: None,
        };
        let anti_cheat = AntiCheatOutput {
            passed: false,
            message: "SUSPICIOUS".to_string(),
        };

        apply_anti_cheat_guard(&mut benchmark, &anti_cheat);

        assert_eq!(benchmark.qps, 0.0);
        assert!(!benchmark.recall_passed);
    }

    #[test]
    fn test_apply_anti_cheat_guard_keeps_clean_result() {
        let mut benchmark = BenchmarkResult {
            qps: 1500.5,
            total_queries: 10000,
            duration_secs: 6.66,
            avg_latency_ms: 2.5,
            p50_latency_ms: 2.0,
            p95_latency_ms: 5.0,
            p99_latency_ms: 10.0,
            recall: 0.98,
            recall_threshold: 0.95,
            recall_passed: true,
            concurrency: 4,
            comparison: None,
        };
        let anti_cheat = AntiCheatOutput {
            passed: true,
            message: "OK".to_string(),
        };

        apply_anti_cheat_guard(&mut benchmark, &anti_cheat);

        assert_eq!(benchmark.qps, 1500.5);
        assert!(benchmark.recall_passed);
    }

    // ─── detect_binary_name tests ───────────────────────────────────────────

    #[test]
    fn test_detect_binary_name_standard() {
        let dir = tempdir();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"my-cool-server\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        assert_eq!(detect_binary_name(&dir).unwrap(), "my-cool-server");
    }

    #[test]
    fn test_detect_binary_name_missing_cargo_toml() {
        let dir = tempdir();
        assert!(detect_binary_name(&dir).is_err());
    }

    #[test]
    fn test_detect_binary_name_no_package_name() {
        let dir = tempdir();
        std::fs::write(dir.join("Cargo.toml"), "[dependencies]\nserde = \"1\"\n").unwrap();
        assert!(detect_binary_name(&dir).is_err());
    }

    // ─── mock integration tests ─────────────────────────────────────────────
    //
    // These tests create a real minimal Cargo project (tiny TCP server) and a
    // mock benchmark binary, then exercise the full run_benchmark /
    // run_correctness_test flow.  They are marked #[ignore] because they invoke
    // `cargo build --release` and `rustc`, which takes time.
    //
    // Run with: cargo test -- --ignored

    /// Scaffold a minimal Cargo project whose binary listens on the PORT env var.
    fn create_mock_server_project(dir: &Path) {
        // Cargo.toml
        std::fs::write(
            dir.join("Cargo.toml"),
            r#"[package]
name = "mock-server"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
        )
        .unwrap();

        // src/main.rs �?bind to PORT env var, accept connections forever
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("main.rs"),
            r#"use std::net::TcpListener;
fn main() {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
    // Signal readiness by accepting connections
    for stream in listener.incoming() {
        let _ = stream;
        // Keep running �?the test will kill us
    }
}
"#,
        )
        .unwrap();
    }

    /// Compile a tiny Rust program with `rustc` that prints fixed benchmark
    /// JSON to stdout and exits 0.  Returns the path to the compiled binary.
    fn create_mock_benchmark_binary(dir: &Path) -> PathBuf {
        let src_path = dir.join("mock_bench.rs");
        std::fs::write(
            &src_path,
            r#"fn main() {
    let json = r#_#_{
        "benchmark": {
            "qps": 1234.5,
            "total_queries": 500,
            "duration_secs": 0.4,
            "avg_latency_ms": 0.8,
            "p50_latency_ms": 0.7,
            "p95_latency_ms": 1.2,
            "p99_latency_ms": 2.0,
            "recall": 0.99,
            "recall_threshold": 0.95,
            "recall_passed": true,
            "concurrency": 1
        },
        "anti_cheat": {
            "passed": true,
            "message": "OK"
        }
    }_#_#;
    println!("{}", json);
}
"#
            .replace("_#_", "#"),
        )
        .unwrap();

        let bin_name = if cfg!(windows) {
            "mock_bench.exe"
        } else {
            "mock_bench"
        };
        let bin_path = dir.join(bin_name);

        let output = std::process::Command::new("rustc")
            .arg(&src_path)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .expect("rustc must be available to run mock tests");
        assert!(
            output.status.success(),
            "rustc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        bin_path
    }

    /// Create minimal stub data files that the benchmark flow expects.
    fn create_mock_data(dir: &Path) {
        std::fs::write(dir.join("base_vectors.json"), "[]").unwrap();
        std::fs::write(dir.join("query_vectors.json"), "[]").unwrap();
        std::fs::write(dir.join("ground_truth.json"), "[]").unwrap();
    }

    #[tokio::test]
    #[ignore] // requires cargo + rustc, run with: cargo test -- --ignored
    async fn test_run_benchmark_mock_full_flow() {
        let work_dir = tempdir();
        let data_dir = tempdir();
        let bench_dir = tempdir();

        create_mock_server_project(&work_dir);
        create_mock_data(&data_dir);
        let benchmark_bin = create_mock_benchmark_binary(&bench_dir);

        let config = BenchConfig {
            benchmark_bin,
            data_dir,
            cpu_cores: None,
        };

        let result = run_benchmark(&work_dir, &config, Some(1), Some(0), Some(0)).await;

        match result {
            ToolResult::RunBenchmark(bench) => {
                assert!((bench.qps - 1234.5).abs() < f64::EPSILON);
                assert_eq!(bench.total_queries, 500);
                assert!(bench.recall_passed);
                assert!((bench.recall - 0.99).abs() < f64::EPSILON);
            }
            ToolResult::Error { message } => {
                panic!("Expected RunBenchmark, got Error: {}", message);
            }
            other => panic!("Expected RunBenchmark, got {:?}", other),
        }
    }

    #[tokio::test]
    #[ignore] // requires cargo + rustc, run with: cargo test -- --ignored
    async fn test_run_correctness_test_mock_full_flow() {
        let work_dir = tempdir();
        let data_dir = tempdir();
        let bench_dir = tempdir();

        create_mock_server_project(&work_dir);
        create_mock_data(&data_dir);
        let benchmark_bin = create_mock_benchmark_binary(&bench_dir);

        let config = BenchConfig {
            benchmark_bin,
            data_dir,
            cpu_cores: None,
        };

        let result = run_correctness_test(&work_dir, &config).await;

        match result {
            ToolResult::RunCorrectnessTest {
                passed,
                recall,
                recall_threshold,
                total_queries,
                message,
                ..
            } => {
                assert!(passed, "Correctness test should pass: {}", message);
                assert!((recall - 0.99).abs() < f64::EPSILON);
                assert!((recall_threshold - RECALL_THRESHOLD).abs() < f64::EPSILON);
                assert_eq!(total_queries, 500);
            }
            ToolResult::Error { message } => {
                panic!("Expected RunCorrectnessTest, got Error: {}", message);
            }
            other => panic!("Expected RunCorrectnessTest, got {:?}", other),
        }
    }

    #[tokio::test]
    #[ignore] // requires cargo + rustc, run with: cargo test -- --ignored
    async fn test_run_benchmark_mock_server_binary_not_found() {
        // Server project exists but benchmark binary doesn't
        let work_dir = tempdir();
        let data_dir = tempdir();

        create_mock_server_project(&work_dir);
        create_mock_data(&data_dir);

        let config = BenchConfig {
            benchmark_bin: PathBuf::from("/nonexistent/benchmark"),
            data_dir,
            cpu_cores: None,
        };

        let result = run_benchmark(&work_dir, &config, Some(1), Some(0), Some(0)).await;

        match result {
            ToolResult::Error { message } => {
                assert!(
                    message.contains("Benchmark binary not found"),
                    "Expected benchmark-not-found error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[tokio::test]
    #[ignore] // requires cargo + rustc, run with: cargo test -- --ignored
    async fn test_run_benchmark_mock_missing_data_files() {
        // Server project exists, benchmark exists, but data dir is empty
        let work_dir = tempdir();
        let data_dir = tempdir(); // empty �?no data files
        let bench_dir = tempdir();

        create_mock_server_project(&work_dir);
        let benchmark_bin = create_mock_benchmark_binary(&bench_dir);

        let config = BenchConfig {
            benchmark_bin,
            data_dir,
            cpu_cores: None,
        };

        let result = run_benchmark(&work_dir, &config, Some(1), Some(0), Some(0)).await;

        match result {
            ToolResult::Error { message } => {
                assert!(
                    message.contains("base_vectors") || message.contains("not found"),
                    "Expected data-not-found error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    // ─── helper ──────────────────────────────────────────────────────────────

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bench_tools_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
