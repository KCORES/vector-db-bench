mod bench_tools;
mod evaluator;
mod logger;
mod sandbox;
mod state;
mod tools;

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use state::AgentState;
use logger::AgentLogger;
use tools::{dispatch_tool_call, get_tool_definitions, BenchmarkResult, ToolCall, ToolResult};

/// Vector DB Agent - Tool Call Agent for LLM evaluation
#[derive(Parser, Debug)]
#[command(name = "vector-db-agent")]
struct Args {
    /// LLM API endpoint URL (OpenAI compatible)
    #[arg(long, required_unless_present = "validate_perf")]
    api_url: Option<String>,

    /// API key for authentication
    #[arg(long, required_unless_present = "validate_perf")]
    api_key: Option<String>,

    /// Model name to use
    #[arg(long, required_unless_present = "validate_perf")]
    model: Option<String>,

    /// Path to system prompt file
    #[arg(long, required_unless_present = "validate_perf")]
    system_prompt: Option<String>,

    /// Working directory (skeleton project path)
    #[arg(long)]
    work_dir: String,

    /// Enable model thinking/reasoning mode.
    /// Values: "false" (default), "true"/"openai", "kimi", "gemini"
    #[arg(long, default_value = "false")]
    thinking_mode: String,

    /// Minimum interval (milliseconds) between LLM API calls to avoid rate limiting.
    /// 0 means no delay (default: 0).
    #[arg(long, default_value = "0")]
    api_interval_ms: u64,

    /// Path to data directory containing base_vectors, query_vectors, ground_truth
    #[arg(long)]
    data_dir: Option<String>,

    /// Path to benchmark binary
    #[arg(long)]
    benchmark_bin: Option<String>,

    /// Validate perf profiling: build, start server, run perf record, report results, then exit.
    /// Does not require LLM API credentials.
    #[arg(long, default_value_t = false)]
    validate_perf: bool,

    /// Pin the server process to specific CPU cores via taskset.
    /// Accepts taskset -c format: "0-3", "4-7", "0,2,4,6", etc.
    /// Prevents the model's code from consuming all CPUs on the machine.
    /// Default: "0-3" (4 cores). Set to empty string "" to disable.
    #[arg(long, default_value = "0-3")]
    cpu_cores: String,

    /// Maximum number of tool calls allowed per session.
    /// Default: 50. The model auto-finishes when this limit is reached.
    #[arg(long, default_value = "50")]
    max_tool_calls: u32,

    /// Enable debug output (e.g. dump full LLM request body to stderr).
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Resume from a previously saved session context.
    /// If a session_context.json exists in work_dir and the session has not
    /// reached max_tool_calls, the agent will restore messages and state
    /// from that file and continue where it left off.
    #[arg(long, default_value_t = false)]
    resume: bool,
}

// ─── OpenAI API types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Reasoning/thinking content — must be echoed back for Kimi thinking mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCallMessage {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<serde_json::Value>,
    tool_choice: String,
    #[serde(flatten)]
    extra_body: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallMessage>>,
    /// Reasoning/thinking content (OpenAI: reasoning_content, Kimi: reasoning_content)
    #[serde(default)]
    reasoning_content: Option<String>,
}

// ─── LLM API client ─────────────────────────────────────────────────────────

/// Maximum number of retries for LLM API calls.
const MAX_RETRIES: u32 = 5;

/// Build extra body fields based on thinking mode setting.
fn build_thinking_extra(thinking_mode: &str) -> std::collections::HashMap<String, serde_json::Value> {
    let mut extra = std::collections::HashMap::new();
    match thinking_mode {
        "true" | "openai" => {
            extra.insert(
                "thinking".to_string(),
                serde_json::json!({"type": "enabled"}),
            );
        }
        "kimi" => {
            extra.insert("enable_thinking".to_string(), serde_json::json!(true));
        }
        "gemini" => {
            extra.insert(
                "reasoning".to_string(),
                serde_json::json!({"enabled": true}),
            );
        }
        _ => {} // "false" or anything else — no extra fields
    }
    extra
}

/// Call the LLM API with retry logic and exponential backoff.
/// `api_interval_ms` is used as the minimum retry backoff for 429 errors.
async fn call_llm(
    client: &reqwest::Client,
    api_url: &str,
    api_key: &str,
    model: &str,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    thinking_mode: &str,
    api_interval_ms: u64,
    debug: bool,
) -> Result<ChatResponseMessage, String> {
    let url = format!("{}/chat/completions", api_url.trim_end_matches('/'));
    let request_body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        tools: tools.to_vec(),
        tool_choice: "auto".to_string(),
        extra_body: build_thinking_extra(thinking_mode),
    };

    if debug {
        match serde_json::to_string_pretty(&request_body) {
            Ok(json_str) => {
                eprintln!("[agent][DEBUG] === LLM REQUEST BODY START ===");
                eprintln!("{}", json_str);
                eprintln!("[agent][DEBUG] === LLM REQUEST BODY END ===");
            }
            Err(e) => {
                eprintln!("[agent][DEBUG] Failed to serialize request body: {}", e);
            }
        }
    }

    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            // Use the larger of exponential backoff or api_interval_ms
            let exp_backoff_ms = 2u64.pow(attempt) * 1000;
            let backoff_ms = exp_backoff_ms.max(api_interval_ms);
            let backoff = std::time::Duration::from_millis(backoff_ms);
            eprintln!(
                "[agent] Retry attempt {}/{} after {}ms",
                attempt + 1,
                MAX_RETRIES,
                backoff_ms,
            );
            tokio::time::sleep(backoff).await;
        }

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    eprintln!(
                        "[agent] LLM API error (status {}): {}",
                        status,
                        &body[..body.len().min(2000)]
                    );
                    if attempt < MAX_RETRIES - 1 {
                        continue;
                    }
                    return Err(format!("LLM API error (status {}): {}", status, body));
                }

                match resp.json::<ChatResponse>().await {
                    Ok(chat_resp) => {
                        if let Some(choice) = chat_resp.choices.into_iter().next() {
                            return Ok(choice.message);
                        }
                        return Err("LLM API returned empty choices".to_string());
                    }
                    Err(e) => {
                        eprintln!("[agent] Failed to parse LLM response: {}", e);
                        if attempt < MAX_RETRIES - 1 {
                            continue;
                        }
                        return Err(format!("Failed to parse LLM response: {}", e));
                    }
                }
            }
            Err(e) => {
                eprintln!("[agent] LLM API request failed: {}", e);
                if attempt < MAX_RETRIES - 1 {
                    continue;
                }
                return Err(format!("LLM API request failed after {} retries: {}", MAX_RETRIES, e));
            }
        }
    }

    Err("LLM API call failed after all retries".to_string())
}

// ─── Tool call parsing ──────────────────────────────────────────────────────

/// Parse a ToolCall from the function name and JSON arguments string.
fn parse_tool_call(name: &str, arguments: &str) -> Result<ToolCall, String> {
    match name {
        "read_file" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::ReadFile {
                path: v["path"]
                    .as_str()
                    .ok_or("Missing 'path' field")?
                    .to_string(),
            })
        }
        "write_file" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::WriteFile {
                path: v["path"]
                    .as_str()
                    .ok_or("Missing 'path' field")?
                    .to_string(),
                content: v["content"]
                    .as_str()
                    .ok_or("Missing 'content' field")?
                    .to_string(),
            })
        }
        "list_files" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::ListFiles {
                path: v["path"]
                    .as_str()
                    .ok_or("Missing 'path' field")?
                    .to_string(),
            })
        }
        "run_benchmark" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::RunBenchmark {
                concurrency: v["concurrency"].as_u64().map(|n| n as usize),
                warmup: v["warmup"].as_u64().map(|n| n as usize),
                max_queries: v["max_queries"].as_u64().map(|n| n as usize),
            })
        }
        "run_profiling" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::RunProfiling {
                duration: v["duration"].as_u64(),
            })
        }
        "run_correctness_test" => Ok(ToolCall::RunCorrectnessTest),
        "build_project" => Ok(ToolCall::BuildProject),
        "get_status" => Ok(ToolCall::GetStatus),
        "finish" => {
            let v: serde_json::Value =
                serde_json::from_str(arguments).map_err(|e| format!("Invalid JSON: {}", e))?;
            Ok(ToolCall::Finish {
                summary: v["summary"]
                    .as_str()
                    .ok_or("Missing 'summary' field")?
                    .to_string(),
            })
        }
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

// ─── Message builders ────────────────────────────────────────────────────────

/// Build a system message.
fn build_system_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "system".to_string(),
        content: Some(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// Build an assistant message with tool calls (from LLM response).
fn build_assistant_tool_calls_message(tool_calls: Vec<ToolCallMessage>, reasoning_content: Option<String>) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(tool_calls),
        tool_call_id: None,
        reasoning_content,
    }
}

/// Build an assistant message with text content (no tool calls).
fn build_assistant_content_message(content: &str, reasoning_content: Option<String>) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: Some(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content,
    }
}

/// Build a tool result message.
fn build_tool_result_message(tool_call_id: &str, result: &ToolResult) -> ChatMessage {
    let content = serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string());
    ChatMessage {
        role: "tool".to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: Some(tool_call_id.to_string()),
        reasoning_content: None,
    }
}

// ─── Eval log ────────────────────────────────────────────────────────────────

/// Save the evaluation log to a JSON file.
fn save_eval_log(work_dir: &Path, state: &AgentState) {
    let log = serde_json::json!({
        "tool_calls_used": state.tool_calls_used,
        "tool_calls_total": state.tool_calls_total,
        "call_log": state.call_log,
        "last_benchmark": state.last_benchmark,
        "best_benchmark": state.best_benchmark,
    });

    let log_path = work_dir.join("eval_log.json");
    match std::fs::write(&log_path, serde_json::to_string_pretty(&log).unwrap_or_default()) {
        Ok(()) => eprintln!("[agent] Eval log saved to {}", log_path.display()),
        Err(e) => eprintln!("[agent] Failed to save eval log: {}", e),
    }
}

// ─── Session context persistence ─────────────────────────────────────────────

const SESSION_CONTEXT_FILE: &str = "session_context.json";

/// Persisted session context for crash recovery / resume.
#[derive(Debug, Serialize, Deserialize)]
struct SessionContext {
    /// Metadata
    tool_calls_used: u32,
    tool_calls_total: u32,
    /// Conversation messages (the full LLM context window)
    messages: Vec<ChatMessage>,
    /// Best benchmark seen so far
    #[serde(default)]
    last_benchmark: Option<BenchmarkResult>,
    #[serde(default)]
    best_benchmark: Option<BenchmarkResult>,
    /// Per-call log entries (mirrors AgentState.call_log)
    #[serde(default)]
    call_log: Vec<state::ToolCallLog>,
}

/// Save session context to `<work_dir>/session_context.json`.
fn save_session_context(
    work_dir: &Path,
    messages: &[ChatMessage],
    state: &AgentState,
) {
    let ctx = SessionContext {
        tool_calls_used: state.tool_calls_used,
        tool_calls_total: state.tool_calls_total,
        messages: messages.to_vec(),
        last_benchmark: state.last_benchmark.clone(),
        best_benchmark: state.best_benchmark.clone(),
        call_log: state.call_log.clone(),
    };
    let path = work_dir.join(SESSION_CONTEXT_FILE);
    match serde_json::to_string(&ctx) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("[agent] Failed to save session context: {}", e);
            }
        }
        Err(e) => eprintln!("[agent] Failed to serialize session context: {}", e),
    }
}

/// Try to load a previously saved session context.
/// Returns `None` if the file doesn't exist or can't be parsed.
fn load_session_context(work_dir: &Path) -> Option<SessionContext> {
    let path = work_dir.join(SESSION_CONTEXT_FILE);
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str::<SessionContext>(&data) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("[agent] Failed to parse session context: {}", e);
                None
            }
        },
        Err(e) => {
            eprintln!("[agent] Failed to read session context: {}", e);
            None
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let work_dir = PathBuf::from(&args.work_dir);

    // Canonicalize work_dir to absolute path
    let work_dir = work_dir.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "[agent] ERROR: Cannot resolve work_dir '{}': {}",
            args.work_dir, e
        );
        std::process::exit(1);
    });

    // Resolve data_dir and benchmark_bin with defaults
    let data_dir = match &args.data_dir {
        Some(d) => PathBuf::from(d),
        None => work_dir.join("data"),
    };
    let benchmark_bin = match &args.benchmark_bin {
        Some(b) => PathBuf::from(b),
        None => work_dir.join("benchmark/target/release/vector-db-benchmark"),
    };

    // Validate paths exist
    if !data_dir.exists() {
        eprintln!(
            "[agent] ERROR: Data directory not found: {}. Use --data-dir to specify the correct path.",
            data_dir.display()
        );
        std::process::exit(1);
    }
    if !benchmark_bin.exists() {
        eprintln!(
            "[agent] ERROR: Benchmark binary not found: {}. Use --benchmark-bin to specify the correct path.",
            benchmark_bin.display()
        );
        std::process::exit(1);
    }

    // Canonicalize all paths to absolute
    let data_dir = data_dir.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "[agent] ERROR: Cannot resolve data_dir: {}",
            e
        );
        std::process::exit(1);
    });
    let benchmark_bin = benchmark_bin.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "[agent] ERROR: Cannot resolve benchmark_bin: {}",
            e
        );
        std::process::exit(1);
    });

    let bench_config = bench_tools::BenchConfig {
        benchmark_bin: benchmark_bin.clone(),
        data_dir: data_dir.clone(),
        cpu_cores: if args.cpu_cores.is_empty() { None } else { Some(args.cpu_cores.clone()) },
    };

    // ── validate-perf mode: build, start server, run perf, report, exit ──
    if args.validate_perf {
        eprintln!("[validate-perf] Starting perf validation...");
        eprintln!("[validate-perf] Work dir: {}", work_dir.display());

        let result = bench_tools::run_profiling(&work_dir, &bench_config, Some(5)).await;
        match &result {
            ToolResult::RunProfiling { flamegraph_svg_path, top_functions, total_samples } => {
                eprintln!("[validate-perf] ✓ perf profiling succeeded!");
                eprintln!("[validate-perf] Total samples: {}", total_samples);
                if !top_functions.is_empty() {
                    eprintln!("[validate-perf] Top functions:");
                    for f in top_functions.iter().take(5) {
                        eprintln!("  {:.2}%  {}", f.percentage, f.function);
                    }
                } else {
                    eprintln!("[validate-perf] (no function samples captured — profiling duration may be too short)");
                }
                if !flamegraph_svg_path.is_empty() {
                    eprintln!("[validate-perf] Flamegraph: {}", flamegraph_svg_path);
                } else {
                    eprintln!("[validate-perf] Flamegraph: skipped (stackcollapse-perf.pl / flamegraph.pl not found)");
                }
                eprintln!("[validate-perf] Validation PASSED.");
            }
            ToolResult::Error { message } => {
                eprintln!("[validate-perf] ✗ perf profiling FAILED: {}", message);
                eprintln!("[validate-perf] Validation FAILED.");
                std::process::exit(1);
            }
            _ => {
                eprintln!("[validate-perf] Unexpected result type: {:?}", result);
                std::process::exit(1);
            }
        }
        return;
    }

    // ── Normal agent mode — unwrap required args (clap guarantees presence) ──
    let api_url = args.api_url.unwrap();
    let api_key = args.api_key.unwrap();
    let model = args.model.unwrap();
    let system_prompt_path = args.system_prompt.unwrap();

    // Load system prompt
    let system_prompt = match std::fs::read_to_string(&system_prompt_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!(
                "[agent] Failed to read system prompt '{}': {}",
                system_prompt_path, e
            );
            std::process::exit(1);
        }
    };

    eprintln!("[agent] Starting agent loop");
    eprintln!("[agent] Model: {}", model);
    eprintln!("[agent] Work dir: {}", work_dir.display());
    eprintln!("[agent] Thinking mode: {}", args.thinking_mode);

    // Initialize real-time logger
    let mut logger = match AgentLogger::new(&work_dir) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[agent] Failed to initialize logger: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("[agent] Real-time log: {}", logger.path().display());
    logger.log_session_start(&model, &args.work_dir, &args.thinking_mode);

    let client = reqwest::Client::new();
    let tool_defs = get_tool_definitions();
    let mut state = AgentState::new(Some(args.max_tool_calls));

    // ── Resume from saved session context if requested ──
    let (mut messages, resumed) = if args.resume {
        match load_session_context(&work_dir) {
            Some(ctx) if ctx.tool_calls_used < ctx.tool_calls_total => {
                eprintln!(
                    "[agent] Resuming session: {}/{} tool calls used",
                    ctx.tool_calls_used, ctx.tool_calls_total
                );
                state.tool_calls_used = ctx.tool_calls_used;
                state.tool_calls_total = ctx.tool_calls_total;
                state.last_benchmark = ctx.last_benchmark;
                state.best_benchmark = ctx.best_benchmark;
                state.call_log = ctx.call_log;
                (ctx.messages, true)
            }
            Some(ctx) => {
                eprintln!(
                    "[agent] Session context found but already completed ({}/{} tool calls). Starting fresh.",
                    ctx.tool_calls_used, ctx.tool_calls_total
                );
                (vec![], false)
            }
            None => {
                eprintln!("[agent] No session context found. Starting fresh.");
                (vec![], false)
            }
        }
    } else {
        (vec![], false)
    };

    if !resumed {
        messages = vec![
            build_system_message(&system_prompt),
            ChatMessage {
                role: "user".to_string(),
                content: Some("Begin. Read the project files and start implementing.".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
    }
    let mut finished = false;
    let api_interval = std::time::Duration::from_millis(args.api_interval_ms);
    let mut last_llm_call: Option<Instant> = None;

    if args.api_interval_ms > 0 {
        eprintln!("[agent] API interval: {}ms", args.api_interval_ms);
    }

    loop {
        // Check tool call limit
        if state.is_limit_reached() {
            eprintln!(
                "[agent] Tool call limit reached ({}/{}). Auto-triggering finish.",
                state.tool_calls_used, state.tool_calls_total
            );
            let finish_call = ToolCall::Finish {
                summary: "Tool call limit reached - auto finish".to_string(),
            };
            let result = dispatch_tool_call(&finish_call, &work_dir, &bench_config, &mut state).await;
            eprintln!("[agent] Final result: {:?}", result);
            logger.log_session_end(
                state.tool_calls_used,
                state.tool_calls_total,
                state.get_status().elapsed_time_secs,
                "tool_call_limit",
            );
            break;
        }

        // Log LLM request
        logger.log_llm_request(messages.len());

        // Rate limit: wait if needed to respect api_interval_ms
        if let Some(last) = last_llm_call {
            let elapsed = last.elapsed();
            if elapsed < api_interval {
                let wait = api_interval - elapsed;
                eprintln!("[agent] Rate limit: waiting {}ms before next API call", wait.as_millis());
                tokio::time::sleep(wait).await;
            }
        }

        // Call LLM
        let llm_start = Instant::now();
        let llm_result = call_llm(
            &client,
            &api_url,
            &api_key,
            &model,
            &messages,
            &tool_defs,
            &args.thinking_mode,
            args.api_interval_ms,
            args.debug,
        )
        .await;
        let llm_duration_ms = llm_start.elapsed().as_millis() as u64;
        // Record when the LLM response came back, so the interval is measured
        // from "response received" to "next request sent".
        last_llm_call = Some(Instant::now());

        let response_msg = match llm_result {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("[agent] LLM API error: {}. Ending session.", e);
                logger.log_error(&format!("LLM API error: {}", e));
                save_session_context(&work_dir, &messages, &state);
                logger.log_session_end(
                    state.tool_calls_used,
                    state.tool_calls_total,
                    state.get_status().elapsed_time_secs,
                    "llm_error",
                );
                break;
            }
        };

        // Process response
        let reasoning_content = response_msg.reasoning_content.clone();
        if let Some(tool_calls) = response_msg.tool_calls {
            if tool_calls.is_empty() {
                // No tool calls in response, treat as content-only
                logger.log_llm_response(
                    false, 0,
                    response_msg.content.as_deref(),
                    reasoning_content.as_deref(),
                    llm_duration_ms,
                );
                if let Some(content) = &response_msg.content {
                    eprintln!("[agent] Assistant (no tools): {}", &content[..content.len().min(200)]);
                    messages.push(build_assistant_content_message(content, reasoning_content));
                    save_session_context(&work_dir, &messages, &state);
                } else {
                    eprintln!("[agent] Empty response from LLM, ending session.");
                    logger.log_session_end(
                        state.tool_calls_used,
                        state.tool_calls_total,
                        state.get_status().elapsed_time_secs,
                        "empty_response",
                    );
                    break;
                }
                continue;
            }

            // Log LLM response with tool calls
            logger.log_llm_response(
                true,
                tool_calls.len(),
                response_msg.content.as_deref(),
                reasoning_content.as_deref(),
                llm_duration_ms,
            );

            // Append assistant message with tool calls
            messages.push(build_assistant_tool_calls_message(tool_calls.clone(), reasoning_content));

            for tc in &tool_calls {
                let tool_name = &tc.function.name;
                let tool_args = &tc.function.arguments;
                let tool_call_id = &tc.id;

                eprintln!(
                    "[agent] Tool call [{}/{}]: {} (id: {})",
                    state.tool_calls_used + 1,
                    state.tool_calls_total,
                    tool_name,
                    tool_call_id
                );

                // Log tool call before execution
                logger.log_tool_call(
                    state.tool_calls_used + 1,
                    tool_name,
                    tool_args,
                    tool_call_id,
                );

                // Parse the tool call
                let parsed = parse_tool_call(tool_name, tool_args);
                let start = Instant::now();

                let (result, call_for_log) = match parsed {
                    Ok(call) => {
                        let result = dispatch_tool_call(&call, &work_dir, &bench_config, &mut state).await;
                        (result, call)
                    }
                    Err(e) => {
                        eprintln!("[agent] Parse error for tool '{}': {}", tool_name, e);
                        logger.log_error(&format!("Parse error for tool '{}': {}", tool_name, e));
                        let result = ToolResult::Error {
                            message: format!("Failed to parse tool call: {}", e),
                        };
                        (
                            result,
                            ToolCall::GetStatus, // placeholder for logging
                        )
                    }
                };

                let duration_ms = start.elapsed().as_millis() as u64;

                // Record in state
                let input_json = serde_json::to_value(&call_for_log).unwrap_or_default();
                let output_json = serde_json::to_value(&result).unwrap_or_default();
                state.record_call(
                    tool_name.clone(),
                    input_json,
                    output_json.clone(),
                    duration_ms,
                );

                // Log tool result after execution
                logger.log_tool_result(
                    state.tool_calls_used,
                    tool_name,
                    tool_call_id,
                    &output_json,
                    duration_ms,
                );

                // Append tool result message
                messages.push(build_tool_result_message(tool_call_id, &result));

                // Persist session context after each tool call for crash recovery
                save_session_context(&work_dir, &messages, &state);

                // Check if this was a finish call
                if tool_name == "finish" {
                    eprintln!("[agent] Finish tool called. Ending session.");
                    logger.log_session_end(
                        state.tool_calls_used,
                        state.tool_calls_total,
                        state.get_status().elapsed_time_secs,
                        "finish_called",
                    );
                    finished = true;
                    break;
                }

                // Check limit after each tool call
                if state.is_limit_reached() {
                    eprintln!(
                        "[agent] Tool call limit reached after executing '{}'. Auto-triggering finish.",
                        tool_name
                    );
                    let finish_call = ToolCall::Finish {
                        summary: "Tool call limit reached - auto finish".to_string(),
                    };
                    let finish_result =
                        dispatch_tool_call(&finish_call, &work_dir, &bench_config, &mut state).await;
                    eprintln!("[agent] Final result: {:?}", finish_result);
                    logger.log_session_end(
                        state.tool_calls_used,
                        state.tool_calls_total,
                        state.get_status().elapsed_time_secs,
                        "tool_call_limit",
                    );
                    finished = true;
                    break;
                }
            }

            if finished {
                break;
            }
        } else if let Some(content) = response_msg.content {
            // Assistant responded with text only (no tool calls)
            logger.log_llm_response(false, 0, Some(&content), reasoning_content.as_deref(), llm_duration_ms);
            eprintln!("[agent] Assistant: {}", &content[..content.len().min(200)]);
            messages.push(build_assistant_content_message(&content, reasoning_content));
            save_session_context(&work_dir, &messages, &state);
        } else {
            // Unexpected: no tool calls and no content
            logger.log_llm_response(false, 0, None, reasoning_content.as_deref(), llm_duration_ms);
            eprintln!("[agent] Unexpected empty response from LLM. Ending session.");
            logger.log_session_end(
                state.tool_calls_used,
                state.tool_calls_total,
                state.get_status().elapsed_time_secs,
                "empty_response",
            );
            break;
        }
    }

    // Save evaluation log
    save_eval_log(&work_dir, &state);

    // Print final status
    let status = state.get_status();
    eprintln!("[agent] Session complete.");
    eprintln!(
        "[agent] Tool calls: {}/{}",
        status.tool_calls_used, status.tool_calls_total
    );
    eprintln!("[agent] Elapsed: {:.1}s", status.elapsed_time_secs);
    if let Some(ref bench) = status.last_benchmark {
        eprintln!(
            "[agent] Final benchmark - QPS: {:.2}, Recall: {:.4}, Passed: {}",
            bench.qps, bench.recall, bench.recall_passed
        );
    }
    if let Some(ref best) = status.best_benchmark {
        eprintln!(
            "[agent] Best benchmark - QPS: {:.2}, Recall: {:.4}, Passed: {}",
            best.qps, best.recall, best.recall_passed
        );
    }
    eprintln!("[agent] Real-time log saved to: {}", logger.path().display());
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Message building tests ──────────────────────────────────────────────

    #[test]
    fn test_build_system_message() {
        let msg = build_system_message("You are a helpful assistant.");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content.as_deref(), Some("You are a helpful assistant."));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn test_build_assistant_content_message() {
        let msg = build_assistant_content_message("Hello!", None);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.as_deref(), Some("Hello!"));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.reasoning_content.is_none());

        // With reasoning content
        let msg = build_assistant_content_message("Hi!", Some("thinking...".to_string()));
        assert_eq!(msg.reasoning_content.as_deref(), Some("thinking..."));
    }

    #[test]
    fn test_build_assistant_tool_calls_message() {
        let tool_calls = vec![ToolCallMessage {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"path": "src/main.rs"}"#.to_string(),
            },
        }];
        let msg = build_assistant_tool_calls_message(tool_calls.clone(), Some("reasoning".to_string()));
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert_eq!(msg.reasoning_content.as_deref(), Some("reasoning"));
        let tcs = msg.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_123");
        assert_eq!(tcs[0].function.name, "read_file");
    }

    #[test]
    fn test_build_tool_result_message() {
        let result = ToolResult::ReadFile {
            content: "fn main() {}".to_string(),
        };
        let msg = build_tool_result_message("call_456", &result);
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_456"));
        assert!(msg.content.is_some());
        let content = msg.content.unwrap();
        assert!(content.contains("ReadFile"));
        assert!(content.contains("fn main() {}"));
    }

    #[test]
    fn test_tool_result_message_serializes_error() {
        let result = ToolResult::Error {
            message: "file not found".to_string(),
        };
        let msg = build_tool_result_message("call_err", &result);
        let content = msg.content.unwrap();
        assert!(content.contains("file not found"));
    }

    // ─── parse_tool_call tests ───────────────────────────────────────────────

    #[test]
    fn test_parse_read_file() {
        let call = parse_tool_call("read_file", r#"{"path": "src/db.rs"}"#).unwrap();
        match call {
            ToolCall::ReadFile { path } => assert_eq!(path, "src/db.rs"),
            _ => panic!("Expected ReadFile"),
        }
    }

    #[test]
    fn test_parse_write_file() {
        let call =
            parse_tool_call("write_file", r#"{"path": "out.rs", "content": "hello"}"#).unwrap();
        match call {
            ToolCall::WriteFile { path, content } => {
                assert_eq!(path, "out.rs");
                assert_eq!(content, "hello");
            }
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_parse_list_files() {
        let call = parse_tool_call("list_files", r#"{"path": "src"}"#).unwrap();
        match call {
            ToolCall::ListFiles { path } => assert_eq!(path, "src"),
            _ => panic!("Expected ListFiles"),
        }
    }

    #[test]
    fn test_parse_run_benchmark_with_params() {
        let call = parse_tool_call(
            "run_benchmark",
            r#"{"concurrency": 8, "warmup": 500}"#,
        )
        .unwrap();
        match call {
            ToolCall::RunBenchmark {
                concurrency,
                warmup,
                max_queries,
            } => {
                assert_eq!(concurrency, Some(8));
                assert_eq!(warmup, Some(500));
                assert!(max_queries.is_none());
            }
            _ => panic!("Expected RunBenchmark"),
        }
    }

    #[test]
    fn test_parse_run_benchmark_empty() {
        let call = parse_tool_call("run_benchmark", r#"{}"#).unwrap();
        match call {
            ToolCall::RunBenchmark {
                concurrency,
                warmup,
                max_queries,
            } => {
                assert!(concurrency.is_none());
                assert!(warmup.is_none());
                assert!(max_queries.is_none());
            }
            _ => panic!("Expected RunBenchmark"),
        }
    }

    #[test]
    fn test_parse_run_profiling() {
        let call = parse_tool_call("run_profiling", r#"{"duration": 60}"#).unwrap();
        match call {
            ToolCall::RunProfiling { duration } => assert_eq!(duration, Some(60)),
            _ => panic!("Expected RunProfiling"),
        }
    }

    #[test]
    fn test_parse_run_correctness_test() {
        let call = parse_tool_call("run_correctness_test", r#"{}"#).unwrap();
        assert!(matches!(call, ToolCall::RunCorrectnessTest));
    }

    #[test]
    fn test_parse_get_status() {
        let call = parse_tool_call("get_status", r#"{}"#).unwrap();
        assert!(matches!(call, ToolCall::GetStatus));
    }

    #[test]
    fn test_parse_build_project() {
        let call = parse_tool_call("build_project", r#"{}"#).unwrap();
        assert!(matches!(call, ToolCall::BuildProject));
    }

    #[test]
    fn test_parse_finish() {
        let call =
            parse_tool_call("finish", r#"{"summary": "Optimized search"}"#).unwrap();
        match call {
            ToolCall::Finish { summary } => assert_eq!(summary, "Optimized search"),
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn test_parse_unknown_tool() {
        let result = parse_tool_call("unknown_tool", r#"{}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tool"));
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_tool_call("read_file", "not json");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid JSON"));
    }

    #[test]
    fn test_parse_missing_required_field() {
        let result = parse_tool_call("read_file", r#"{"wrong_field": "value"}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing"));
    }

    // ─── ChatMessage serialization tests ─────────────────────────────────────

    #[test]
    fn test_system_message_serialization() {
        let msg = build_system_message("test prompt");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "test prompt");
        // Optional fields should be absent (skip_serializing_if)
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_call_id").is_none());
    }

    #[test]
    fn test_tool_result_message_serialization() {
        let result = ToolResult::ListFiles {
            files: vec!["a.rs".to_string(), "b.rs".to_string()],
        };
        let msg = build_tool_result_message("call_789", &result);
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_789");
        assert!(json["content"].is_string());
    }

    #[test]
    fn test_chat_request_serialization() {
        let messages = vec![build_system_message("hello")];
        let tools = get_tool_definitions();
        let req = ChatRequest {
            model: "gpt-4".to_string(),
            messages,
            tools,
            tool_choice: "auto".to_string(),
            extra_body: std::collections::HashMap::new(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4");
        assert_eq!(json["tool_choice"], "auto");
        assert!(json["messages"].is_array());
        assert!(json["tools"].is_array());
        assert_eq!(json["tools"].as_array().unwrap().len(), 9);
    }

    // ─── save_eval_log test ──────────────────────────────────────────────────

    // ─── thinking mode tests ─────────────────────────────────────────────────

    #[test]
    fn test_thinking_mode_false() {
        let extra = build_thinking_extra("false");
        assert!(extra.is_empty());
    }

    #[test]
    fn test_thinking_mode_true() {
        let extra = build_thinking_extra("true");
        assert_eq!(extra["thinking"], serde_json::json!({"type": "enabled"}));
    }

    #[test]
    fn test_thinking_mode_openai() {
        let extra = build_thinking_extra("openai");
        assert_eq!(extra["thinking"], serde_json::json!({"type": "enabled"}));
    }

    #[test]
    fn test_thinking_mode_kimi() {
        let extra = build_thinking_extra("kimi");
        assert_eq!(extra["enable_thinking"], serde_json::json!(true));
    }

    #[test]
    fn test_thinking_mode_gemini() {
        let extra = build_thinking_extra("gemini");
        assert_eq!(extra["reasoning"], serde_json::json!({"enabled": true}));
    }

    #[test]
    fn test_thinking_mode_gemini_flattened_in_request() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            tool_choice: "auto".to_string(),
            extra_body: build_thinking_extra("gemini"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["enabled"], true);
        assert!(json.get("extra_body").is_none());
    }

    #[test]
    fn test_thinking_mode_flattened_in_request() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            tool_choice: "auto".to_string(),
            extra_body: build_thinking_extra("openai"),
        };
        let json = serde_json::to_value(&req).unwrap();
        // "thinking" should be at the top level, not nested under "extra_body"
        assert_eq!(json["thinking"]["type"], "enabled");
        assert!(json.get("extra_body").is_none());
    }

    #[test]
    fn test_thinking_mode_kimi_flattened_in_request() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            tool_choice: "auto".to_string(),
            extra_body: build_thinking_extra("kimi"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["enable_thinking"], true);
        assert!(json.get("extra_body").is_none());
    }

    #[test]
    fn test_save_eval_log_creates_file() {
        let dir = std::env::temp_dir().join(format!("agent_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut state = AgentState::new(None);
        state.record_call(
            "read_file".to_string(),
            serde_json::json!({"path": "test.rs"}),
            serde_json::json!({"content": "hello"}),
            10,
        );

        save_eval_log(&dir, &state);

        let log_path = dir.join("eval_log.json");
        assert!(log_path.exists());

        let content = std::fs::read_to_string(&log_path).unwrap();
        let log: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(log["tool_calls_used"], 1);
        assert_eq!(log["tool_calls_total"], 50);
        assert_eq!(log["call_log"].as_array().unwrap().len(), 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─── session context tests ───────────────────────────────────────────────

    #[test]
    fn test_session_context_save_and_load() {
        let dir = std::env::temp_dir().join(format!("agent_ctx_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut state = AgentState::new(Some(50));
        state.record_call(
            "read_file".to_string(),
            serde_json::json!({"path": "test.rs"}),
            serde_json::json!({"content": "hello"}),
            10,
        );

        let messages = vec![
            build_system_message("system prompt"),
            ChatMessage {
                role: "user".to_string(),
                content: Some("Begin.".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        save_session_context(&dir, &messages, &state);

        let ctx = load_session_context(&dir).expect("should load session context");
        assert_eq!(ctx.tool_calls_used, 1);
        assert_eq!(ctx.tool_calls_total, 50);
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role, "system");
        assert_eq!(ctx.messages[1].role, "user");
        assert_eq!(ctx.call_log.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_session_context_load_nonexistent() {
        let dir = std::env::temp_dir().join(format!("agent_ctx_none_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let ctx = load_session_context(&dir);
        assert!(ctx.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
