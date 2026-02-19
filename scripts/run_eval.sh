#!/bin/bash
# =============================================================================
# run_eval.sh - 完整评测流程编排脚本
#
# 一键执行从数据准备到最终评分的全部步骤：
#   1. 下载 SIFT1M 数据集（如已存在则跳过）
#   2. 转换 fvecs/ivecs → JSON（如已存在则跳过）
#   3. 生成 ground truth（如已存在则跳过）
#   4. 构建 benchmark client 和 agent 框架
#   5. 初始化工作目录（复制骨架代码）
#   6. 启动 Agent 框架运行评测
#   7. 收集结果并更新排行榜
#
# 配置（通过环境变量覆盖）：
#   MODEL_NAME      - 被测模型名称（必需）
#   API_URL         - LLM API 端点（必需）
#   API_KEY         - LLM API 密钥（必需）
#   MODEL_ID        - API 模型标识符（必需）
#   THINKING_MODE   - 模型思考模式（默认: false，可选: true/openai/kimi/gemini）
#   API_INTERVAL_MS - LLM API 调用最小间隔毫秒数，用于限速（默认: 0，不限速）
#   CPU_CORES       - 服务器绑定的 CPU 核心，taskset 格式（默认: 0-3，空字符串禁用）
#   WORK_DIR        - 模型工作目录（默认: ./workdir，会被清空重建）
#   DATA_DIR        - 数据集目录（默认: ./data）
#   RESULTS_DIR     - 结果输出目录（默认: ./results）
#   MAX_TOOL_CALLS  - 最大工具调用次数（默认: 50）
#
# Usage:
#   MODEL_NAME=gpt-4o API_URL=https://api.openai.com/v1 API_KEY=sk-xxx MODEL_ID=gpt-4o \
#     bash scripts/run_eval.sh
#
#   # 带限速和自定义核心绑定：
#   MODEL_NAME=qwen3.5-plus API_INTERVAL_MS=2000 CPU_CORES=4-7 THINKING_MODE=openai \
#     WORK_DIR=./leaderboard/qwen3.5-plus \
#     API_URL=https://... API_KEY=sk-xxx MODEL_ID=qwen-plus \
#     bash scripts/run_eval.sh
# =============================================================================
set -euo pipefail

# ─── Resolve project root (directory containing this script's parent) ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Check for rebuild-leaderboard mode (no other env vars needed) ───────────
if [[ "${1:-}" == "--rebuild-leaderboard" ]]; then
    LEADERBOARD_DIR="${LEADERBOARD_DIR:-${PROJECT_ROOT}/leaderboard}"
    RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
    exec python3 "${SCRIPT_DIR}/rebuild_leaderboard.py" \
        --leaderboard-dir "${LEADERBOARD_DIR}" \
        --output "${RESULTS_DIR}/leaderboard.json"
fi

# ─── Configuration with defaults ─────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME environment variable is required}"
API_URL="${API_URL:?ERROR: API_URL environment variable is required}"
API_KEY="${API_KEY:?ERROR: API_KEY environment variable is required}"
MODEL_ID="${MODEL_ID:?ERROR: MODEL_ID environment variable is required}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/workdir}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
THINKING_MODE="${THINKING_MODE:-false}"
API_INTERVAL_MS="${API_INTERVAL_MS:-0}"
CPU_CORES="${CPU_CORES:-0-3}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-50}"
DEBUG="${DEBUG:-false}"

# Sanitize MODEL_ID for use in file paths (replace / with -)
MODEL_ID_SAFE="$(echo "${MODEL_ID}" | sed 's|/|-|g')"

# Sanitize MODEL_NAME for use in file paths (replace / with -)
MODEL_NAME_SAFE="$(echo "${MODEL_NAME}" | sed 's|/|-|g')"

SYSTEM_PROMPT="${PROJECT_ROOT}/agent/system_prompt.txt"
SKELETON_DIR="${PROJECT_ROOT}/skeleton"
BENCHMARK_BIN="${PROJECT_ROOT}/benchmark/target/release/vector-db-benchmark"
AGENT_BIN="${PROJECT_ROOT}/agent/target/release/vector-db-agent"
RESUME_SESSION=false

# ─── Helper functions ─────────────────────────────────────────────────────────
log() {
    echo "[run_eval] $(date '+%H:%M:%S') $*" >&2
}

die() {
    echo "[run_eval] ERROR: $*" >&2
    exit 1
}

# ─── Step 1: Download SIFT1M dataset ─────────────────────────────────────────
step_download_dataset() {
    log "Step 1: Checking SIFT1M dataset..."

    local needed_files=(
        "${DATA_DIR}/sift_base.fvecs"
        "${DATA_DIR}/sift_query.fvecs"
        "${DATA_DIR}/sift_groundtruth.ivecs"
    )

    local all_exist=true
    for f in "${needed_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            all_exist=false
            break
        fi
    done

    if $all_exist; then
        log "  Dataset files already exist. Skipping download."
    else
        log "  Downloading SIFT1M dataset..."
        python3 "${PROJECT_ROOT}/scripts/download_dataset.py" \
            || die "Dataset download failed"
        log "  Download complete."
    fi
}

# ─── Step 2: Convert fvecs/ivecs → JSON ──────────────────────────────────────
step_convert_data() {
    log "Step 2: Checking JSON data files..."

    # Check if base vectors JSON exists (single or sharded)
    if [[ -f "${DATA_DIR}/base_vectors.json" ]] || ls "${DATA_DIR}"/base_vectors_*.json &>/dev/null; then
        if [[ -f "${DATA_DIR}/query_vectors.json" ]]; then
            log "  JSON data files already exist. Skipping conversion."
            return
        fi
    fi

    log "  Converting fvecs/ivecs to JSON..."
    python3 "${PROJECT_ROOT}/scripts/convert_data.py" --data-dir "${DATA_DIR}" \
        || die "Data conversion failed"
    log "  Conversion complete."
}

# ─── Step 3: Generate ground truth ───────────────────────────────────────────
step_generate_ground_truth() {
    log "Step 3: Checking ground truth..."

    if [[ -f "${DATA_DIR}/ground_truth.json" ]]; then
        log "  Ground truth already exists. Skipping generation."
        return
    fi

    log "  Generating ground truth via brute-force search (this may take a while)..."
    python3 "${PROJECT_ROOT}/scripts/generate_ground_truth.py" --data-dir "${DATA_DIR}" --top-k 100 \
        || die "Ground truth generation failed"
    log "  Ground truth generation complete."
}

# ─── Step 4: Build Rust binaries ─────────────────────────────────────────────
step_build_binaries() {
    log "Step 4: Building Rust binaries..."

    log "  Building benchmark client..."
    (cd "${PROJECT_ROOT}/benchmark" && cargo build --release) \
        || die "Benchmark client build failed"

    log "  Building agent framework..."
    (cd "${PROJECT_ROOT}/agent" && cargo build --release) \
        || die "Agent framework build failed"

    # Verify binaries exist
    [[ -f "${BENCHMARK_BIN}" ]] || die "Benchmark binary not found at ${BENCHMARK_BIN}"
    [[ -f "${AGENT_BIN}" ]] || die "Agent binary not found at ${AGENT_BIN}"

    log "  Build complete."
}

# ─── Step 5: Initialize working directory ────────────────────────────────────
step_init_workdir() {
    log "Step 5: Initializing working directory..."

    # If resuming (session_context.json exists and not completed), skip re-init
    local ctx_file="${WORK_DIR}/session_context.json"
    if [[ -f "${ctx_file}" ]]; then
        # Check if session is already completed (tool_calls_used >= tool_calls_total)
        local used total
        used="$(python3 -c "import json; d=json.load(open('${ctx_file}')); print(d.get('tool_calls_used',0))" 2>/dev/null || echo 0)"
        total="$(python3 -c "import json; d=json.load(open('${ctx_file}')); print(d.get('tool_calls_total',50))" 2>/dev/null || echo 50)"
        if [[ "${used}" -lt "${total}" ]]; then
            log "  Session context found (${used}/${total} tool calls). Skipping workdir re-init for resume."
            RESUME_SESSION=true
            return
        else
            log "  Session context found but already completed (${used}/${total}). Re-initializing."
        fi
    fi

    RESUME_SESSION=false

    if [[ -d "${WORK_DIR}" ]]; then
        log "  Removing existing working directory: ${WORK_DIR}"
        rm -rf "${WORK_DIR}"
    fi

    log "  Copying skeleton to ${WORK_DIR}..."
    cp -r "${SKELETON_DIR}" "${WORK_DIR}" \
        || die "Failed to copy skeleton to working directory"

    log "  Working directory initialized."
}

# ─── Step 6: Run the Agent ───────────────────────────────────────────────────
step_run_agent() {
    log "Step 6: Starting Agent framework..."
    log "  Model: ${MODEL_NAME} (${MODEL_ID})"
    log "  API URL: ${API_URL}"
    log "  Work dir: ${WORK_DIR}"

    [[ -f "${SYSTEM_PROMPT}" ]] || die "System prompt not found at ${SYSTEM_PROMPT}"
    [[ -d "${DATA_DIR}" ]] || die "Data directory not found at ${DATA_DIR}"
    [[ -f "${BENCHMARK_BIN}" ]] || die "Benchmark binary not found at ${BENCHMARK_BIN}"

    local debug_flag=""
    if [[ "${DEBUG}" == "true" || "${DEBUG}" == "1" ]]; then
        debug_flag="--debug"
        log "  Debug mode: ON"
    fi

    local resume_flag=""
    if [[ "${RESUME_SESSION:-false}" == "true" ]]; then
        resume_flag="--resume"
        log "  Resume mode: ON"
    fi

    "${AGENT_BIN}" \
        --api-url "${API_URL}" \
        --api-key "${API_KEY}" \
        --model "${MODEL_ID}" \
        --system-prompt "${SYSTEM_PROMPT}" \
        --work-dir "${WORK_DIR}" \
        --thinking-mode "${THINKING_MODE}" \
        --api-interval-ms "${API_INTERVAL_MS}" \
        --data-dir "${DATA_DIR}" \
        --benchmark-bin "${BENCHMARK_BIN}" \
        --cpu-cores "${CPU_CORES}" \
        --max-tool-calls "${MAX_TOOL_CALLS}" \
        ${debug_flag} \
        ${resume_flag} \
        || die "Agent framework exited with error"

    log "  Agent session complete."
}

# ─── Step 7: Collect results and update leaderboard ──────────────────────────
step_collect_results() {
    log "Step 7: Collecting results and updating leaderboard..."

    mkdir -p "${RESULTS_DIR}"

    local eval_log="${WORK_DIR}/eval_log.json"
    if [[ ! -f "${eval_log}" ]]; then
        die "Evaluation log not found at ${eval_log}. Agent may not have completed."
    fi

    # Copy eval log to results directory with model name
    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    local result_file="${RESULTS_DIR}/${MODEL_NAME_SAFE}_${timestamp}.json"

    # Build a combined result file with model metadata.
    # Uses best_benchmark (tracked across all runs) as primary source,
    # falls back to last_benchmark, then scans benchmarks/*.json files.
    python3 -c "
import json, sys, os, glob

eval_log_path = '${eval_log}'
model_name = '${MODEL_NAME}'
model_id = '${MODEL_ID}'
work_dir = '${WORK_DIR}'

with open(eval_log_path, 'r') as f:
    eval_log = json.load(f)

result = {
    'model_name': model_name,
    'model_id': model_id,
    'eval_log': eval_log,
}

# Priority 1: best_benchmark from eval_log (tracked across all runs by agent)
best_bench = eval_log.get('best_benchmark')

# Priority 2: last_benchmark (from the final finish() call)
last_bench = eval_log.get('last_benchmark')

# Priority 3: scan all benchmarks/*.json for the highest QPS with passing recall
def scan_benchmark_files():
    bench_dir = os.path.join(work_dir, 'benchmarks')
    if not os.path.isdir(bench_dir):
        return None
    best = None
    for f in sorted(glob.glob(os.path.join(bench_dir, 'benchmark_*.json'))):
        try:
            with open(f, 'r') as fh:
                b = json.load(fh)
            if b.get('recall_passed', False) and b.get('qps', 0) > 0:
                if best is None or b['qps'] > best['qps']:
                    best = b
        except (json.JSONDecodeError, KeyError):
            continue
    return best

# Pick the best available result
chosen = None
chosen_source = 'none'

if best_bench and best_bench.get('recall_passed', False) and best_bench.get('qps', 0) > 0:
    chosen = best_bench
    chosen_source = 'best_benchmark'

if last_bench and last_bench.get('recall_passed', False) and last_bench.get('qps', 0) > 0:
    if chosen is None or last_bench['qps'] > chosen['qps']:
        chosen = last_bench
        chosen_source = 'last_benchmark'

scanned = scan_benchmark_files()
if scanned:
    if chosen is None or scanned['qps'] > chosen['qps']:
        chosen = scanned
        chosen_source = 'benchmark_files_scan'

if chosen:
    result['final_benchmark'] = chosen
    result['qps'] = chosen.get('qps', 0)
    result['recall'] = chosen.get('recall', 0)
    result['recall_passed'] = chosen.get('recall_passed', False)
    result['result_source'] = chosen_source
    print(f'[collect] Using {chosen_source}: QPS={chosen[\"qps\"]:.2f}, Recall={chosen[\"recall\"]:.4f}', file=sys.stderr)
else:
    result['qps'] = 0
    result['recall'] = 0
    result['recall_passed'] = False
    result['result_source'] = 'none'
    print('[collect] No valid benchmark result found', file=sys.stderr)

with open('${result_file}', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps({
    'model': model_name,
    'qps': result['qps'],
    'recall': result['recall'],
    'recall_passed': result['recall_passed'],
    'tool_calls_used': eval_log.get('tool_calls_used', 0),
    'result_source': result.get('result_source', 'none'),
}, indent=2))
" || die "Failed to collect results"

    log "  Results saved to ${result_file}"

    # Update leaderboard
    local leaderboard="${RESULTS_DIR}/leaderboard.json"
    python3 -c "
import json, sys
from datetime import datetime, timezone

result_path = '${result_file}'
leaderboard_path = '${leaderboard}'

with open(result_path, 'r') as f:
    result = json.load(f)

# Load existing leaderboard or start fresh
try:
    with open(leaderboard_path, 'r') as f:
        entries = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    entries = []

# Compute final score: QPS = 0 if recall < 0.95
qps = result.get('qps', 0)
recall = result.get('recall', 0)
if recall < 0.95:
    qps = 0

entry = {
    'model_name': result['model_name'],
    'qps': qps,
    'recall': recall,
    'recall_passed': result.get('recall_passed', False),
    'tool_calls_used': result.get('eval_log', {}).get('tool_calls_used', 0),
    'result_source': result.get('result_source', 'unknown'),
    'timestamp': datetime.now(timezone.utc).isoformat(),
}

entries.append(entry)

# Sort: QPS descending, then recall descending for ties
entries.sort(key=lambda e: (-e['qps'], -e['recall']))

with open(leaderboard_path, 'w') as f:
    json.dump(entries, f, indent=2)

print()
print('=== Leaderboard ===')
for i, e in enumerate(entries):
    marker = ' <-- NEW' if e['model_name'] == result['model_name'] and e['timestamp'] == entry['timestamp'] else ''
    src = f\" [{e.get('result_source', '?')}]\" if e.get('result_source') else ''
    print(f\"  {i+1}. {e['model_name']:20s}  QPS: {e['qps']:>10.2f}  Recall: {e['recall']:.4f}{src}{marker}\")
print()
" || die "Failed to update leaderboard"

    log "  Leaderboard updated at ${leaderboard}"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    log "=========================================="
    log "Vector DB Bench - Evaluation Pipeline"
    log "=========================================="
    log "Model:       ${MODEL_NAME}"
    log "API URL:     ${API_URL}"
    log "Model ID:    ${MODEL_ID}"
    log "Thinking:    ${THINKING_MODE}"
    log "API interval:${API_INTERVAL_MS}ms"
    log "CPU cores:   ${CPU_CORES}"
    log "Max tools:   ${MAX_TOOL_CALLS}"
    log "Data dir:    ${DATA_DIR}"
    log "Work dir:    ${WORK_DIR}"
    log "Results dir: ${RESULTS_DIR}"
    log "=========================================="

    step_download_dataset
    step_convert_data
    step_generate_ground_truth
    step_build_binaries
    step_init_workdir
    step_run_agent
    step_collect_results

    log "=========================================="
    log "Evaluation complete for ${MODEL_NAME}!"
    log "=========================================="
}

main "$@"
