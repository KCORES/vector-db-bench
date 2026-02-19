# Vector DB Bench 使用指南

## 项目简介

Vector DB Bench 是一个大模型后端代码能力评测系统。它让大模型从零实现一个向量数据库（Rust），然后自主进行性能优化，最终以搜索 QPS 作为核心指标进行排名。

评测流程：提供骨架代码 → 模型实现基础版本 → 模型自主迭代优化（50 次 tool call）→ 最终 benchmark → 进入天梯排行榜。

## 环境要求

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Rust | stable (1.70+) | 编译骨架代码、benchmark client、agent 框架 |
| Python | 3.8+ | 数据预处理脚本 |
| git + git-lfs | — | 从 HuggingFace 下载 SIFT1M 数据集 |
| numpy | 推荐安装 | 加速 ground truth 生成（可选，纯 Python 回退） |
| Linux | 推荐 | 评测运行环境（profiling 依赖 perf） |

可选依赖：
- `perf` — Linux 性能分析工具，用于 `run_profiling` 工具
- `flamegraph.pl` + `stackcollapse-perf.pl` — 火焰图生成（来自 [FlameGraph](https://github.com/brendangregg/FlameGraph)）

## 项目结构

```
vector-db-bench/
├── skeleton/           # 骨架代码（提供给被测模型的 Rust 项目模板）
│   └── src/
│       ├── main.rs     # HTTP 服务入口（只读）
│       ├── api.rs      # API 类型定义（只读）
│       ├── db.rs       # VectorDB 实现区域（模型填写）
│       └── distance.rs # L2 距离计算（模型填写）
├── benchmark/          # Benchmark Client（性能测试客户端）
│   └── src/
│       ├── main.rs     # CLI 入口
│       ├── loader.rs   # 数据加载
│       ├── runner.rs   # 并发查询运行器
│       ├── scorer.rs   # QPS/recall 评分
│       └── anti_cheat.rs # 反作弊检测
├── agent/              # Agent 框架（Tool Call Agent 运行时）
│   ├── src/
│   │   ├── main.rs     # Agent 主循环 + LLM API 客户端
│   │   ├── tools.rs    # Tool Call 类型定义和路由
│   │   ├── sandbox.rs  # 文件操作 + 命令执行
│   │   ├── bench_tools.rs # Benchmark/Profiling 工具
│   │   ├── state.rs    # 状态管理（计数/日志）
│   │   └── evaluator.rs # 评分和排行榜逻辑
│   └── system_prompt.txt # 系统 Prompt（公平性保障）
├── scripts/            # 数据预处理和评测脚本
│   ├── download_dataset.py      # 下载 SIFT1M 数据集
│   ├── convert_data.py          # fvecs/ivecs → JSON 转换
│   ├── generate_ground_truth.py # 暴力搜索生成 ground truth
│   ├── load_data.py             # 数据加载到被测服务
│   ├── run_eval.sh              # 一键评测流程脚本
│   └── rebuild_leaderboard.py   # 重建排行榜脚本
├── data/               # 数据集目录（运行时生成）
└── results/            # 评测结果和排行榜（运行时生成）
```

## 快速开始

### 一键评测（推荐）

最简单的方式是使用 `run_eval.sh` 脚本，它会自动完成所有步骤：

```bash
MODEL_NAME=gpt-4o \
CPU_CORES=0-3 \
THINKING_MODE=openai \
API_INTERVAL_MS=10000 \
API_URL=https://api.openai.com/v1 \
API_KEY=sk-your-api-key \
MODEL_ID=gpt-4o \
WORK_DIR=./leaderboard/gpt-4o-turn-1 \
bash scripts/run_eval.sh
```

如需启用模型思考模式（thinking/reasoning）：

```bash
# OpenAI 系列（如 o1、o3）— 发送 {"thinking": {"type": "enabled"}}
THINKING_MODE=openai bash scripts/run_eval.sh

# Kimi 系列 — 发送 {"enable_thinking": true}
THINKING_MODE=kimi bash scripts/run_eval.sh

# Gemini 系列（如 gemini-3.1-pro-preview）— 发送 {"reasoning": {"enabled": true}}
THINKING_MODE=gemini bash scripts/run_eval.sh
```

脚本会依次执行：
1. 下载 SIFT1M 数据集（约 500MB，已存在则跳过）
2. 转换 fvecs → JSON 格式（已存在则跳过）
3. 生成 ground truth（已存在则跳过）
4. 编译 benchmark client 和 agent 框架
5. 初始化工作目录（复制骨架代码，如检测到可恢复的会话则跳过）
6. 启动 Agent 运行评测（如有可恢复会话则自动 resume）
7. 收集结果并更新排行榜

### 断点续跑（Session Resume）

Agent 在运行过程中会自动将会话上下文（LLM 对话历史、tool call 计数、benchmark 记录等）持久化到 `session_context.json`。如果评测中途因 API 错误、网络中断或进程崩溃而失败，重新运行同一命令即可自动恢复：

```bash
# 首次运行（中途失败）
MODEL_NAME=gemini WORK_DIR=./leaderboard/gemini-turn-1 \
  API_URL=https://openrouter.ai/api/v1 API_KEY=sk-xxx MODEL_ID=google/gemini-3.1-pro-preview \
  bash scripts/run_eval.sh

# 直接重新运行同一命令，脚本会自动检测 session_context.json 并恢复
MODEL_NAME=gemini WORK_DIR=./leaderboard/gemini-turn-1 \
  API_URL=https://openrouter.ai/api/v1 API_KEY=sk-xxx MODEL_ID=google/gemini-3.1-pro-preview \
  bash scripts/run_eval.sh
```

恢复机制：
- 脚本检测到 `WORK_DIR/session_context.json` 且 tool call 未用尽时，跳过工作目录重建，以 `--resume` 模式启动 Agent
- Agent 从 `session_context.json` 恢复完整的对话历史和状态，继续与 LLM 交互
- 如果会话已完成（tool call 已用尽），则正常重新开始

手动使用 `--resume` 参数：

```bash
./agent/target/release/vector-db-agent \
  --api-url https://openrouter.ai/api/v1 \
  --api-key sk-xxx \
  --model google/gemini-3.1-pro-preview \
  --system-prompt agent/system_prompt.txt \
  --work-dir ./workdir \
  --resume
```

注意：`MODEL_ID` 中的 `/`（如 `google/gemini-3.1-pro-preview`）会被自动替换为 `-`，避免生成结果文件时创建意外的子目录。

### 环境变量配置

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_NAME` | 是 | — | 被测模型名称（用于排行榜显示） |
| `API_URL` | 是 | — | LLM API 端点（OpenAI 兼容格式） |
| `API_KEY` | 是 | — | LLM API 密钥 |
| `MODEL_ID` | 是 | — | API 中的模型标识符 |
| `THINKING_MODE` | 否 | `false` | 模型思考模式（`false`/`true`/`openai`/`kimi`/`gemini`） |
| `API_INTERVAL_MS` | 否 | `0` | LLM API 调用最小间隔（毫秒），用于限速防止 429 |
| `CPU_CORES` | 否 | `0-3` | 服务器进程绑定的 CPU 核心（taskset 格式，空字符串禁用） |
| `MAX_TOOL_CALLS` | 否 | `50` | 最大工具调用次数，耗尽后自动结束评测 |
| `WORK_DIR` | 否 | `./workdir` | 模型工作目录（会被清空重建） |
| `DATA_DIR` | 否 | `./data` | 数据集存放目录 |
| `RESULTS_DIR` | 否 | `./results` | 结果输出目录 |

### 支持的 LLM API

Agent 框架使用 OpenAI 兼容的 Chat Completions API（支持 function calling）。兼容的服务包括：

```bash
# OpenAI
API_URL=https://api.openai.com/v1 MODEL_ID=gpt-4o

# Anthropic (通过兼容层)
API_URL=https://your-proxy/v1 MODEL_ID=claude-sonnet-4-20250514

# 本地部署 (vLLM / Ollama 等)
API_URL=http://localhost:8000/v1 MODEL_ID=your-model-name
```

## 分步手动操作

如果需要更细粒度的控制，可以手动执行各步骤。

### 1. 数据准备

```bash
# 下载 SIFT1M 数据集到 data/ 目录（从 HuggingFace clone，需要 git-lfs）
python3 scripts/download_dataset.py

# 转换为 JSON 格式（支持分片，默认每片 10 万条）
python3 scripts/convert_data.py --data-dir data/ --shard-size 100000

# 生成 ground truth（暴力搜索 Top-100，分块落盘，支持断点续传）
python3 scripts/generate_ground_truth.py --data-dir data/ --top-k 100

# 多进程加速 + 自定义分块大小
python3 scripts/generate_ground_truth.py --data-dir data/ --top-k 100 --workers 8 --chunk-size 500

# 如果中途中断，重新运行同一命令即可自动跳过已完成的分块
```

数据准备完成后，`data/` 目录下应有：
```
data/
├── sift_base.fvecs          # 原始二进制（1M 条 128 维向量）
├── sift_query.fvecs         # 原始二进制（1 万条查询向量）
├── sift_groundtruth.ivecs   # 原始二进制 ground truth
├── base_vectors_0.json      # JSON 格式 base vectors（分片）
├── base_vectors_1.json
├── ...
├── query_vectors.json       # JSON 格式 query vectors
└── ground_truth.json        # JSON 格式 ground truth（Top-100）
```

### 2. 编译 Rust 组件

```bash
# 编译骨架代码（验证模板可用）
cd skeleton && cargo build --release && cd ..

# 编译 benchmark client
cd benchmark && cargo build --release && cd ..

# 编译 agent 框架
cd agent && cargo build --release && cd ..
```

### 3. 运行评测

```bash
# 初始化工作目录
cp -r skeleton/ workdir/

# 启动 agent
./agent/target/release/vector-db-agent \
  --api-url https://api.openai.com/v1 \
  --api-key sk-your-api-key \
  --model gpt-4o \
  --system-prompt agent/system_prompt.txt \
  --work-dir ./workdir \
  --thinking-mode false \
  --api-interval-ms 0 \
  --cpu-cores 0-3 \
  --max-tool-calls 50
```

Agent 会自动：
- 加载系统 Prompt
- 与 LLM 进行 Tool Call 交互循环
- 记录每次 tool call 的输入/输出/耗时
- 每次 tool call 后自动保存会话上下文到 `session_context.json`（支持断点续跑）
- 在 tool call 用尽（默认 50 次，可通过 `--max-tool-calls` 调整）或模型调用 `finish` 时结束
- 保存评测日志到 `workdir/eval_log.json`

如需从中断处恢复，添加 `--resume` 参数重新运行即可。

### 4. 手动运行 Benchmark

如果想单独测试一个已实现的向量数据库：

```bash
# 先启动被测服务（在 workdir 中编译并运行）
cd workdir && cargo build --release && ./target/release/vector-db-skeleton &

# 加载数据
python3 scripts/load_data.py --server-url http://127.0.0.1:8080 --data-dir data/

# 运行 benchmark
./benchmark/target/release/vector-db-benchmark \
  --server-url http://127.0.0.1:8080 \
  --concurrency 4 \
  --warmup 1000 \
  --base-vectors data/base_vectors.json \
  --query-vectors data/query_vectors.json \
  --ground-truth data/ground_truth.json \
  --recall-threshold 0.95 \
  --seed 42
```

Benchmark 输出 JSON 格式结果到 stdout，包含：
```json
{
  "benchmark": {
    "qps": 1500.0,
    "total_queries": 10000,
    "duration_secs": 6.67,
    "avg_latency_ms": 2.5,
    "p50_latency_ms": 2.0,
    "p95_latency_ms": 5.0,
    "p99_latency_ms": 10.0,
    "recall": 0.97,
    "recall_threshold": 0.95,
    "recall_passed": true,
    "concurrency": 4
  },
  "anti_cheat": {
    "passed": true,
    "avg_jaccard_similarity": 0.05,
    "unique_ids": 9800,
    "total_results": 100000,
    "message": "OK: ..."
  }
}
```

## Benchmark 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server-url` | `http://127.0.0.1:8080` | 被测服务地址 |
| `--concurrency` | `4` | 并发查询线程数 |
| `--warmup` | `1000` | 预热查询数（不计入评分） |
| `--recall-threshold` | `0.95` | Recall 通过阈值 |
| `--seed` | `42` | 查询顺序随机种子 |
| `--batch-size` | `5000` | bulk_insert 每批向量数 |

## 评分规则

1. 正确性优先：recall 必须 >= 95%，否则 QPS 直接记为 0
2. 排行榜按 QPS 降序排列，QPS 相同时 recall 更高者排名靠前
3. 反作弊检测：分析查询结果的多样性，检测硬编码结果（Jaccard 相似度阈值 0.8）

## 模型可用的 Tool Call

Agent 框架为被测模型提供 9 个工具：

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件（只读文件会被拒绝） |
| `list_files` | 列出目录文件 |
| `run_command` | 执行 shell 命令（默认超时 120 秒） |
| `run_benchmark` | 运行完整 benchmark（QPS + recall） |
| `run_profiling` | 性能分析（火焰图 + 热点函数） |
| `run_correctness_test` | 快速正确性校验 |
| `get_status` | 查看剩余 tool call 次数、时间等 |
| `finish` | 完成优化，触发最终 benchmark |

只读文件（模型不可修改）：
- `src/main.rs`、`src/api.rs` — HTTP 路由和 API 类型
- `benchmark/` — 整个 benchmark 目录
- `scripts/load_data.py` — 数据加载脚本

## 公平性保障

- 系统 Prompt 不包含任何具体算法名称（HNSW、IVF、PQ 等）
- 系统 Prompt 不包含任何优化技术名称（SIMD、内存池、lock-free 等）
- 所有模型使用完全相同的 Prompt 和工具集
- 查询顺序随机打乱，防止缓存预测
- 反作弊检测分析结果多样性

## 查看结果

评测完成后，结果保存在 `results/` 目录：

```bash
results/
├── leaderboard.json              # 排行榜（所有模型）
├── gpt-4o_20260215_143000.json   # 单次评测详细结果
└── claude-sonnet_20260215_150000.json
```

查看排行榜：
```bash
cat results/leaderboard.json | python3 -m json.tool
```

### 重建排行榜

如果排行榜数据出现问题（如评测中途异常），可以从 `leaderboard/` 目录下的所有条目重新构建：

```bash
# 方式一：通过 run_eval.sh
bash scripts/run_eval.sh --rebuild-leaderboard

# 方式二：直接运行 Python 脚本
python3 scripts/rebuild_leaderboard.py --leaderboard-dir ./leaderboard --output ./results/leaderboard.json

# 自定义路径
LEADERBOARD_DIR=./my-results RESULTS_DIR=./my-output bash scripts/run_eval.sh --rebuild-leaderboard
```

脚本会扫描 `leaderboard/` 下每个子目录，从 `agent_log.jsonl` 第一行读取模型名称，然后按以下优先级提取最佳 QPS 结果：

1. `eval_log.json` 中的 `best_benchmark`（Agent 运行期间跟踪的最高 QPS）
2. `eval_log.json` 中的 `last_benchmark`（`finish()` 调用时的最终结果）
3. `final_report.json` 中的 `benchmark`（如果存在）
4. 扫描 `benchmarks/*.json` 文件，取 recall 通过且 QPS 最高的结果

### 最佳 QPS 自动备份

Agent 在运行过程中会自动跟踪历史最高 QPS（仅计 recall >= 95% 的结果）。每当出现新的最高 QPS 时：

- `src/` 目录会被自动备份到工作目录下的 `src_best_qps/`
- `eval_log.json` 中会记录 `best_benchmark` 字段

这样即使模型在后续迭代中改坏了代码（如 tool call 用尽时正好处于编译失败状态），之前的最佳实现和成绩都不会丢失。`finish()` 时如果最终 benchmark 失败，会自动回退使用 `best_benchmark` 的结果。

查看单次评测的 tool call 日志：
```bash
cat workdir/eval_log.json | python3 -m json.tool
```

查看实时运行日志（JSONL 格式，每行一个事件）：
```bash
# 实时跟踪（Linux/macOS）
tail -f workdir/agent_log.jsonl

# 查看全部日志
cat workdir/agent_log.jsonl
```

日志事件类型包括：`session_start`、`llm_request`、`llm_response`、`tool_call`、`tool_result`、`error`、`session_end`。

## 运行测试

```bash
# Rust 测试
cd benchmark && cargo test && cd ..
cd agent && cargo test && cd ..

# Python 测试
python3 -m pytest scripts/test_convert_data.py scripts/test_generate_ground_truth.py -v
```

## 并行测试多个模型

在多核服务器上可以同时测试多个模型，通过 `CPU_CORES` 为每个 agent 分配不同的 CPU 核心组，避免互相抢占：

```bash
# Agent A: 测试 GPT-4o，绑定 core 0-3
MODEL_NAME=gpt-4o CPU_CORES=0-3 WORK_DIR=./leaderboard/gpt-4o \
  API_URL=... API_KEY=... MODEL_ID=gpt-4o \
  bash scripts/run_eval.sh &

# Agent B: 测试 Claude，绑定 core 4-7
MODEL_NAME=claude-sonnet CPU_CORES=4-7 WORK_DIR=./leaderboard/claude-sonnet \
  API_URL=... API_KEY=... MODEL_ID=claude-sonnet-4-20250514 \
  bash scripts/run_eval.sh &

# Agent C: 测试 Qwen（带限速和思考模式），绑定 core 8-11
MODEL_NAME=qwen3.5-plus CPU_CORES=8-11 THINKING_MODE=openai API_INTERVAL_MS=2000 \
  WORK_DIR=./leaderboard/qwen3.5-plus \
  API_URL=... API_KEY=... MODEL_ID=qwen-plus \
  bash scripts/run_eval.sh &
```

注意事项：
- 每个 agent 必须使用不同的 `WORK_DIR`
- 核心组不要重叠，否则会互相影响 QPS 结果
- 建议同一 NUMA node 内分配（避免跨 NUMA 内存访问）
- 端口已自动随机分配，无需手动指定
- 设置 `CPU_CORES=""` 可禁用 CPU 绑定（不推荐用于正式评测）

## 常见问题

**Q: ground truth 生成太慢？**
安装 numpy 可以显著加速：`pip install numpy`。还可以使用 `--workers N` 参数启用多进程并行（N 为 CPU 核心数），例如 `python3 scripts/generate_ground_truth.py --workers 8`。处理过程分块落盘（默认每块 1000 条查询），中断后重新运行会自动跳过已完成的分块。可通过 `--chunk-size` 调整分块大小。

**Q: profiling 工具不可用？**
`run_profiling` 依赖 Linux 的 `perf` 工具和 FlameGraph 脚本。如果不在 Linux 上运行或未安装这些工具，profiling 会返回错误，但不影响其他功能。

**Q: 如何对比多个模型？**
多次运行 `run_eval.sh`，每次设置不同的 `MODEL_NAME` 和 `MODEL_ID`，结果会自动追加到同一个 `leaderboard.json`。

**Q: 如何自定义 recall 阈值？**
修改 `agent/src/bench_tools.rs` 中的 `RECALL_THRESHOLD` 常量，或在手动运行 benchmark 时通过 `--recall-threshold` 参数指定。

**Q: 模型的 tool call 用完会怎样？**
Agent 框架会自动触发最终 benchmark 并结束评测。如果此时代码恰好处于编译失败状态，Agent 会自动回退使用运行期间记录的最佳 QPS 结果（`best_benchmark`）。最佳实现的源码也已备份在 `src_best_qps/` 目录中。默认 50 次 tool call，可通过 `--max-tool-calls` 参数调整。

**Q: 排行榜数据不对怎么办？**
运行 `bash scripts/run_eval.sh --rebuild-leaderboard` 或 `python3 scripts/rebuild_leaderboard.py` 从 `leaderboard/` 目录重新扫描所有条目并重建排行榜。
