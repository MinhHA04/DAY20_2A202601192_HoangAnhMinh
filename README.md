# Lab 20: Multi-Agent Research System

Hệ thống nghiên cứu cho bài lab **Multi-Agent Systems**, gồm **Supervisor + Researcher + Analyst + Writer + citation Critic** và benchmark paired với single-agent baseline.

Repo đã hoàn thiện cả chế độ OpenAI thật và deterministic offline fallback từ corpus đi kèm. Mọi run giữ source provenance, route, timed span, lỗi fallback và token usage trong shared state.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |------> Critic            -> citation audit
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + implementations
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow + baseline
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark metrics + report
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Routing, workflow, report, config tests
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev,llm]"
cp .env.example .env       # Windows: Copy-Item .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết.

```bash
OPENAI_API_KEY=...
# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

`OPENAI_API_KEY` là optional nếu chạy `--offline`. Retriever mặc định đọc các source card có
provenance từ `ai_agent_offline_research_corpus_v2`; không cần Internet. Khi có LangSmith key,
tracing online được bật tự động; JSON trace cục bộ vẫn luôn có trong state.

### 3. Chạy smoke test

```bash
pytest
ruff check src tests
mypy src
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy single-agent baseline

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Baseline dùng một model call để làm toàn bộ research interpretation, analysis và writing từ cùng
source ledger. Chạy deterministic bằng cách thêm `--offline`.

### 5. Chạy multi-agent workflow

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Route mặc định là `researcher -> analyst -> writer -> done`, sau đó Critic audit citation IDs.

### 6. Chạy benchmark và xuất deliverables

```bash
python -m multi_agent_research_lab.cli benchmark
# Không gọi API, phù hợp CI/reproduction:
python -m multi_agent_research_lab.cli benchmark --offline
```

Lệnh tạo `reports/benchmark_report.md`, `reports/benchmark_metrics.json` và JSON state/trace của
từng run trong `reports/traces/`.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Phần đã triển khai

1. OpenAI Responses API client có timeout/retry/token logging.
2. Retriever corpus offline có ranking, provenance và nhãn synthetic.
3. Supervisor policy deterministic, max iterations và LangGraph recursion guard.
4. Researcher, Analyst, Writer và bonus citation Critic.
5. Conditional LangGraph workflow và single-agent baseline dùng cùng nguồn/model.
6. LangSmith optional + portable JSON spans.
7. Benchmark/report có latency, tokens, structural quality heuristic, citation coverage và failure.

## Deliverables

Học viên nộp:

1. GitHub repo cá nhân.
2. Screenshot LangSmith trace hoặc JSON trace trong `reports/traces/`.
3. `reports/benchmark_report.md` so sánh single vs multi-agent.
4. Một đoạn giải thích failure mode và cách fix.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
