# Cortex — Enterprise Multi-Agent RAG System

## 1) What Cortex Does

At runtime, Cortex follows one of four primary paths plus memory enrichment:

1. **Supervisor path**
   - Receives the user query and classifies intent.
   - Routes to one of:
     - Knowledge Agent
     - Action Agent
     - Research Agent
     - Unsupported safe handler

2. **Knowledge Agent path**
   - Answers questions from internal documents via the RAG pipeline.
   - Applies RBAC/access filtering before retrieval and generation.
   - Produces grounded responses with citations, used chunks, and retrieval trace metadata.

3. **Action Agent path**
   - Handles structured workflow actions (e.g., create ticket, ticket status, escalate, notify, software request).
   - Uses structured tool contracts/schemas.
   - Passes through runtime reliability and cost controls.

4. **Research Agent path**
   - Handles open-ended or external research queries.
   - Uses a `web_search` style tool interface.
   - Supports offline/live/fallback behavior based on environment and runtime availability.

5. **Memory enrichment (cross-cutting)**
   - Session memory stores recent conversation context (Redis when enabled, with fallback behavior).
   - Entity memory stores longer-lived user/entity facts (PostgreSQL when enabled, with fallback behavior).
   - Memory context can be injected into downstream agent behavior.

## 2) Architecture Diagram
![alt text](<Screenshot 2026-05-27 at 12.58.06 AM.png>)
<img width="696" height="356" alt="Screenshot 2026-05-27 at 2 04 18 AM" src="https://github.com/user-attachments/assets/c53efa80-b244-4a97-981d-0a4147c65742" />


## 3) RAG Pipeline Diagram
![alt text](<Screenshot 2026-05-27 at 1.00.31 AM.png>)

**Important:** ingestion is an explicit step. It does **not** run automatically on every query.

## 4) Memory Architecture

- **Session memory**
  - Tracks short-term conversational context.
  - Uses Redis when configured.
  - Falls back gracefully when Redis is unavailable.

- **Entity memory**
  - Stores longer-term structured facts.
  - Uses PostgreSQL when configured.
  - Falls back gracefully when PostgreSQL entity persistence is unavailable.

- **Memory context usage**
  - Supervisor and downstream agents can consume memory context.
  - UI surfaces memory summary/debug fields for observability.

## 5) Tools and Reliability Layer

- **Tool layer**
  - Tools are modeled with structured contracts/schemas.
  - Covers knowledge, action, research, and memory helper operations.

- **Reliability layer**
  - Rate limiting
  - Retries with exponential backoff
  - Circuit breaker behavior
  - Cost tracking / budget signaling

This layer is applied to control failure handling and improve predictable system behavior under degraded conditions.

## 6) Setup Instructions

### Prerequisites

- Python 3.11+
- Docker (for Redis/PostgreSQL/PGVector services)

### Quick start (default local mode)

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

This mode works with local defaults/fallbacks and does not require enabling OpenAI features.

### Optional: PGVector + OpenAI mode

```bash
docker-compose up -d
cp .env.example .env
# then edit .env to enable PostgreSQL/PGVector and OpenAI-backed settings
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

OpenAI usage is **optional** and only active when its corresponding environment variables/settings are enabled.

## 7) Docker / Infrastructure Commands

```bash
# start infrastructure
docker-compose up -d

# view status
docker-compose ps

# tail logs
docker-compose logs -f redis
docker-compose logs -f pgvector

# stop infrastructure
docker-compose down
```

## 8) Environment Variables

Use the project `.env.example` as the source of truth and keep names aligned exactly with code and config expectations.

Common categories:

- **Core app/runtime settings**
- **Redis/session-memory settings**
- **PostgreSQL/PGVector settings**
- **OpenAI settings (optional)**
- **Retrieval/runtime tuning values**

OpenAI is optional by default; do not set OpenAI variables unless you want LLM/embedding-backed behaviors enabled.

## 9) Ingestion Workflow

Ingestion is explicit and should be run when documents change or when setting up a fresh environment.

Typical flow:

1. Load source documents from the repository dataset.
2. Register/track ingest metadata in the ingestion registry.
3. Perform section-aware chunking.
4. Build/update lexical and vector indexes.
5. Run queries against the indexed corpus.

Because ingestion is explicit, query latency is decoupled from document processing.

## 10) Running the Streamlit UI
```bash
streamlit run ui/app.py
```

The UI includes:

- Supervisor routing summary
- Agent execution path
- RAG retrieval and ranking inspection
- Grounding/citation views
- Memory context inspection
- Evaluation tab for retrieval metrics

## 11) Running Evaluation

```bash
python -m evaluation.evaluate
```

The evaluation module reports retrieval quality metrics including:

- Precision@K (P@K)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

## 12) Example Queries

- **Knowledge**: `What is the parental leave policy?`
- **Knowledge (restricted domain intent)**: `What is the executive strategy for APAC expansion?`
- **Action**: `Create a support ticket for VPN access issues.`
- **Action**: `Escalate ticket INC-1042 as production-blocking.`
- **Research**: `Find recent AI governance trends relevant to enterprise risk.`
- **Unsupported-safe route**: `Write a poem about office chairs in iambic pentameter.`

## 13) Test Commands

```bash
pytest -q
```

Optional targeted runs:

```bash
pytest tests/test_supervisor.py -q
pytest tests/test_rag.py -q
pytest tests/test_reliability.py -q
pytest tests/test_memory.py -q
pytest tests/test_evaluation.py -q
```
