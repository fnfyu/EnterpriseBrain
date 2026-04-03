# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Enterprise Brain (企业智脑)** — An enterprise AI assistant platform enabling employees to query company knowledge and execute business processes via natural language. Target: medium-to-large enterprises (500+ people), especially knowledge-intensive industries (law, consulting, manufacturing, pharma).

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet 4.6 via `xingjiabiapi.org` |
| Orchestration | LangGraph (StateGraph) + LangChain |
| Embedding | BGE-M3 (open source, bilingual CN/EN, 8k ctx) |
| Vector DB | PGVector |
| Graph DB | Neo4j |
| Relational DB | PostgreSQL |
| Cache / Session | Redis |
| API Framework | FastAPI (async) |
| Observability | LangSmith |
| Doc Parsing | LlamaParse + Unstructured |
| Code Sandbox | E2B |
| Deployment | Docker + Kubernetes |

## Architecture

Four layers:

1. **User Interaction** — Web chat UI, WeChat Work bot, DingTalk bot, embedded JS SDK
2. **API Gateway (FastAPI)** — Auth, rate limiting, logging, routing, load balancing
3. **Intelligent Orchestration (LangGraph)** — Core "brain"; routes between three execution paths:
   - `rag_pipeline` — knowledge queries via Advanced RAG
   - `agent_executor` — task execution via ReAct + Tool Calling
   - `human_handoff` — escalation when intent is unclear
4. **Support Layers** — Knowledge Retrieval, Tool Execution, Data Storage, LangSmith observability

### LangGraph State

The central `AgentState` TypedDict carries: `messages`, `user_id`, `intent`, `retrieved_docs`, `tool_calls`, `final_response`, `needs_human`.

Graph entry point: `intent_classifier` node → conditional routing via `route_by_intent` → one of the three execution nodes → `response_generator`.

### Advanced RAG Pipeline

Multi-path retrieval with RRF fusion:
1. Query rewriting (HyDE / multi-query expansion)
2. Parallel retrieval: vector search (BGE-M3) + BM25 keyword search + graph search (Neo4j entity relations)
3. Reciprocal Rank Fusion (RRF) for result merging
4. Cross-Encoder reranking (BGE-Reranker), returns Top-K

### Memory Architecture (Three-Tier)

- **Short-term**: `RedisChatMessageHistory` — current session messages
- **Mid-term**: `PostgresCheckpointer` — cross-session recent history
- **Long-term**: `VectorStoreRetriever` — persistent facts and user preferences

### Agent Tools (ReAct Pattern)

Tool categories: data queries (SQL generation), system operations (internal API calls), content generation (templates + LLM), external services (third-party APIs), code execution (E2B sandbox).

## Development Setup

```bash
# Environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your secrets

# Start all backing services (Postgres+pgvector, Redis, Neo4j)
docker-compose up -d postgres redis neo4j

# Run FastAPI dev server (auto-creates DB tables on startup)
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test file
pytest tests/test_rag.py -v

# Run a single test
pytest tests/test_rag.py::test_rrf_merges_and_deduplicates -v

# Run everything via Docker
docker-compose up --build
```

## Project Structure

```
app/
  main.py              # FastAPI app, lifespan, middleware, router registration
  core/
    config.py          # Pydantic settings (reads .env)
    database.py        # SQLAlchemy async engine, ORM models, init_db()
    graph_db.py        # Neo4j async driver, GraphStore
    redis_client.py    # Redis async client singleton
    security.py        # JWT + bcrypt
    logging.py         # structlog setup
  rag/
    parser.py          # Document parsing (LlamaParse / Unstructured)
    embedder.py        # BGE-M3 singleton, embed_texts / embed_query
    vector_store.py    # PGVector add/search
    keyword_search.py  # BM25 in-memory index
    reranker.py        # RRF fusion + BGE-Reranker cross-encoder
    pipeline.py        # LangGraph RAG node (parallel retrieval → RRF → rerank)
    ingestor.py        # parse → embed → store pipeline for file upload
  agent/
    state.py           # AgentState TypedDict
    llm.py             # ChatAnthropic factory (custom base_url)
    nodes.py           # All LangGraph node functions
    graph.py           # StateGraph assembly and compilation
  memory/
    manager.py         # MemoryManager: Redis short-term, PGVector long-term, Postgres profile
  tools/
    registry.py        # All @tool definitions + ALL_TOOLS list
  api/
    schemas.py         # Pydantic request/response models
    deps.py            # JWT auth dependency
    auth.py            # /auth/register, /auth/token
    chat.py            # /chat (JSON + SSE streaming)
    knowledge.py       # /knowledge/upload
tests/
  test_api.py
  test_rag.py
  test_security.py
```

## Key Design Constraints

- All retrieval is **async** (`asyncio.gather` for parallel multi-path search)
- LLM calls use the custom endpoint `xingjiabiapi.org` — configure via environment variable, not hardcoded
- LangSmith tracing must be enabled in all environments for observability
- The ReAct agent loop is built with `langgraph.prebuilt.create_react_agent` + `ToolNode`
- Document parsing must handle PDF, Word, Excel, PPT, and scanned images
