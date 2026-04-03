# 企业智脑 Enterprise Brain

> 让员工用自然语言获取企业知识、执行业务流程。

面向中大型企业（500人以上）的智能 AI 助手平台，知识密集型行业优先（律所、咨询、制造、医药）。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Web聊天   │  │ 企业微信  │  │  钉钉    │  │ 嵌入业务系统   │  │
│  │ 界面     │  │ 机器人   │  │ 机器人   │  │  (JS SDK)      │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API 网关层（FastAPI）                          │
│          认证鉴权 │ 限流 │ 日志 │ 请求路由 │ 负载均衡             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   智能编排层（LangGraph）                         │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐       │
│  │  意图路由器  │──►│  执行引擎    │◄──│   记忆管理器     │       │
│  │  (Entry)    │   │  (Nodes)    │   │  (State: user)  │       │
│  └─────────────┘   └──────┬──────┘   └─────────────────┘       │
│                           │                                     │
│             ┌─────────────┼─────────────┐                      │
│             ▼             ▼             ▼                      │
│        ┌─────────┐  ┌──────────┐  ┌───────────┐               │
│        │ RAG查询 │  │ Agent执行 │  │ 人工介入   │               │
│        │  节点   │  │   节点   │  │   节点    │               │
│        └─────────┘  └──────────┘  └───────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   知识检索层     │  │   工具执行层     │  │   数据存储层     │
│  (LangChain)    │  │  (LangChain)    │  │                 │
│                 │  │                 │  │  ┌───────────┐  │
│ ┌───────────┐   │  │ ┌───────────┐   │  │  │ PGVector  │  │
│ │ 文档解析   │   │  │ │ 内部API   │   │  │  └───────────┘  │
│ │(LlamaParse│   │  │ │ 数据库    │   │  │  ┌───────────┐  │
│ │  +Unstr.) │   │  │ │ 代码执行  │   │  │  │  Neo4j    │  │
│ └───────────┘   │  │ └───────────┘   │  │  └───────────┘  │
│ ┌───────────┐   │  │ ┌───────────┐   │  │  ┌───────────┐  │
│ │ 多路检索   │   │  │ │ 搜索引擎   │   │  │  │PostgreSQL │  │
│ │(向量+BM25 │   │  │ │ 邮件系统   │   │  │  └───────────┘  │
│ │  +图谱)   │   │  │ └───────────┘   │  │  ┌───────────┐  │
│ └───────────┘   │  │                 │  │  │   Redis   │  │
│ ┌───────────┐   │  │                 │  │  └───────────┘  │
│ │ RRF+重排序 │   │  │                 │  │                 │
│ └───────────┘   │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   可观测性层（LangSmith）                        │
│       调用追踪 │ 性能监控 │ 效果评估 │ Prompt版本管理 │ A/B测试   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 技术选型

| 层级 | 技术 |
|------|------|
| **LLM** | Claude Sonnet 4.6（`xingjiabiapi.org`） |
| **编排** | LangGraph + LangChain |
| **Embedding** | BGE-M3（开源，中英双语，1024维，8k上下文） |
| **向量数据库** | PGVector（PostgreSQL 扩展） |
| **图数据库** | Neo4j |
| **关系数据库** | PostgreSQL |
| **缓存/会话** | Redis |
| **API 框架** | FastAPI（异步） |
| **文档解析** | LlamaParse + Unstructured |
| **代码沙箱** | E2B |
| **可观测性** | LangSmith |
| **部署** | Docker + Kubernetes |

---

## 项目结构

```
app/
  main.py               # FastAPI 入口，lifespan、中间件、路由注册
  core/
    config.py           # Pydantic 配置（读取 .env）
    database.py         # SQLAlchemy 异步引擎 + ORM 模型 + PGVector
    graph_db.py         # Neo4j 异步驱动 + GraphStore
    redis_client.py     # Redis 异步客户端单例
    security.py         # JWT + bcrypt
    logging.py          # structlog 结构化日志
  rag/
    parser.py           # 文档解析（LlamaParse / Unstructured）
    embedder.py         # BGE-M3 向量化
    vector_store.py     # PGVector 存取
    keyword_search.py   # BM25 关键词检索（内存索引）
    reranker.py         # RRF 融合 + BGE-Reranker 精排
    pipeline.py         # LangGraph RAG 节点
    ingestor.py         # 文件入库流水线
  agent/
    state.py            # AgentState TypedDict
    llm.py              # ChatAnthropic 工厂（自定义 base_url）
    nodes.py            # 所有 LangGraph 节点函数
    graph.py            # StateGraph 组装与编译
  memory/
    manager.py          # 三层记忆管理器
  tools/
    registry.py         # 所有 @tool 定义 + ALL_TOOLS 列表
  api/
    schemas.py          # 请求/响应 Pydantic 模型
    deps.py             # JWT 鉴权依赖
    auth.py             # /auth/register, /auth/token
    chat.py             # /chat（JSON + SSE 流式）
    knowledge.py        # /knowledge/upload
tests/
  test_api.py
  test_rag.py
  test_security.py
Dockerfile
docker-compose.yml      # Postgres+PGVector, Redis, Neo4j, App
requirements.txt
.env.example
```

---

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写以下必填项：
# LLM_API_KEY, POSTGRES_PASSWORD, NEO4J_PASSWORD, SECRET_KEY
```

### 2. 启动依赖服务

```bash
docker-compose up -d postgres redis neo4j
```

### 3. 启动应用

```bash
# 本地开发
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 或全部 Docker 启动
docker-compose up --build
```

应用启动后自动创建数据库表并初始化 Neo4j schema。

API 文档：http://localhost:8000/docs

### 4. 运行测试

```bash
pytest                                         # 全部测试
pytest tests/test_rag.py -v                    # 单个文件
pytest tests/test_rag.py::test_rrf_merges_and_deduplicates -v  # 单个测试
```

---

## 核心模块详解

### 智能编排层（LangGraph）

`AgentState` 贯穿整个流程：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    session_id: str
    intent: str           # knowledge_query | task_execution | unclear
    retrieved_docs: list
    tool_calls: list
    final_response: str
    needs_human: bool
    memory_context: dict
```

路由逻辑：`intent_classifier` → `route_by_intent` → `rag_pipeline` / `agent_executor` / `human_handoff` → `response_generator`

### 知识检索层（Advanced RAG）

三路并行检索 → RRF 融合 → Cross-Encoder 精排：

```python
vector_results, keyword_results, graph_results = await asyncio.gather(
    vector_search(db, query),      # BGE-M3 语义检索
    keyword_search(db, query),     # BM25 关键词检索
    _graph_search(query),          # Neo4j 实体关系检索
)
fused = reciprocal_rank_fusion([vector_results, keyword_results, graph_results])
reranked = rerank(query, fused, top_k=5)   # BGE-Reranker
```

### 记忆管理层（三层）

| 层级 | 存储 | 内容 |
|------|------|------|
| 短期 | Redis（TTL 24h） | 当前会话消息 |
| 中期 | PostgresCheckpointer | 跨会话历史（LangGraph） |
| 长期 | PGVector | 用户事实、偏好向量 |

### 工具执行层（ReAct + Tool Calling）

| 工具 | 说明 |
|------|------|
| `query_database` | 只读 SQL 查询 |
| `create_ticket` | 创建工单 |
| `call_internal_api` | 调用内部 API |
| `generate_report` | 生成 Markdown 报告 |
| `send_email` | 发送企业邮件 |
| `web_search` | 公网搜索（DuckDuckGo） |
| `execute_python_code` | E2B 沙箱代码执行 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/auth/register` | 注册用户 |
| `POST` | `/auth/token` | 登录获取 JWT |
| `POST` | `/chat` | 对话（支持 `stream: true` SSE） |
| `POST` | `/knowledge/upload` | 上传知识文档 |
| `GET`  | `/health` | 健康检查 |

所有 `/chat` 和 `/knowledge` 接口需在 Header 携带 `Authorization: Bearer <token>`。

---

## 数据流示例

**员工问：** "帮我查一下张三上季度的销售额，并生成一份对比报告"

```
意图识别 → task_execution
    │
    ▼
Agent 执行循环（LangGraph ReAct）
    Step 1: 调用 query_database → 查询张三 Q3 销售数据 → 500万
    Step 2: 调用 query_database → 查询张三 Q2 销售数据 → 420万
    Step 3: 调用 generate_report → 生成 Markdown 对比报告
    Step 4: 任务完成，输出最终回复
    │
    ▼
response_generator → 返回带数据来源的中文回复
    │
    ▼
LangSmith 记录完整调用链
```
