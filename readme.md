&#x20;

\## 系统定位



\*\*产品名称\*\*：企业智脑（Enterprise Brain）  

\*\*核心价值\*\*：让员工用自然语言获取企业知识、执行业务流程  

\*\*目标用户\*\*：中大型企业（500人以上），知识密集型企业优先（律所、咨询、制造、医药）



\---



\## 整体架构



```

┌─────────────────────────────────────────────────────────────────┐

│                         用户交互层                               │

│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │

│  │ Web聊天   │  │ 企业微信  │  │ 钉钉     │  │ 嵌入业务系统    │  │

│  │ 界面     │  │ 机器人   │  │ 机器人   │  │ (JS SDK)       │  │

│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │

└─────────────────────────────────────────────────────────────────┘

&#x20;                             │

&#x20;                             ▼

┌─────────────────────────────────────────────────────────────────┐

│                      API网关层（FastAPI）                        │

│           认证鉴权 │ 限流 │ 日志 │ 请求路由 │ 负载均衡            │

└─────────────────────────────────────────────────────────────────┘

&#x20;                             │

&#x20;                             ▼

┌─────────────────────────────────────────────────────────────────┐

│                     智能编排层（LangGraph）                        │

│                                                                 │

│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │

│   │  意图路由器  │───►│  执行引擎    │◄───│   记忆管理器     │   │

│   │  (Entry)    │    │  (Nodes)    │    │  (State: user)  │   │

│   └─────────────┘    └──────┬──────┘    └─────────────────┘   │

│                             │                                   │

│              ┌──────────────┼──────────────┐                   │

│              ▼              ▼              ▼                   │

│        ┌─────────┐   ┌──────────┐   ┌───────────┐              │

│        │ RAG查询 │   │ Agent执行 │   │ 人工介入   │              │

│        │  节点   │   │   节点   │   │   节点    │              │

│        └─────────┘   └──────────┘   └───────────┘              │

│                                                                 │

└─────────────────────────────────────────────────────────────────┘

&#x20;                             │

&#x20;             ┌───────────────┼───────────────┐

&#x20;             ▼               ▼               ▼

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐

│   知识检索层     │  │   工具执行层     │  │   数据存储层     │

│  (LangChain)    │  │  (LangChain)    │  │                 │

│                 │  │                 │  │  ┌───────────┐  │

│ ┌───────────┐  │  │ ┌───────────┐  │  │  │ 向量数据库 │  │

│ │ 文档解析   │  │  │ │ 内部API   │  │  │  │(Chroma/  │  │

│ │(Unstructured│  │  │ │ 数据库    │  │  │  │PGVector) │  │

│ │  LlamaParse)│  │  │ │ 代码执行  │  │  │  └───────────┘  │

│ └───────────┘  │  │ └───────────┘  │  │                 │

│ ┌───────────┐  │  │ ┌───────────┐  │  │  ┌───────────┐  │

│ │ 多路检索   │  │  │ │ 搜索引擎   │  │  │  │ 图数据库   │  │

│ │(向量+关键词)│  │  │ │ 邮件系统   │  │  │  │(Neo4j/   │  │

│ └───────────┘  │  │ └───────────┘  │  │  │FalkorDB)  │  │

│ ┌───────────┐  │  │                │  │  └───────────┘  │

│ │ 重排序    │  │  │                │  │                 │

│ │(Cross-Encoder│ │  │                │  │  ┌───────────┐  │

│ └───────────┘  │  │                │  │  │ 关系数据库  │  │

│                 │  │                │  │  │(PostgreSQL)│  │

└─────────────────┘  └─────────────────┘  └─────────────────┘

&#x20;                             │

&#x20;                             ▼

┌─────────────────────────────────────────────────────────────────┐

│                    可观测性层（LangSmith）                       │

│        调用追踪 │ 性能监控 │ 效果评估 │ Prompt版本管理 │ A/B测试   │

└─────────────────────────────────────────────────────────────────┘

```



\---



\## 核心模块详解



\### 1. 智能编排层（LangGraph）



这是系统的大脑，负责决策和流程控制。



```python

\# 简化的 State 定义

from typing import TypedDict, Annotated, Sequence

from langchain\_core.messages import BaseMessage

import operator



class AgentState(TypedDict):

&#x20;   messages: Annotated\[Sequence\[BaseMessage], operator.add]  # 对话历史

&#x20;   user\_id: str                                              # 用户标识

&#x20;   intent: str                                               # 识别意图：rag/agent/human

&#x20;   retrieved\_docs: list                                      # 检索到的文档

&#x20;   tool\_calls: list                                          # 需要调用的工具

&#x20;   final\_response: str                                        # 最终回复

&#x20;   needs\_human: bool                                         # 是否需要人工介入



\# Graph 结构

graph = StateGraph(AgentState)



\# 节点定义

graph.add\_node("intent\_classifier", classify\_intent)      # 意图识别

graph.add\_node("rag\_pipeline", rag\_node)                  # RAG 查询

graph.add\_node("agent\_executor", agent\_node)              # Agent 执行

graph.add\_node("human\_handoff", human\_node)               # 人工介入

graph.add\_node("response\_generator", generate\_response)   # 生成回复



\# 边定义（条件路由）

graph.add\_conditional\_edges(

&#x20;   "intent\_classifier",

&#x20;   route\_by\_intent,

&#x20;   {

&#x20;       "knowledge\_query": "rag\_pipeline",

&#x20;       "task\_execution": "agent\_executor", 

&#x20;       "unclear": "human\_handoff"

&#x20;   }

)

```



\### 2. 知识检索层（Advanced RAG）



超越简单向量检索，实现企业级精度。



| 组件 | 技术选型 | 作用 |

|------|----------|------|

| \*\*文档解析\*\* | LlamaParse / Unstructured / 自研 | 处理 PDF、Word、Excel、PPT、扫描件 |

| \*\*文本分割\*\* | 语义分块 + 层次结构保留 | 按段落/章节分割，保留标题层级 |

| \*\*向量模型\*\* | BGE-M3 / OpenAI text-embedding-3 | 中英双语，支持 8k 长度 |

| \*\*向量数据库\*\* | PGVector / Milvus | 存储向量 + 元数据过滤 |

| \*\*图数据库\*\* | Neo4j / FalkorDB | 存储实体关系，支持 GraphRAG |

| \*\*重排序\*\* | BGE-Reranker / Cross-Encoder | 精排 Top-K 结果 |

| \*\*查询优化\*\* | HyDE / 查询重写 / 多查询扩展 | 提升召回率 |



\*\*检索流程\*\*：



```python

\# 多路召回 + 融合排序

async def retrieve(state: AgentState):

&#x20;   query = state\["messages"]\[-1].content

&#x20;   

&#x20;   # 1. 查询理解与重写

&#x20;   rewritten\_queries = await rewrite\_query(query, state\["messages"])

&#x20;   

&#x20;   # 2. 多路并行检索

&#x20;   results = await asyncio.gather(

&#x20;       vector\_search(rewritten\_queries),      # 语义检索

&#x20;       keyword\_search(rewritten\_queries),     # 关键词检索（BM25）

&#x20;       graph\_search(extract\_entities(query))   # 图检索（实体关系）

&#x20;   )

&#x20;   

&#x20;   # 3. 结果融合与去重

&#x20;   fused\_results = reciprocal\_rank\_fusion(results)

&#x20;   

&#x20;   # 4. 重排序

&#x20;   reranked = await reranker.rerank(query, fused\_results, top\_k=5)

&#x20;   

&#x20;   return {"retrieved\_docs": reranked}

```



\### 3. 工具执行层（Agent）



让系统能"动手"解决问题。



\*\*工具分类\*\*：



| 类型 | 示例 | 实现方式 |

|------|------|----------|

| \*\*数据查询\*\* | 查订单、查库存、查客户信息 | SQL 生成 + 数据库连接 |

| \*\*系统操作\*\* | 创建工单、审批流程、预约会议 | 内部 API 调用 |

| \*\*内容生成\*\* | 生成报告、写邮件、做 PPT | 模板 + LLM 生成 |

| \*\*外部服务\*\* | 查天气、查快递、发短信 | 第三方 API |

| \*\*代码执行\*\* | 数据分析、图表生成 | 沙箱环境（E2B）|



\*\*Agent 设计（ReAct + Tool Calling）\*\*：



```python

from langchain\_core.tools import tool

from langgraph.prebuilt import ToolNode



@tool

def query\_customer\_db(customer\_id: str) -> str:

&#x20;   """查询客户信息，输入客户ID"""

&#x20;   # 实际实现...

&#x20;   pass



@tool

def create\_ticket(title: str, description: str, priority: str) -> str:

&#x20;   """创建工单，输入标题、描述、优先级"""

&#x20;   # 实际实现...

&#x20;   pass



tools = \[query\_customer\_db, create\_ticket, search\_knowledge, send\_email]



\# Agent 节点

agent = create\_react\_agent(llm, tools, state\_modifier=system\_prompt)

```



\### 4. 记忆管理层



支持长期记忆，让系统越用越懂用户。



```python

\# 三层记忆架构

class MemoryManager:

&#x20;   def \_\_init\_\_(self):

&#x20;       self.short\_term = RedisChatMessageHistory()      # 当前会话

&#x20;       self.medium\_term = PostgresCheckpointer()       # 跨会话短期

&#x20;       self.long\_term = VectorStoreRetriever()          # 长期事实/偏好

&#x20;   

&#x20;   async def get\_context(self, user\_id: str, query: str):

&#x20;       # 1. 短期记忆：最近对话

&#x20;       recent = await self.short\_term.get\_messages(user\_id, last\_k=10)

&#x20;       

&#x20;       # 2. 长期记忆：相关事实

&#x20;       relevant\_facts = await self.long\_term.retrieve(

&#x20;           query, 

&#x20;           filter={"user\_id": user\_id, "type": "fact"}

&#x20;       )

&#x20;       

&#x20;       # 3. 用户画像

&#x20;       profile = await self.get\_user\_profile(user\_id)

&#x20;       

&#x20;       return {

&#x20;           "recent\_messages": recent,

&#x20;           "relevant\_facts": relevant\_facts,

&#x20;           "user\_profile": profile

&#x20;       }

```



\---



\## 数据流示例



\### 场景：员工问"帮我查一下张三上季度的销售额，并生成一份对比报告"



```

用户输入

&#x20;   │

&#x20;   ▼

┌─────────────────┐

│ 意图识别：task\_execution（需要查数据+生成报告）

└─────────────────┘

&#x20;   │

&#x20;   ▼

┌─────────────────────────────────────────┐

│ Agent 执行循环（LangGraph）              │

│                                         │

│ Step 1: 思考 → 需要调用 query\_sales\_db  │

│         行动 → 查询数据库（张三，Q3）      │

│         观察 → 获得数据：{sales: 500万}   │

│                                         │

│ Step 2: 思考 → 需要对比数据，查Q2         │

│         行动 → 查询数据库（张三，Q2）      │

│         观察 → 获得数据：{sales: 420万}   │

│                                         │

│ Step 3: 思考 → 需要生成报告               │

│         行动 → 调用 generate\_report 工具 │

│         观察 → 获得 Markdown 报告        │

│                                         │

│ Step 4: 思考 → 任务完成，生成回复         │

└─────────────────────────────────────────┘

&#x20;   │

&#x20;   ▼

生成回复（带报告内容 + 数据来源说明）

&#x20;   │

&#x20;   ▼

LangSmith 记录完整调用链（用于后续优化）

```

\---



\## 技术选型



| 层级 | 推荐方案 | 

|------|----------|

| \*\*LLM\*\* | Claude-Sonnet-4-6 api\_url:xingjiabiapi.org

| \*\*Embedding\*\* | BGE-M3（开源） 

| \*\*向量数据库\*\* | PGVector（简单）

| \*\*图数据库\*\* | Neo4j |

| \*\*缓存/消息\*\* | Redis | 

| \*\*应用框架\*\* | FastAPI + LangChain + LangGraph | 

| \*\*部署\*\* | Docker + K8s / 云函数 | 

| \*\*监控\*\* | \*\*LangSmith\*\*





