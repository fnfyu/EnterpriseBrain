"""
LangGraph nodes:
  - intent_classifier
  - rag_node
  - agent_node
  - human_handoff_node
  - response_generator
"""
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from app.agent.state import AgentState
from app.agent.llm import get_llm
from app.tools.registry import ALL_TOOLS
from app.core.logging import get_logger

logger = get_logger(__name__)

_INTENT_SYSTEM = """你是一个意图分类器。根据用户最后一条消息，判断意图类型：
- knowledge_query：用户想查询知识、文档、政策、规范等信息
- task_execution：用户想执行操作，如查数据、创建工单、发邮件、生成报告等
- unclear：意图不清晰，需要人工确认

只输出以下三个词之一：knowledge_query | task_execution | unclear"""


async def intent_classifier_node(state: AgentState) -> Dict:
    llm = get_llm()
    last_msg = state["messages"][-1].content
    response = await llm.ainvoke([
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user", "content": last_msg},
    ])
    intent = response.content.strip().lower()
    if intent not in ("knowledge_query", "task_execution", "unclear"):
        intent = "unclear"
    logger.info("Intent classified", intent=intent, query=last_msg[:60])
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


async def rag_node(state: AgentState, db) -> Dict:
    from app.rag.pipeline import rag_pipeline_node
    llm = get_llm()
    return await rag_pipeline_node(state, db, llm)


_AGENT_SYSTEM = """你是企业智脑助手，能调用工具帮员工解决实际问题。
使用工具时要思考：用什么工具？输入是什么？结果如何利用？
如果任务完成，请给出清晰简洁的中文回复。"""


async def agent_executor_node(state: AgentState) -> Dict:
    llm = get_llm()
    agent = create_react_agent(llm, ALL_TOOLS, prompt=_AGENT_SYSTEM)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {
        "messages": result["messages"][len(state["messages"]):],  # only new messages
        "tool_calls": [
            m.tool_calls for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls
        ],
    }


async def human_handoff_node(state: AgentState) -> Dict:
    logger.info("Human handoff triggered", user_id=state["user_id"])
    return {
        "needs_human": True,
        "final_response": "您的问题需要人工协助，正在为您转接专员，请稍候。",
    }


async def response_generator_node(state: AgentState) -> Dict:
    llm = get_llm()

    # Build context from RAG docs if present
    context = ""
    if state.get("retrieved_docs"):
        docs_text = "\n\n".join(
            f"[{i+1}] {d['content']}" for i, d in enumerate(state["retrieved_docs"])
        )
        context = f"\n\n参考资料：\n{docs_text}"

    memory_ctx = state.get("memory_context", {})
    user_profile = memory_ctx.get("user_profile", {})
    profile_hint = f"（用户：{user_profile.get('name', '')}，部门：{user_profile.get('department', '')}）" if user_profile.get("name") else ""

    system_prompt = f"""你是企业智脑助手{profile_hint}。
根据以下参考资料和对话历史，用中文给出准确、简洁的回答。
如果引用了资料，请在末尾注明来源编号。{context}"""

    response = await llm.ainvoke(
        [{"role": "system", "content": system_prompt}] + list(state["messages"])
    )
    return {"final_response": response.content}
