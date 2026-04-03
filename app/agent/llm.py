"""LLM factory — wraps Anthropic via custom base URL."""
from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from app.core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.llm_api_key,
        base_url=settings.llm_api_url,
        temperature=temperature,
        max_tokens=4096,
    )
