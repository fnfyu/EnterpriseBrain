from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    llm_api_url: str = "https://xingjiabiapi.org"
    llm_api_key: str
    llm_model: str = "claude-sonnet-4-6"

    # PostgreSQL
    database_url: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "enterprise_brain"
    postgres_user: str = "postgres"
    postgres_password: str

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-brain"
    langchain_tracing_v2: bool = True

    # LlamaParse
    llama_cloud_api_key: str = ""

    # E2B
    e2b_api_key: str = ""

    # Auth
    secret_key: str
    access_token_expire_minutes: int = 60

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
