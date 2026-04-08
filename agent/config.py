from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    tavily_api_key: str = ""

    # --- Memory constraints ---------------------------------------------------
    # Maximum research-context tokens allowed in any single LLM call.
    # System prompt / instructions are excluded from this budget.
    max_context_tokens: int = 2048

    # Maximum tokens the running draft answer may occupy before it is
    # compressed to make room for the next sub-query's findings.
    max_draft_tokens: int = 1200

    # How many vector-search results to retrieve per lookup.
    vector_top_k: int = 3

    # --- Model selection ------------------------------------------------------
    llm_model: str = "llama-3.3-70b-versatile"
    # Base URL for any OpenAI-compatible API (Groq, OpenRouter, Together, etc.)
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # --- Summarisation targets ------------------------------------------------
    # Target token count when compressing a single search-result page.
    source_summary_tokens: int = 200
    # Target token count when synthesising findings for one sub-query.
    subquery_summary_tokens: int = 300

    # --- Search ---------------------------------------------------------------
    max_search_results: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
