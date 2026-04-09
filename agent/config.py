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
    max_draft_tokens: int = 1800

    # How many vector-search results to retrieve per lookup.
    vector_top_k: int = 3

    # --- Model selection ------------------------------------------------------
    # Primary model — used for synthesis and complex reasoning.
    llm_model: str = "gpt-5-mini"

    # Set True when llm_model is an OpenAI reasoning model 
    llm_is_reasoning_model: bool = True

    # Summary model — used for decompose, source summarisation,
    # subquery merging, draft compression, and query classification.
    summary_llm_model: str = "gpt-4o-mini"

    # Max completion tokens for the synthesis (refine) step. Raise this for reasoning models.
    synthesis_max_tokens: int = 8000


    source_summary_tokens: int = 400
    subquery_summary_tokens: int = 400

    max_search_results: int = 2
    search_recency_days: int = 365

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
