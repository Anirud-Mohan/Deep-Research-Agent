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
    # Primary (large) model — used for synthesis and complex reasoning.
    # OpenAI: "gpt-4o" | "gpt-4o-mini" | "gpt-5-mini"
    # Groq:   "llama-3.3-70b-versatile"
    llm_model: str = "gpt-5-mini"
    # Leave empty for OpenAI. For Groq/OpenRouter set the base URL.
    # Groq: "https://api.groq.com/openai/v1"
    llm_base_url: str = ""

    # Set True when llm_model is an OpenAI reasoning model (o1, o3, o4,
    # gpt-5 series). Reasoning models do not accept `temperature` and require
    # `max_completion_tokens` instead of `max_tokens`.
    llm_is_reasoning_model: bool = True

    # Summary (small) model — used for decompose, source summarisation,
    # subquery merging, draft compression, and query classification.
    # OpenAI: "gpt-4o-mini"  (cheap and fast, same key)
    # Groq:   "llama-3.1-8b-instant" (500K TPD free)
    summary_llm_model: str = "gpt-4o-mini"

    # Max completion tokens for the synthesis (refine) step.
    # Raise this for reasoning models — their internal chain-of-thought
    # tokens count against this budget before visible output is produced.
    synthesis_max_tokens: int = 8000

    # --- Summarisation targets ------------------------------------------------
    # Target token count when compressing a single search-result page.
    source_summary_tokens: int = 200
    # Target token count when synthesising findings for one sub-query.
    subquery_summary_tokens: int = 400

    # --- Search ---------------------------------------------------------------
    max_search_results: int = 2
    search_recency_days: int = 365

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
