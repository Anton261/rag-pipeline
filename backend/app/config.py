from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Mistral AI
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-large-latest"
    mistral_fast_model: str = "mistral-small-latest"

    # Chunking
    chunk_size_tokens: int = 512   # 1 token ≈ 4 chars → ~2 048 chars max per chunk
    chunk_overlap_tokens: int = 64

    # Retrieval
    top_k: int = 8
    similarity_threshold: float = 0.80  # below this → "insufficient evidence"
    rrf_k: int = 60                      # Reciprocal Rank Fusion constant

    # Safety
    max_file_size_mb: int = 100
    max_query_length: int = 2000

    # Storage paths (relative to backend working directory)
    data_dir: Path = Path("data")
    embeddings_dir: Path = Path("data/embeddings")
    db_path: Path = Path("data/rag.db")
    bm25_index_path: Path = Path("data/bm25_index.json")

    # Rate limiting
    rate_limit: str = "30/minute"


settings = Settings()
