from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen3:8b"
    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore
