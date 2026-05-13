from langfuse import Langfuse

from llm_tools_lab.config import settings

_langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_base_url,
)


def get_langfuse() -> Langfuse:
    return _langfuse
