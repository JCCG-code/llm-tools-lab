from pathlib import Path

from llm_tools_lab.rag.indexer import index_text


def file_indexer_text(filepath: str) -> int:
    # Extract path model
    path = Path(filepath)
    # Path exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    # File suffix
    if path.suffix != ".txt":
        raise ValueError(f"Only .txt files are supported, got: {path.suffix}")
    # Index into database
    return index_text(path.read_text(encoding="utf-8"), path.name)
