from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE_TEXT = """
Python is a high-level programming language. It was created by Guido van Rossum.
Python emphasizes code readability and simplicity.

FastAPI is a modern web framework for Python. It is based on standard Python type hints.
FastAPI is one of the fastest Python frameworks available.

Ollama allows running large language models locally. It supports many open source
models.
Models like Llama, Qwen, and Gemma can run on consumer hardware.
"""


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)
