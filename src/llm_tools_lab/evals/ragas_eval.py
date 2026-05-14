from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from ragas import (
    EvaluationDataset,
    evaluate,
)
from ragas.dataset_schema import (
    EvaluationResult,
    SingleTurnSample,
    SingleTurnSampleOrMultiTurnSample,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from llm_tools_lab.config import settings
from llm_tools_lab.rag.rag_agent import text_rag_agent

# Initializations
_groq_client = OpenAI(
    api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1"
)


def get_ragas_llm():
    """Get LLM configured for RAGAS evaluation."""
    return llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=_groq_client,
    )


def get_ragas_embeddings():
    """Get embeddings configured for RAGAS via Ollama."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    return LangchainEmbeddingsWrapper(embed)


def _run_evaluate(samples: list[SingleTurnSampleOrMultiTurnSample]) -> EvaluationResult:
    """Internal: run RAGAS evaluate on samples."""
    return evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=get_ragas_llm(),  # type: ignore[arg-type] — ragas v0.4.3 type hints incomplete
        embeddings=get_ragas_embeddings(),
        run_config=RunConfig(timeout=120),
    )


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> EvaluationResult:
    """Evaluate RAG pipeline with RAGAS metrics."""
    # RAGAS dataset
    samples: list[SingleTurnSampleOrMultiTurnSample] = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        samples.append(
            SingleTurnSample(
                user_input=q, retrieved_contexts=ctx, response=a, reference=gt
            )
        )
    return _run_evaluate(samples)


def evaluate_real_rag(
    questions: list[str], ground_truths: list[str]
) -> EvaluationResult:
    """Evaluate RAG pipeline with RAGAS metrics."""
    # Initializations
    samples: list[SingleTurnSampleOrMultiTurnSample] = []
    # RAGAS dataset
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        print(f"Getting answer {i + 1}/{len(questions)}...")
        answer, context = text_rag_agent(q, use_hybrid=True)
        samples.append(
            SingleTurnSample(
                user_input=q, retrieved_contexts=context, response=answer, reference=gt
            )
        )
    return _run_evaluate(samples)


questions = [
    "Who is Jay Gatsby?",
    "What does the green light symbolize?",
    "Who is Daisy Buchanan?",
]

ground_truths = [
    "Jay Gatsby is a mysterious wealthy man living in West Egg.",
    "The green light represents Gatsby's dreams and longing for Daisy.",
    "Daisy Buchanan is Gatsby's love interest, married to Tom Buchanan.",
]

print(evaluate_real_rag(questions, ground_truths))
