from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from llm_tools_lab.config import settings
from llm_tools_lab.rag.rag_agent import text_rag_agent

# Initializations
client = OpenAI(
    api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1"
)


def get_ragas_llm():
    """Get LLM configured for RAGAS evaluation."""
    return llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=client,
    )


def get_ragas_embeddings():
    """Get embeddings configured for RAGAS via Ollama."""
    embed = OllamaEmbeddings(model="nomic-embed-text")
    return LangchainEmbeddingsWrapper(embed)


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> EvaluationResult:
    """Evaluate RAG pipeline with RAGAS metrics."""
    # RAGAS dataset
    samples = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        samples.append(
            SingleTurnSample(
                user_input=q, retrieved_contexts=ctx, response=a, reference=gt
            )
        )
    dataset = EvaluationDataset(samples=samples)
    # Configuring metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_ragas_llm(),  # type: ignore[arg-type]
        embeddings=get_ragas_embeddings(),
        run_config=RunConfig(timeout=120),
    )


def evaluate_real_rag(
    questions: list[str], ground_truths: list[str]
) -> EvaluationResult:
    """Evaluate RAG pipeline with RAGAS metrics."""
    # Initializations
    samples = []
    # RAGAS dataset
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        answer, context = text_rag_agent(q, use_hybrid=True)
        samples.append(
            SingleTurnSample(
                user_input=q, retrieved_contexts=context, response=answer, reference=gt
            )
        )
    dataset = EvaluationDataset(samples=samples)
    # Configuring metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_ragas_llm(),  # type: ignore[arg-type]
        embeddings=get_ragas_embeddings(),
        run_config=RunConfig(timeout=120),
    )
