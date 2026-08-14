"""
RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

  1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
     metrics — faithfulness, context_precision, context_recall, and
     answer_correctness — against CUAD's human-annotated answer spans as
     the `reference`.

  2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
     clause of that category in the contract, so there's no real reference
     to recall against. These are excluded from the RAGAS dataset entirely
     and instead scored with a separate "correctly identified absence"
     metric: did the pipeline's own evidence auditor / answer correctly
     signal that nothing was found, rather than hallucinating a clause?
     Mixing the two would silently tank context_recall/answer_correctness
     on questions that were never answerable in the first place.

Requires (not in the base requirements.txt — see requirements.txt's
"Evaluation" section): `ragas`, `langchain-huggingface`, and ONE of
`langchain-anthropic` / `langchain-groq` depending on RAGAS_JUDGE_PROVIDER
(see _build_ragas_judge_llm below). RAGAS's LLM-judged metrics (faithfulness,
context precision/recall, answer_correctness) need a judge LLM + embeddings
model; the judge defaults to Claude but is swappable via env vars
(RAGAS_JUDGE_PROVIDER=anthropic|groq, RAGAS_JUDGE_MODEL=<model id>) —
embeddings always use the SAME model resources.py's HybridSearchAgent uses
(all-mpnet-base-v2) so "similar enough" judgments are consistent with what
retrieval itself considers similar, regardless of which provider judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_client import MODEL_NAME
from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
from ..tracing import traceable
from .cuad_loader import CUADExample
from .scripted_reviewer import ScriptedRunResult

# ---------------------------------------------------------------------------
# Absence-detection scoring (unanswerable / is_impossible questions)
# ---------------------------------------------------------------------------

_ABSENCE_PHRASES = (
    "no clause", "does not contain", "no such clause", "not found", "no provision",
    "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
)


def _looks_like_absence_answer(answer: Optional[str]) -> bool:
    if not answer:
        return False
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


@dataclass
class AbsenceDetectionResult:
    qas_id: str
    contract_title: str
    clause_category: str
    evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
    evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
    answer_stated_absence: bool            # final answer text reads as "no such clause"
    correctly_identified_absence: bool     # any of the above three


def score_absence_detection(
    results: list[ScriptedRunResult], examples: list[CUADExample]
) -> list[AbsenceDetectionResult]:
    """
    Scores the `is_impossible` subset. `correctly_identified_absence` is
    True if EITHER the evidence auditor flagged insufficient evidence, the
    evidence checkpoint rejected outright, or the synthesized answer itself
    reads as an absence statement — any of these means the pipeline did NOT
    fabricate a clause that CUAD's annotators confirmed doesn't exist.
    """
    scored = []
    for example, result in zip(examples, results):
        assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
        evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
        evidence_rejected = result.status == "evidence_rejected"
        answer_absence = _looks_like_absence_answer(result.final_answer)
        scored.append(
            AbsenceDetectionResult(
                qas_id=example.qas_id,
                contract_title=example.contract_title,
                clause_category=example.clause_category,
                evidence_marked_insufficient=evidence_insufficient,
                evidence_rejected=evidence_rejected,
                answer_stated_absence=answer_absence,
                correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
            )
        )
    return scored


def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
    if not scored:
        return float("nan")
    return sum(s.correctly_identified_absence for s in scored) / len(scored)


# ---------------------------------------------------------------------------
# RAGAS dataset construction (answerable questions only)
# ---------------------------------------------------------------------------

@dataclass
class RagasRow:
    qas_id: str
    contract_title: str
    clause_category: str
    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


def build_ragas_rows(
    results: list[ScriptedRunResult], examples: list[CUADExample]
) -> list[RagasRow]:
    """
    Builds one row per answerable example. `retrieved_contexts` comes from
    `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
    are forced onto the `hybrid` route — see scripted_reviewer.py's
    force_route default and cuad_ingest.py's module docstring for why.
    Rows where the pipeline never produced a final_answer (evidence or
    answer rejected) still get a row with response="" so RAGAS scores that
    as a real failure rather than silently dropping it from the average.
    """
    rows = []
    for example, result in zip(examples, results):
        assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
        contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
            hit.get("text", "") for hit in result.graph_hits
        ]
        rows.append(
            RagasRow(
                qas_id=example.qas_id,
                contract_title=example.contract_title,
                clause_category=example.clause_category,
                user_input=example.question,
                retrieved_contexts=contexts,
                response=result.final_answer or "",
                reference=example.reference_answer,
                evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
            )
        )
    return rows


def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
    """
    Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
    ragas lazily so the rest of this module (and the non-RAGAS parts of the
    eval harness) works even if `ragas` isn't installed.
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r.user_input,
            retrieved_contexts=r.retrieved_contexts,
            response=r.response,
            reference=r.reference,
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)
def _build_ragas_judge_llm():
    """
    Build the RAGAS judge using Hugging Face Inference API.

    Uses HF_TOKEN from the environment / .env file.

    Example:
        HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
        RAGAS_JUDGE_MODEL=openai/gpt-oss-120b:groq
    """
    import os

    from huggingface_hub import InferenceClient
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    from ragas.llms import LangchainLLMWrapper

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Add HF_TOKEN=hf_... to your .env file."
        )

    model = os.getenv(
        "RAGAS_JUDGE_MODEL",
        "openai/gpt-oss-120b:groq",
    )

    class HuggingFaceChatModel(BaseChatModel):
        client: object = Field(exclude=True)
        model: str

        @property
        def _llm_type(self) -> str:
            return "huggingface_inference"

        @property
        def _identifying_params(self) -> dict:
            return {
                "model": self.model,
            }

        def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:

            hf_messages = []

            for message in messages:

                if isinstance(message, SystemMessage):
                    role = "system"

                elif isinstance(message, HumanMessage):
                    role = "user"

                elif isinstance(message, AIMessage):
                    role = "assistant"

                else:
                    role = "user"

                hf_messages.append(
                    {
                        "role": role,
                        "content": message.content,
                    }
                )

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=hf_messages,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.0),
            )

            text = completion.choices[0].message.content or ""

            generation = ChatGeneration(
                message=AIMessage(content=text)
            )

            return ChatResult(
                generations=[generation]
            )

    client = InferenceClient(
        api_key=hf_token,
        provider="auto",
    )

    hf_llm = HuggingFaceChatModel(
        client=client,
        model=model,
    )

    return LangchainLLMWrapper(hf_llm)

# def _build_ragas_judge_llm():
#     """
#     RAGAS's LLM-judged metrics need a langchain-wrapped chat model to act as
#     judge. Configurable via env vars so you can point this at whatever the
#     rest of your eval stack already uses, independent of llm_client.py's
#     Claude model:

#         RAGAS_JUDGE_PROVIDER = "anthropic" (default) | "groq"
#         RAGAS_JUDGE_MODEL    = provider-specific model id
#                                 (default: llm_client.MODEL_NAME for anthropic,
#                                  "openai/gpt-oss-120b" for groq)

#     For provider="groq": if RAGAS_JUDGE_MODEL carries a trailing
#     ":<routing-hint>" suffix (e.g. "openai/gpt-oss-120b:groq" — an
#     OpenRouter-style provider suffix), it's stripped before being passed to
#     ChatGroq, since Groq's own API takes the bare model id
#     ("openai/gpt-oss-120b") and rejects the suffixed form. Requires
#     GROQ_API_KEY to be set.
#     """
#     import os

#     from ragas.llms import LangchainLLMWrapper

#     provider = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic").lower()

#     if provider == "groq":
#         from langchain_groq import ChatGroq

#         model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
#         model = model.split(":", 1)[0]  # strip any ":<routing-hint>" suffix — see docstring
#         return LangchainLLMWrapper(ChatGroq(model=model))

#     if provider == "anthropic":
#         from langchain_anthropic import ChatAnthropic

#         model = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
#         return LangchainLLMWrapper(ChatAnthropic(model=model))

#     raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER={provider!r} — expected 'anthropic' or 'groq'.")


# def _build_ragas_llm_and_embeddings():
#     """
#     Judge LLM (see _build_ragas_judge_llm) + embeddings model. Embeddings
#     use the SAME model HybridSearchAgent uses (EMBEDDING_MODEL_NAME), so
#     RAGAS's notion of "similar" for answer_correctness matches what
#     retrieval itself uses, regardless of which provider judges the LLM
#     metrics.
#     """
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from ragas.embeddings import LangchainEmbeddingsWrapper

#     llm = _build_ragas_judge_llm()
#     embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
#     return llm, embeddings
def _build_ragas_llm_and_embeddings():
    """
    Build the Hugging Face judge LLM + Hugging Face embeddings.
    """

    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = _build_ragas_judge_llm()

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    return llm, embeddings

@traceable(name="evaluation.run_ragas", run_type="chain")
def run_ragas_evaluation(rows: list[RagasRow]):
    """
    Runs faithfulness, context_precision, context_recall, and
    answer_correctness over `rows` (answerable questions only — see this
    module's docstring). Returns ragas's EvaluationResult (has both an
    aggregate `.to_pandas()` table and per-metric scores).
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        AnswerCorrectness,
    )

    dataset = _build_ragas_evaluation_dataset(rows)
    llm, embeddings = _build_ragas_llm_and_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
    return evaluate(dataset=dataset, metrics=metrics)


# """
# RAGAS evaluation of the legal_graphrag query pipeline against CUAD ground truth.

# Two separate scoring paths, per cuad_loader.split_answerable()'s docstring:

#   1. ANSWERABLE questions (is_impossible=False): scored with standard RAGAS
#      metrics — faithfulness, context_precision, context_recall, and
#      answer_correctness — against CUAD's human-annotated answer spans as
#      the `reference`.

#   2. UNANSWERABLE questions (is_impossible=True): CUAD annotators found NO
#      clause of that category in the contract, so there's no real reference
#      to recall against. These are excluded from the RAGAS dataset entirely
#      and instead scored with a separate "correctly identified absence"
#      metric: did the pipeline's own evidence auditor / answer correctly
#      signal that nothing was found, rather than hallucinating a clause?
#      Mixing the two would silently tank context_recall/answer_correctness
#      on questions that were never answerable in the first place.

# Requires (not in the base requirements.txt — see requirements.txt's
# "Evaluation" section): `ragas`, `langchain-HF`, `langchain-huggingface`.
# RAGAS's LLM-judged metrics (faithfulness, context precision/recall,
# answer_correctness) need an LLM + embeddings model; this module reuses
# Claude (via langchain_HF, matching llm_client.py's MODEL_NAME) as
# the judge, and the SAME embedding model resources.py's HybridSearchAgent
# uses (all-mpnet-base-v2) so "similar enough" judgments are consistent with
# what retrieval itself considers similar.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Optional

# from ..llm_client import MODEL_NAME
# from ..ingestion.pdf_pipeline import EMBEDDING_MODEL_NAME
# from ..tracing import traceable
# from .cuad_loader import CUADExample
# from .scripted_reviewer import ScriptedRunResult

# import os

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings

# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# ---------------------------------------------------------------------------
# # Absence-detection scoring (unanswerable / is_impossible questions)
# # ---------------------------------------------------------------------------
# MODEL_NAME = "openai/gpt-oss-120b:groq"
# _ABSENCE_PHRASES = (
#     "no clause", "does not contain", "no such clause", "not found", "no provision",
#     "not present", "no mention", "not addressed", "does not appear", "no relevant clause",
# )


# def _looks_like_absence_answer(answer: Optional[str]) -> bool:
#     if not answer:
#         return False
#     lowered = answer.lower()
#     return any(phrase in lowered for phrase in _ABSENCE_PHRASES)


# @dataclass
# class AbsenceDetectionResult:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     evidence_marked_insufficient: bool     # auditor's own `sufficient` verdict was False
#     evidence_rejected: bool                # evidence checkpoint short-circuited (status == "evidence_rejected")
#     answer_stated_absence: bool            # final answer text reads as "no such clause"
#     correctly_identified_absence: bool     # any of the above three


# def score_absence_detection(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[AbsenceDetectionResult]:
#     """
#     Scores the `is_impossible` subset. `correctly_identified_absence` is
#     True if EITHER the evidence auditor flagged insufficient evidence, the
#     evidence checkpoint rejected outright, or the synthesized answer itself
#     reads as an absence statement — any of these means the pipeline did NOT
#     fabricate a clause that CUAD's annotators confirmed doesn't exist.
#     """
#     scored = []
#     for example, result in zip(examples, results):
#         assert example.is_impossible, "score_absence_detection expects only is_impossible examples"
#         evidence_insufficient = not bool(result.evidence_verdict.get("sufficient", True))
#         evidence_rejected = result.status == "evidence_rejected"
#         answer_absence = _looks_like_absence_answer(result.final_answer)
#         scored.append(
#             AbsenceDetectionResult(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 evidence_marked_insufficient=evidence_insufficient,
#                 evidence_rejected=evidence_rejected,
#                 answer_stated_absence=answer_absence,
#                 correctly_identified_absence=evidence_insufficient or evidence_rejected or answer_absence,
#             )
#         )
#     return scored


# def absence_detection_accuracy(scored: list[AbsenceDetectionResult]) -> float:
#     if not scored:
#         return float("nan")
#     return sum(s.correctly_identified_absence for s in scored) / len(scored)


# # ---------------------------------------------------------------------------
# # RAGAS dataset construction (answerable questions only)
# # ---------------------------------------------------------------------------

# @dataclass
# class RagasRow:
#     qas_id: str
#     contract_title: str
#     clause_category: str
#     user_input: str
#     retrieved_contexts: list[str]
#     response: str
#     reference: str
#     evidence_sufficient: bool = field(default=False)  # covariate for slicing scores post-hoc, not a RAGAS field


# def build_ragas_rows(
#     results: list[ScriptedRunResult], examples: list[CUADExample]
# ) -> list[RagasRow]:
#     """
#     Builds one row per answerable example. `retrieved_contexts` comes from
#     `hybrid_hits` (this project's HybridSearchAgent output) since eval runs
#     are forced onto the `hybrid` route — see scripted_reviewer.py's
#     force_route default and cuad_ingest.py's module docstring for why.
#     Rows where the pipeline never produced a final_answer (evidence or
#     answer rejected) still get a row with response="" so RAGAS scores that
#     as a real failure rather than silently dropping it from the average.
#     """
#     rows = []
#     for example, result in zip(examples, results):
#         assert not example.is_impossible, "build_ragas_rows expects only answerable examples"
#         contexts = [hit.get("text", "") for hit in result.hybrid_hits] or [
#             hit.get("text", "") for hit in result.graph_hits
#         ]
#         rows.append(
#             RagasRow(
#                 qas_id=example.qas_id,
#                 contract_title=example.contract_title,
#                 clause_category=example.clause_category,
#                 user_input=example.question,
#                 retrieved_contexts=contexts,
#                 response=result.final_answer or "",
#                 reference=example.reference_answer,
#                 evidence_sufficient=bool(result.evidence_verdict.get("sufficient", False)),
#             )
#         )
#     return rows


# def _build_ragas_evaluation_dataset(rows: list[RagasRow]):
#     """
#     Converts RagasRow -> ragas.dataset_schema.EvaluationDataset. Imports
#     ragas lazily so the rest of this module (and the non-RAGAS parts of the
#     eval harness) works even if `ragas` isn't installed.
#     """
#     from ragas import EvaluationDataset, SingleTurnSample

#     samples = [
#         SingleTurnSample(
#             user_input=r.user_input,
#             retrieved_contexts=r.retrieved_contexts,
#             response=r.response,
#             reference=r.reference,
#         )
#         for r in rows
#     ]
#     return EvaluationDataset(samples=samples)


# def _build_ragas_llm_and_embeddings():
#     """
#     RAGAS's LLM-judged metrics need a langchain-wrapped LLM + embeddings
#     model. Uses the same Claude model as the rest of the pipeline
#     (llm_client.MODEL_NAME) and the same embedding model HybridSearchAgent
#     uses (EMBEDDING_MODEL_NAME), so judgments are made with the same notion
#     of "similar" that retrieval itself uses.
#     """
#     llm_endpoint = HuggingFaceEndpoint(
#         repo_id="openai/gpt-oss-120b:groq",
#         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#         max_new_tokens=1000,
#         temperature=0.1,
#     )

#     hf_llm = ChatHuggingFace(
#         llm=llm_endpoint
#     )

#     llm = LangchainLLMWrapper(hf_llm)

#     # -----------------------------
#     # Hugging Face Embeddings
#     # -----------------------------
#     embeddings_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME
#     )

#     embeddings = LangchainEmbeddingsWrapper(
#         embeddings_model
#     )
#     return llm, embeddings
#     # from langchain_anthropic import ChatAnthropic
#     # from langchain_huggingface import HuggingFaceEmbeddings
#     # from ragas.llms import LangchainLLMWrapper
#     # from ragas.embeddings import LangchainEmbeddingsWrapper

#     # llm = LangchainLLMWrapper(ChatAnthropic(model=MODEL_NAME))
#     # embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    


# @traceable(name="evaluation.run_ragas", run_type="chain")
# def run_ragas_evaluation(rows: list[RagasRow]):
#     """
#     Runs faithfulness, context_precision, context_recall, and
#     answer_correctness over `rows` (answerable questions only — see this
#     module's docstring). Returns ragas's EvaluationResult (has both an
#     aggregate `.to_pandas()` table and per-metric scores).
#     """
#     from ragas import evaluate
#     from ragas.metrics import (
#         Faithfulness,
#         LLMContextPrecisionWithReference,
#         LLMContextRecall,
#         AnswerCorrectness,
#     )

#     dataset = _build_ragas_evaluation_dataset(rows)
#     llm, embeddings = _build_ragas_llm_and_embeddings()

#     metrics = [
#         Faithfulness(llm=llm),
#         LLMContextPrecisionWithReference(llm=llm),
#         LLMContextRecall(llm=llm),
#         AnswerCorrectness(llm=llm, embeddings=embeddings),
#     ]
#     return evaluate(dataset=dataset, metrics=metrics)
