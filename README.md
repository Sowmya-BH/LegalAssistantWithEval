# Legal GraphRAG

PDF ingestion (OCR-aware, table-aware) → vector store (Chroma) + graph store (Neo4j) →
hybrid retrieval → human-approved answers. Built for legal contract review.

## Project structure

```
legal_graphrag/
├── .vscode/
│   ├── settings.json          # interpreter, test discovery, analysis paths
│   └── launch.json             # run/debug configs for the demo CLI and tests
├── .env.example                 # copy to .env and fill in
├── .gitignore
├── pyproject.toml               # editable install + pytest config
├── requirements.txt
├── src/legal_graphrag/
│   ├── config.py                 # loads .env once; require_env() for fail-fast startup
│   ├── llm_client.py              # shared Anthropic client + JSON/text call helpers
│   ├── tracing.py                  # LangSmith @traceable wrapper (safe no-op if langsmith isn't installed)
│   ├── resources.py                # shared singletons: Neo4j store, embedder, Chroma client
│   ├── ingestion/
│   │   ├── pdf_pipeline.py        # pdfplumber extraction, OCR fallback, table extraction,
│   │   │                          # clean/chunk, Chroma vector storage (+ extra_metadata hook)
│   │   └── langgraph_agent.py     # same pipeline as a LangGraph StateGraph (conditional OCR routing)
│   ├── retrieval/                  # HybridSearchAgent's implementation
│   │   ├── contract_metadata.py     # LLM extraction of contract_type/parties/dates/value/law
│   │   └── hybrid_search.py         # dense (Chroma) + lexical (BM25) + RRF fusion + reranker
│   ├── graphrag/                   # GraphRAGAgent's implementation
│   │   ├── neo4j_store.py         # Neo4j schema + all read/write operations
│   │   ├── extraction.py          # LLM clause/conflict/risk extraction, text-to-Cypher
│   │   └── langgraph_agent.py     # standalone ingestion graph + standalone query graph
│   │                               # (kept for reference; agents/legal_pipeline.py supersedes
│   │                               #  its query graph with the router+auditor+synthesizer design)
│   └── agents/                     # Router + Auditor + Synthesizer — the main pipeline
│       ├── prompts.py               # classify_route / verify_evidence / synthesize_legal_answer
│       └── legal_pipeline.py        # the full LangGraph state machine (see its module docstring
│                                    # for an ASCII diagram of the whole flow)
├── scripts/
│   ├── run_demo.py                # CLI: `ingest` and `ask` subcommands
│   └── run_cuad_eval.py           # CLI: CUAD ingestion + batch RAGAS evaluation (see "Evaluation" below)
├── tests/
│   ├── test_pdf_pipeline.py
│   └── test_graphrag_extraction.py
└── data/
    ├── uploads/                   # PDFs you ingest locally (gitignored)
    ├── metadata/                  # per-document chunk metadata JSON (gitignored)
    └── chroma_db/                 # local vector store (gitignored)
```

## Main pipeline: router + specialist agents + auditor + synthesizer

`src/legal_graphrag/agents/legal_pipeline.py` is the primary query-answering
entrypoint — a Router classifies each question into `hybrid` / `graph` /
`direct`, dispatches to the matching specialist agent (`HybridSearchAgent` or
`GraphRAGAgent`), passes the result through an Auditor with a human evidence
checkpoint, then a Synthesizer with a second human approval checkpoint, and
optionally writes the reviewed answer back into Neo4j. See the module
docstring in that file for the full flow diagram.

```python
from legal_graphrag.agents.legal_pipeline import build_legal_agent_graph
from legal_graphrag.output_formatting import render_full_answer
from langgraph.types import Command

app = build_legal_agent_graph()
config = {"configurable": {"thread_id": "job-1"}}

result = app.invoke({"question": "...", "collection_name": "..."}, config=config)
print(result["__interrupt__"])  # evidence checkpoint — reviewer inspects this

result = app.invoke(Command(resume={"proceed": True, "reviewer": "jane.doe"}), config=config)
print(result["__interrupt__"])  # answer checkpoint

result = app.invoke(Command(resume={"action": "approve", "reviewer": "jane.doe"}), config=config)
print(render_full_answer(result))
```

`graphrag/langgraph_agent.py`'s standalone query graph (`build_query_graph`)
is kept for reference/backward compatibility, but `agents/legal_pipeline.py`
is the design this project actually recommends running — it's what
`scripts/run_demo.py ask` and `scripts/run_cuad_eval.py` both drive.

## Answer output format

The synthesizer no longer just returns a paragraph — every approved answer
is a **structured card**: a short direct `answer`, a separate `evidence`
excerpt explaining why, a `confidence` (High/Medium/Low, derived
deterministically from the evidence auditor's own verdict — see
`agents/prompts.derive_confidence`), and a `document`/`source_section`/
`source_page` attribution. `src/legal_graphrag/output_formatting.py` renders
this three ways from the same underlying dict, so they never drift apart:

- `format_answer_text()` — the plain template:
  ```
  Document: DOMINIADVISORTRUST...
  Question: What is the agreement date?
  Answer:
  The agreement date is February 18, 2005.
  Evidence:
  The agreement identifies February 18, 2005 as the date of the Sponsorship Agreement.
  Confidence: High
  Source: Section/Clause/Page X
  ```
- `format_answer_card()` — the boxed terminal/UI card (used by `scripts/run_demo.py ask`):
  ```
  ┌─────────────────────────────────────────────┐
  │ Answer                                       │
  ├─────────────────────────────────────────────┤
  │ February 18, 2005                            │
  │                                               │
  │ Confidence: High                              │
  ├─────────────────────────────────────────────┤
  │ Evidence                                     │
  │                                               │
  │ "..."                                        │
  ├─────────────────────────────────────────────┤
  │ Source                                       │
  │ Sponsorship Agreement · Page 1                │
  └─────────────────────────────────────────────┘
  ```
- `format_technical_details()` — route, retrieved chunks, the evidence
  auditor's full verdict, and revision count, behind a collapsed
  **"▶ Retrieval & reasoning details"** section (a real collapsible
  `<details>` block for markdown output; a `--verbose`-gated teaser line
  for the CLI, e.g. `python -m scripts.run_demo ask ... --verbose`).

`render_full_answer(state, verbose=False, markdown=False)` is the single
entrypoint that combines the above — call it with the pipeline's final
`LegalAgentState` (specifically its `final_structured_answer` field).

## Evaluation: CUAD + RAGAS + LangSmith

`scripts/run_cuad_eval.py` evaluates the query pipeline (`agents/legal_pipeline.py`)
against [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding
Atticus Dataset), a human-annotated legal contract QA benchmark. This is a
separate, optional evaluation harness — it doesn't run as part of the main
application, and none of it is imported unless you run the eval script.

### What it does

1. **Load & sample** — parses a local `CUAD_v1.json`, deterministically
   samples a set of contracts (`--n-contracts`), keeping the answerable /
   `is_impossible` question ratio representative.
2. **Ingest (vector store only)** — embeds each sampled contract into its
   own Chroma collection via the existing chunking pipeline
   (`ingestion/pdf_pipeline.py`). Deliberately **skips** the Neo4j
   clause/conflict/risk extraction graph — CUAD-scale ingestion through that
   path would mean many extra LLM calls per contract before evaluation even
   starts, and RAGAS's metrics only assess retrieval + answer quality, which
   the `hybrid` route covers on its own. Neo4j is still used at query time
   for the pipeline's job/audit-trail records, just never populated with
   CUAD clause data.
3. **Run the real pipeline, scripted** — for every question, drives the
   actual compiled LangGraph (`build_legal_agent_graph()`) end to end,
   auto-resuming both `interrupt()` checkpoints with a scripted "always
   approve" reviewer (`evaluation/scripted_reviewer.py`) instead of a human,
   via `Command(resume=...)`. Forced onto the `hybrid` route
   (`force_route="hybrid"`, a small addition to `LegalAgentState`/
   `router_node`) since no clause graph exists for CUAD contracts.
4. **Score, split by `is_impossible`**:
   - **Answerable questions** → scored with RAGAS: `faithfulness`,
     `context_precision`, `context_recall`, `answer_correctness`, against
     CUAD's human-annotated answer spans as the reference.
   - **`is_impossible` questions** (CUAD annotators found no clause of that
     category) → excluded from the RAGAS set and scored separately with a
     **correctly-identified-absence accuracy**: did the auditor flag
     insufficient evidence, did the evidence checkpoint reject, or did the
     final answer itself read as "no such clause" — rather than
     hallucinating one. Mixing these into the RAGAS set would tank
     `context_recall`/`answer_correctness` on questions that were never
     answerable to begin with.
5. **Trace every checkpoint in LangSmith** (optional — set `LANGSMITH_API_KEY`
   to enable; the run works identically without it). Each traced run shows:
   `router` → `hybrid_search_agent` → `auditor` → `checkpoint.human_evidence`
   → `synthesizer` → `checkpoint.human_answer` → `finalize`, plus the raw
   Anthropic calls underneath each LLM-backed step, tagged with the CUAD
   contract type and answerable/`is_impossible` status
   (`evaluation/langsmith_tracing.py`) so individual questions are
   filterable in the LangSmith UI. Traces land in a dedicated
   `legal-graphrag-cuad-eval` project, separate from interactive usage.

### Running it

```bash
pip install ragas langchain-anthropic langchain-huggingface langsmith huggingface_hub  # eval-only deps

export ANTHROPIC_API_KEY=sk-ant-...
export LANGSMITH_API_KEY=ls__...     # optional — omit to run without tracing

# Auto-downloads CUAD from Hugging Face (theatticusproject/cuad) and caches it locally:
python -m scripts.run_cuad_eval --n-contracts 20 --seed 42 --out data/metadata/cuad_eval_results.json

# Or use your own local copy instead:
python -m scripts.run_cuad_eval \
    --cuad-json data/uploads/CUAD_v1.json \
    --n-contracts 20 \
    --seed 42 \
    --out data/metadata/cuad_eval_results.json
```

Requires a running Neo4j instance (see Setup above) even though CUAD
contracts themselves are never written to it, and `ANTHROPIC_API_KEY` for
both the pipeline's own LLM calls and RAGAS's LLM-judged metrics (which
reuse the same Claude model via `langchain-anthropic`).

### Known limitations

- The scripted reviewer always approves both checkpoints on the first
  pass — it never scripts a "revise" round, since that would mean
  fabricating reviewer feedback a human never actually gave. A stricter
  policy (e.g. auto-reject when the auditor's `sufficient` verdict is
  `False`) is available as `reject_evidence_if_insufficient` in
  `scripted_reviewer.py` if you want to evaluate that behavior instead.
- Absence detection uses a keyword heuristic (`_ABSENCE_PHRASES` in
  `ragas_eval.py`) as a cheap first pass over the final answer text. Swap in
  an LLM judge there if the heuristic's false-negative rate looks high on
  your sample.
- CUAD's title-derived `contract_type`/`parties` metadata is coarser than
  what a real ingest would extract via `contract_metadata.py` — party names
  and dates aren't in CUAD's title, so `metadata_filter` tests against CUAD
  contracts are limited. This doesn't affect the QA evaluation itself.



```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# System packages (Ubuntu/Debian — adjust for your OS):
sudo apt-get install poppler-utils tesseract-ocr

pip install -r requirements.txt --no-deps-for sentence-transformers  # see note below
pip install -e .

cp .env.example .env               # then fill in NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / ANTHROPIC_API_KEY
```

> **Note on `sentence-transformers`:** if you already have a specific `torch` build
> installed (e.g. CUDA-matched), install it with `pip install sentence-transformers --no-deps`
> and then install its other dependencies (`transformers`, `tokenizers`, `huggingface-hub`,
> `safetensors`, `scikit-learn`, `scipy`, `Pillow`, `tqdm`) separately, so pip doesn't
> silently downgrade your existing torch to satisfy its pin.

You also need a running Neo4j instance (local via Docker, or Aura):

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/change-me neo4j:5
```

## Running in VS Code

Open the folder in VS Code, select the `.venv` interpreter (Python: Select Interpreter),
then use the Run and Debug panel — two configs are preconfigured in `.vscode/launch.json`:

- **Run ingestion demo** — ingests `data/uploads/sample.pdf`, pauses in the integrated
  terminal for approval, then persists to both Chroma and Neo4j.
- **Run query demo** — asks the example multi-hop question, pauses for approval, prints
  the final approved answer.

Or from a terminal:

```bash
python -m scripts.run_demo ingest --file data/uploads/sample.pdf --vendor "ABC Ltd."
python -m scripts.run_demo ask --question "Show all contracts where ABC Ltd. is the vendor, the same clause appears in another contract, and that clause has been interpreted by multiple judgments." --collection sample_pdf
```

## Tests

```bash
pytest
```

Tests cover the parts of the pipeline that don't require live Neo4j/Chroma/LLM
connections (table markdown rendering, text cleaning, the read-only Cypher guard).
They're a starting point, not full coverage — extraction quality (clause detection,
conflict detection, risk flagging) needs evaluation against real documents, not unit tests.
