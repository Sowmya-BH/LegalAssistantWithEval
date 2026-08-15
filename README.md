
```markdown
# Legal GraphRAG

Legal GraphRAG is a system designed for automated legal contract review using a combination of vector- and graph-based retrieval that converts unstructured legal contracts like PDFs into a searchable structured knowledge base. The architecture utilizes OCR-aware PDF ingestion to populate a hybrid database consisting of Chroma for semantic search and Neo4j for relationship mapping.
At its core, a LangGraph-driven pipeline coordinates specialized agents that classify queries, audit evidence, and synthesize structured answers while maintaining human-in-the-loop approval checkpoints. The framework emphasizes traceable evidence, deterministic confidence levels, and performance monitoring for complex legal analysis.
---

## 🛠️ Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **API** | FastAPI | Upload, query, jobs, and review endpoints |
| **Orchestration** | LangGraph | Stateful agent workflow |
| **PDF Extraction** | pdfplumber | Text and table extraction |
| **OCR** | OCR fallback | Processing scanned documents |
| **Embeddings** | Sentence Transformers | Semantic vector representation |
| **Vector DB** | Chroma | Dense semantic retrieval |
| **Keyword Retrieval**| BM25 | Exact legal terminology retrieval |
| **Graph DB** | Neo4j | Clause and entity relationship mapping |
| **LLM** | LLM API (Configurable) | Extraction, reasoning, and synthesis |
| **Agent / RAG Layer**| LangGraph + Custom Agents | Routing, retrieval, and auditing |
| **Observability** | LangSmith | LLM and pipeline tracing |
| **Evaluation** | CUAD + RAGAS | Retrieval and answer quality assessment |
| **Configuration** | `.env` | API keys and environment configuration |

---

## 🏗️ Project Structure

```text
legal_graphrag/
├── .vscode/
│   ├── settings.json          # Interpreter, test discovery, analysis paths
│   └── launch.json             # Run/debug configs for demo CLI and tests
├── .env.example                 # Copy to .env and fill in configuration
├── .gitignore
├── pyproject.toml               # Editable install + pytest config
├── requirements.txt
├── src/legal_graphrag/
│   ├── config.py                 # Fail-fast env configuration loader
│   ├── llm_client.py              # Shared Anthropic client + JSON/text helpers
│   ├── tracing.py                 # LangSmith @traceable wrapper
│   ├── resources.py               # Singletons: Neo4j store, embedder, Chroma client
│   ├── ingestion/
│   │   ├── pdf_pipeline.py        # PDF/table extraction, OCR fallback, Chroma storage
│   │   └── langgraph_agent.py     # Ingestion StateGraph with conditional OCR routing
│   ├── retrieval/                 
│   │   ├── contract_metadata.py     # LLM extraction of contract types, dates, parties
│   │   └── hybrid_search.py         # Dense (Chroma) + Lexical (BM25) + RRF + reranking
│   ├── graphrag/                   
│   │   ├── neo4j_store.py         # Neo4j schema & read/write operations
│   │   ├── extraction.py          # LLM clause/conflict/risk extraction, text-to-Cypher
│   │   └── langgraph_agent.py     # Reference standalone ingestion/query graphs
│   └── agents/                     # Core Agent Pipeline
│       ├── prompts.py               # Route, audit, and synthesizer prompt logic
│       └── legal_pipeline.py        # Main LangGraph state machine
├── scripts/
│   ├── run_demo.py                # CLI interface for ingestion and queries
│   └── run_cuad_eval.py           # CUAD dataset ingestion + RAGAS evaluation CLI
├── tests/
│   ├── test_pdf_pipeline.py
│   └── test_graphrag_extraction.py
└── data/
    ├── uploads/                   # Local PDFs for ingestion (gitignored)
    ├── metadata/                  # Per-document chunk metadata (gitignored)
    └── chroma_db/                 # Local Chroma vector store (gitignored)

```

---

## 🛡️ Security & Safety

For legal and enterprise systems, accuracy and database integrity are non-negotiable. The system incorporates two core safety layers to prevent hallucinations and database corruption:

| Safety Layer | Implementation Strategy | Risk Mitigated |
| :--- | :--- | :--- |
| **Grounding** | LLM constrained to answer solely using retrieved vector/graph context. | **Hallucinations & unverified statements**<br><sub>Evaluated using metrics like Faithfulness: *"Is every claim in the answer backed by the context provided?"*</sub> |
| **Execution Safety** | Static analysis & validation of generated Cypher queries prior to database execution. | **Unauthorized writes, data deletion, or injection**<br><sub>Ensures all generated database operations are strictly read-only.</sub> |



## 🤖 Main Pipeline: Router + Specialist Agents + Auditor + Synthesizer

`src/legal_graphrag/agents/legal_pipeline.py` is the primary query-answering entrypoint. A **Router** classifies questions into `hybrid`, `graph`, or `direct`, dispatches them to matching specialist agents (`HybridSearchAgent` or `GraphRAGAgent`), passes the results through an **Auditor** with a human evidence checkpoint, and sends them to a **Synthesizer** with a second human approval checkpoint.

```python
from legal_graphrag.agents.legal_pipeline import build_legal_agent_graph
from legal_graphrag.output_formatting import render_full_answer
from langgraph.types import Command

app = build_legal_agent_graph()
config = {"configurable": {"thread_id": "job-1"}}

# Step 1: Execute until evidence checkpoint
result = app.invoke({"question": "...", "collection_name": "..."}, config=config)
print(result["__interrupt__"])  # Review evidence

# Step 2: Resume after evidence approval -> Pause at answer checkpoint
result = app.invoke(Command(resume={"proceed": True, "reviewer": "jane.doe"}), config=config)
print(result["__interrupt__"])  # Review answer

# Step 3: Approve final answer
result = app.invoke(Command(resume={"action": "approve", "reviewer": "jane.doe"}), config=config)
print(render_full_answer(result))

```

---

## 📋 Answer Output Format
Here you can find two tabs 
1. ASK CUAD -> To demonstrate the working of my project against CUAD Dataset downloaded from HuggingFace Data Hub to measure the quality of answers generated by my system
2. Upload and Ask -> The user is allowed to upload a pdf of their choice and ask questions. Every user question passes through two human checkpoints.
    a. Evidence checkpoint : Either Accept or Reject
    b. 3-way Answer Checkpoint: Approve or Revise(the control passes back to synthesizer agent to come up with revised answer) or Reject and stop the workflow

   <img width="3024" height="1964" alt="317B1C5F-1856-4331-8EC4-C6BC7F0F7918" src="https://github.com/user-attachments/assets/097c2908-9bec-46c4-835e-eabebb2d8140" />

<img width="3024" height="1964" alt="1ACC4DB0-974C-4E67-91B4-8073C401945B" src="https://github.com/user-attachments/assets/daee557f-211b-4bd8-b6c6-46237920df65" />


```

---

## 📊 Evaluation: CUAD + RAGAS + LangSmith

The query pipeline can be evaluated against the [CUAD Benchmark](https://www.atticusprojectai.org/cuad) using RAGAS metrics via `scripts/run_cuad_eval.py`.

### Evaluation Workflow

1. **Load & Sample:** Parses local `CUAD_v1.json` keeping representative answerable/impossible ratios.
2. **Ingest (Vector Store):** Embeds contracts into Chroma via `pdf_pipeline.py`.
3. **Automated Pipeline Execution:** Runs the LangGraph state machine auto-resuming checkpoints with a scripted reviewer.
4. **Scoring:**
* **Answerable questions:** Scored using RAGAS metrics (`faithfulness`, `context_precision`, `context_recall`, `answer_correctness`).
* **Unanswerable (`is_impossible`) questions:** Evaluated for correct absence-detection accuracy to prevent hallucinations.


5. **Tracing:** Tracks traces in LangSmith under the `legal-graphrag-cuad-eval` project.

### Running Evaluations

```bash
# Install evaluation dependencies
pip install ragas langchain-anthropic langchain-huggingface langsmith huggingface_hub

export ANTHROPIC_API_KEY=sk-ant-...
export LANGSMITH_API_KEY=ls__...  # Optional tracing key

# Run evaluation (auto-downloads dataset from HuggingFace)
python -m scripts.run_cuad_eval --n-contracts 20 --seed 42 --out data/metadata/cuad_eval_results.json

```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Installation

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt --no-deps-for sentence-transformers
pip install -e .

# Environment setup
cp .env.example .env

```

> **Note on `sentence-transformers`:** If using a specific PyTorch build (e.g., CUDA-matched), install `sentence-transformers` with `--no-deps` to avoid automatic PyTorch downgrades.

### 2. Database Setup

Start a local Neo4j instance using Docker:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/change-me neo4j:5

```

### 3. Usage & Execution

**CLI Commands:**

```bash
# Ingest a PDF document
python -m scripts.run_demo ingest --file data/uploads/sample.pdf --vendor "ABC Ltd."

# Execute a complex legal query
python -m scripts.run_demo ask --question "Show all contracts where ABC Ltd. is the vendor..." --collection sample_pdf

```

**Running Tests:**

```bash
pytest

```

Project Deliverables:
1. https://gamma.app/docs/Legal-Contract-Intelligence-Platform-swcgtvc4mi86m71?mode=doc
2. https://docs.google.com/document/d/1u1ATQ1UbCB7MGS4LKG_hHCQiHUTk2tam8LGX71UEXOc/edit?usp=sharing (Detailed Technical Documentation)
3. https://drive.google.com/file/d/1k5EwDAYmrmh1WuqW1WFD5pFNkLnLFqnl/view?usp=sharing ->project walkthrough
4. https://drive.google.com/file/d/1MsHjkdwhWdga0IWt3ARA5IlJ8YD1ZZRV/view?usp=sharing --> UI demo
