# Smart AI Research Assistant (PipeChain RAG Engine)

Project covers a high-performance, asynchronous, and type-safe Retrieval-Augmented Generation (RAG) pipeline built using modern Python toolsets (`uv`), LangChain, and structured validation layers. 

This project demonstrates an enterprise-ready architecture for document ingestion, contextual memory retention, dynamic multi-query expansion, and deterministic data extraction. It serves as a benchmark platform for analyzing retrieval accuracy and token density.

## Architectural Overview

The application splits computational workloads into three isolated components:
1. **Data Ingestion & Extraction (`document_loader.py`)**: A decoupled layer utilizing `TextLoader` and `PyPDFLoader` to ingest filesystem binaries into normalized textual memory.
2. **The RAG Engine Core (`researcher.py`)**: Houses token splitters (`RecursiveCharacterTextSplitter`), the low-latency local vector DB connection (`Chroma`), conversation history trackers (`InMemoryChatMessageHistory`), and advanced multi-query automation routines (`MultiQueryRetriever`).
3. **The Orchestration Layer (`main.py`)**: Coordinates execution loops, implements structured logging via `structlogger`, manages environment contexts, and handles the asynchronous scheduler.

## Key Technical Showcases

### 1. Asynchronous Execution Network I/O (`asyncio`)
Network latency to remote LLM clusters is the primary bottleneck of modern AI agents. This assistant implements fully non-blocking asynchronous operations using LangChain's `.ainvoke()` design patterns. Thread execution scales gracefully, releasing resources back to the event loop during cloud API round-trips.

### 2. Hard-Enforced Data Typing (OpenAI Structured Outputs)
Instead of relying on prompt instructions to eliminate conversational pleasantries, the `ask_structured` pipeline binds a strict Pydantic model (`ResearchResponse`) natively to the LLM core using `.with_structured_output()`. 

The system guarantees deterministic JSON responses complying with this verification schema:
- **`answer`**: Explicit context-bounded synthesis string.
- **`confidence`**: Strict high/medium/low quality metric rating.
- **`sources`**: List array of validated source file names used during extraction.
- **`key_quotes`**: Word-for-word string quotes captured directly from source chunks.
- **`follow_up_questions`**: 2-3 machine-generated proactive logical discovery loops.

### 3. Advanced Multi-Query Retrieval & Deduplication
Multi-Query Retriever utilizes `gpt-4o-mini` to automatically rewrite a single user inquiry into multiple contextual variations. The engine searches the vector space for all queries simultaneously, combining and deduplicating the resulting document arrays to optimize information recall.

### 4. Telemetry Reporting & Diagnostic Auditing
The pipeline includes a built-in benchmark method (`compare_retrievers`) that prints footprint reports side-by-side for analyzing retrieval accuracy and token density. 

## Repository Structure
```text
smart-ai-research-assistant/
├── .venv                           # Project dedicated Python environment
├── database/                       # Persistent Chroma local vector store (Git ignored)
├── documents/                      # Target directory for source materials (.txt, .pdf)
│   ├── attention_mechanisms.txt
│   ├── langchain_docs.pdf
│   └── rag_survey.txt
├── src/
│   └── smartairesearchassistance/
│       ├── __init__.py
│       ├── document_loader.py      # Router for PyPDF & Text loaders
│       ├── main.py                 # Synchronous wrapper and async execution coordinator
│       ├── researcher.py           # AIResearchAssistant core logic and Pydantic schemas
│       └── structlogger.py         # Structured logging engine configuration
├── .env.example                    # Environment configuration template
├── .gitignore                      # Protection matrix preventing data/credential leaks
├── pyproject.toml                  # Metadata configuration
├── uv.lock                         # Locked dependency footprint tracking
└── README.md                       # Architecture documentation
```