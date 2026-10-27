# PlainSight

PlainSight is an experimental contract review app for Australian small businesses. It parses uploaded PDF or DOCX contracts, extracts clauses, writes plain-English summaries, and displays clause-level risk annotations. The project is archived and is not maintained as a legal advice product.

## What is in this repository

- **Contract review app:** The root `app.py`, `components/`, `utils/`, and `prompts/` contain the Streamlit prototype. `utils/llm_interface.py` uses GPT-4o-mini for clause extraction and summaries and GPT-4o for risk analysis. It also has a mock mode for interface development.
- **Earlier retrieval experiment:** [`archive/rag-prototype/`](archive/rag-prototype/) preserves source recovered from local editor history. It contains a LangChain FAISS vector store and an earlier risk-analysis path that inserts retrieved context into the LLM prompt. This code is separate from the current app and has not been validated end to end. The legal corpus and vector index used during development were not recovered.
- **Application screenshots:** The images below were included in my 2025 incubator application. They show the landing page and a clause-analysis screen from prototype iterations.

The incubator application described plans to expand retrieval over Australian legal sources, add rules, and validate clause coverage with users. Those plans should not be read as features or results established by this repository. The separate BM25 + FAISS case-law research project is not part of PlainSight.

## Screenshots

![PlainSight landing page](docs/screenshots/incubator-landing.png)

![Clause-level analysis screen](docs/screenshots/incubator-clause-analysis.png)

## Technical notes

The current app accepts PDF and DOCX files through `utils/document_parser.py`, then uses the prompts in `prompts/` for extraction, summarization, and risk analysis. Risk annotations are rendered in `components/detailed_analysis.py`.

The archived retrieval experiment uses generic text splitting and vector similarity search over a FAISS index. It does **not** implement hybrid retrieval or clause-aware chunking. Those methods appear in a different legal research project and should not be attributed to PlainSight. The archived files include illustrative legal text that may be outdated; do not use them as a source of current law.

This repository is a historical prototype. No evaluation benchmark or current production reliability claim is provided here.
