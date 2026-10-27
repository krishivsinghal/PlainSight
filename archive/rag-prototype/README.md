# Recovered RAG experiment

These files were recovered from local editor history after the current public repository was found to contain only the non-retrieval contract review app. They preserve an exploratory PlainSight path, rather than a verified release.

- `utils/vector_store.py`: LangChain FAISS storage using OpenAI embeddings, generic recursive text splitting, and similarity search.
- `utils/llm_interface.py`: an earlier version of the app's LLM interface that obtains legal context from `LegalVectorStore` before prompting the risk model.
- `prompts/risk_prompt.py`: the accompanying prompt with a `legal_context` placeholder.

The original legal corpus, saved FAISS index, dependency lockfile, and evidence of deployment were not recovered. The original code does not show that retrieval was used in the incubator screenshots or in a production app. Its hard-coded legal examples may be outdated. The files are kept as evidence of the prototype's design and are not wired into the current app.
