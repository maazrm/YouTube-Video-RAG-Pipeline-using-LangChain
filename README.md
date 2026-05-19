# YouTube Video RAG Pipeline using LangChain

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows users to chat with a YouTube video transcript. This project uses LangChain, OpenAI embeddings, a FAISS vector database, and LangChain Expression Language (LCEL) to build a clean, production-ready AI workflow.

## 🚀 Features
* **Automated Transcription:** Fetches captions directly from YouTube using the `youtube-transcript-api`.
* **Smart Text Chunking:** Segments long transcripts using `RecursiveCharacterTextSplitter` with overlapping windows to preserve context.
* **Vector Store Indexing:** Embeds text chunks using OpenAI's `text-embedding-3-small` and stores them in a local `FAISS` vector database.
* **Modern LCEL Composition:** Orchestrates the retrieval, context formatting, and prompt injection steps cleanly using LangChain Expression Language (`RunnableParallel`, `RunnablePassthrough`).

---

## 🛠️ Tech Stack
* **Framework:** LangChain (Core, OpenAI, Community)
* **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Environment Management:** Python `python-dotenv`

---