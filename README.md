Deployment Link: https://compliance-engine-4nyd.onrender.com


# Compliance Engine

A document intelligence tool that lets you upload compliance PDFs and ask questions about them. Built with FastAPI, Groq, and Cohere.

**Live:** https://compliance-engine-4nyd.onrender.com

---

## What it does

Upload any compliance document — HIPAA guidelines, GDPR policies, SOC2 frameworks, HR handbooks, legal contracts — and ask questions in plain English. The system reads the document, finds the most relevant sections, and generates an answer based strictly on what the document says.

It never answers from general knowledge. If the answer isn't in the document, it says so.

---

## Features

- Upload multiple PDFs per session
- Semantic search across all uploaded documents
- Chat history within a session
- Answers include confidence level and source reference
- Remove individual PDFs without clearing the session
- Works on mobile and desktop

---

## How it works

When you upload a PDF, the system splits the text into chunks of around 800 characters. Each chunk is converted into a semantic vector using the Cohere embedding API. These vectors are stored in memory.

When you ask a question, the question is also converted to a vector. The system finds the three most similar chunks using cosine similarity, then passes those chunks along with your question to the Groq LLM. The model generates a structured answer using only the retrieved content.

---

## Tech stack

- **Backend:** Python 3.11, FastAPI
- **LLM:** Groq — llama-3.1-8b-instant
- **Embeddings:** Cohere — embed-english-v3.0
- **Vector search:** NumPy cosine similarity (in-memory)
- **PDF processing:** LangChain, PyPDF
- **Frontend:** HTML, CSS, JavaScript (single file)
- **Hosting:** Render

---

## Running locally

Clone the repo and install dependencies:

```bash
git clone https://github.com/yadavchiragg/compliance-engine.git
cd compliance-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

Get free API keys:
- Groq: https://console.groq.com
- Cohere: https://cohere.com

Start the server:

```bash
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Frontend |
| GET | /health | Health check |
| POST | /new-session | Start a session |
| POST | /upload-pdf | Upload a PDF |
| POST | /ask-compliance | Ask a question |
| DELETE | /session/{id}/pdf/{name} | Remove a PDF |
| POST | /clear-history/{id} | Clear chat history |

---

## Notes

The free tier on Render spins down after 15 minutes of inactivity. The first request after that takes around 30 seconds to wake up. Sessions are stored in memory, so uploaded PDFs reset if the server restarts.
