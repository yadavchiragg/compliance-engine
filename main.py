"""
main.py — RAG Compliance Engine v14
Uses Groq for LLM. Uses nomic-embed via Groq API for embeddings.
No HuggingFace needed.
"""

import os
import json
import shutil
import uuid
import logging
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY missing.")

app = FastAPI(title="RAG Compliance Engine", version="14.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions = {}


import hashlib

def get_embedding(text: str) -> list:
    """
    Simple hash-based embedding — no API needed.
    Creates a 512-dim vector using character n-grams.
    Works offline, surprisingly effective for keyword matching.
    """
    text = text.lower()[:2000]
    vec = [0.0] * 512
    # character trigrams
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
        idx = h % 512
        vec[idx] += 1.0
    # normalize
    norm = sum(x*x for x in vec) ** 0.5
    if norm > 0:
        vec = [x/norm for x in vec]
    return vec


class TinyVectorStore:
    def __init__(self):
        self.texts     = []
        self.metadatas = []
        self.vectors   = []

    def add(self, texts, metadatas):
        for text, meta in zip(texts, metadatas):
            vec = get_embedding(text)
            self.texts.append(text)
            self.metadatas.append(meta)
            self.vectors.append(vec)

    def search(self, query_vec, k=3):
        if not self.vectors:
            return []
        mat   = np.array(self.vectors)
        q     = np.array(query_vec)
        norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        sims  = (mat @ q) / norms
        idxs  = np.argsort(sims)[::-1][:k]
        return [{"text": self.texts[i], "meta": self.metadatas[i]} for i in idxs]


@app.on_event("startup")
def load_components():
    logger.info("Loading Groq LLM...")
    from langchain_groq import ChatGroq
    app.state.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    logger.info("Ready!")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}

@app.post("/new-session")
def new_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"store": TinyVectorStore(), "pdfs": [], "history": []}
    return {"session_id": sid}

@app.get("/session/{session_id}")
def get_session(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"pdfs": s["pdfs"], "history": s["history"]}

@app.delete("/session/{session_id}/pdf/{pdf_name}")
def remove_pdf(session_id: str, pdf_name: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    keep = [(t,m,v) for t,m,v in zip(s["store"].texts, s["store"].metadatas, s["store"].vectors) if m.get("source_pdf") != pdf_name]
    s["store"].texts     = [x[0] for x in keep]
    s["store"].metadatas = [x[1] for x in keep]
    s["store"].vectors   = [x[2] for x in keep]
    s["pdfs"] = [p for p in s["pdfs"] if p["name"] != pdf_name]
    return {"message": f"{pdf_name} removed."}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(...)):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    s = sessions[session_id]
    if any(p["name"] == file.filename for p in s["pdfs"]):
        raise HTTPException(status_code=400, detail=f"{file.filename} already uploaded.")

    tmp_path = f"/tmp/{session_id}_{file.filename}"
    with open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    try:
        pages = PyPDFLoader(tmp_path).load()
        if not pages:
            raise HTTPException(status_code=400, detail="PDF appears empty.")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents(pages)

        texts     = [c.page_content for c in chunks]
        metadatas = [{"source_pdf": file.filename} for _ in chunks]

        logger.info(f"Embedding {len(chunks)} chunks via Groq...")
        s["store"].add(texts, metadatas)
        s["pdfs"].append({"name": file.filename, "pages": len(pages), "chunks": len(chunks)})
        logger.info("Done!")

        return {
            "message": "PDF added.",
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "total_pdfs": len(s["pdfs"]),
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


SYSTEM_PROMPT = """You are a strict compliance assistant. Answer ONLY from the context below.
Provide a thorough answer in 3 to 6 sentences. Plain prose only. Include relevant details, context, and any important conditions or exceptions mentioned in the document.
If insufficient info, set answer to "Insufficient information in the provided compliance documents."
Respond ONLY with this JSON — nothing else:
{{
  "answer": "<answer>",
  "confidence": "high|medium|low",
  "source_hint": "<5 words max>",
  "source_pdf": "<pdf name or unknown>"
}}

CONTEXT: {context}
HISTORY: {history}
QUESTION: {question}
JSON:"""


class ComplianceQuery(BaseModel):
    session_id: str
    query:      str

class ComplianceResponse(BaseModel):
    answer:      str
    confidence:  str
    source_hint: str
    source_pdf:  str
    chunks_used: list[str]
    history:     list[dict]


def parse_llm_json(raw):
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)

def format_history(history):
    if not history:
        return "None"
    return "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history[-6:])


@app.post("/ask-compliance", response_model=ComplianceResponse)
async def ask_compliance(payload: ComplianceQuery):
    s = sessions.get(payload.session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not s["store"].texts:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    query_vec = get_embedding(query)
    results   = s["store"].search(query_vec, k=3)
    context   = "\n\n---\n\n".join(f"[From: {r['meta'].get('source_pdf','unknown')}]\n{r['text']}" for r in results)
    prompt    = SYSTEM_PROMPT.format(context=context, history=format_history(s["history"]), question=query)

    raw_output = app.state.llm.invoke(prompt)

    try:
        parsed = parse_llm_json(raw_output.content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM non-JSON: {raw_output.content[:300]}") from exc

    answer = parsed.get("answer", "")
    s["history"].append({"role": "user",      "content": query})
    s["history"].append({"role": "assistant", "content": answer})

    return ComplianceResponse(
        answer      = answer,
        confidence  = parsed.get("confidence", "low"),
        source_hint = parsed.get("source_hint", ""),
        source_pdf  = parsed.get("source_pdf", "unknown"),
        chunks_used = [r["text"][:300] for r in results],
        history     = s["history"],
    )


@app.post("/clear-history/{session_id}")
def clear_history(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    s["history"] = []
    return {"message": "History cleared."}