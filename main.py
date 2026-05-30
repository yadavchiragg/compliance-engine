"""
Multi-Tenant RAG Compliance Engine with Web Interface
Fixed version with correct imports
"""
import os
import uuid
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vector_store import SessionVectorStore

# Initialize FastAPI
app = FastAPI(title="Multi-Tenant RAG Engine")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize components
vector_store = SessionVectorStore()
llm = None
embeddings = None

# Session storage
session_documents = {}

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    global llm, embeddings
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # FIXED: Correct embedding import
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    print("✅ Server started with Gemini integration")

# Helper function to process PDF
async def process_pdf(file: UploadFile, session_id: str) -> Dict:
    """Process uploaded PDF and add to vector store"""
    pdf_reader = PdfReader(file.file)
    documents = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            documents.append({
                'text': text,
                'page': page_num,
                'source': file.filename
            })
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        text_chunks = text_splitter.split_text(doc['text'])
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'source': doc['source'],
                    'page': doc['page'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks)
                }
            })
    
    # Generate embeddings
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)
    chunk_metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Add to vector store
    vector_store.add_documents(
        session_id=session_id,
        chunks=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas
    )
    
    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": file.filename
    }

# Web Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    session_info = vector_store.get_session_info(session_id)
    
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "session_info": session_info,
        "documents": session_documents.get(session_id, [])
    })
    
    response.set_cookie(key="session_id", value=session_id, max_age=86400*30)
    return response

@app.post("/upload")
async def upload_pdf(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and process a PDF for a session"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        result = await process_pdf(file, session_id)
        
        if session_id not in session_documents:
            session_documents[session_id] = []
        session_documents[session_id].append({
            "filename": result["filename"],
            "pages": result["pages"],
            "chunks": result["chunks"],
            "uploaded_at": str(uuid.uuid4())
        })
        
        return JSONResponse(content={
            "success": True,
            "message": f"Successfully processed {result['filename']}",
            "pages": result["pages"],
            "chunks": result["chunks"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    session_id: str
    question: str
    temperature: float = 0.1

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question based on uploaded documents"""
    query_embedding = embeddings.embed_query(request.question)
    results = vector_store.similarity_search(
        session_id=request.session_id,
        query_embedding=query_embedding,
        k=3
    )
    
    if not results:
        return JSONResponse(content={
            "answer": "Please upload a PDF document first. I don't have any documents to answer from.",
            "sources": [],
            "confidence": 0
        })
    
    # Format context
    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc['metadata'].get('source', 'Unknown')
        page = doc['metadata'].get('page', '?')
        context_parts.append(
            f"[Source: {source}, Page: {page}, Relevance: {doc['score']:.2f}]\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a compliance assistant. Answer based ONLY on the provided context.
        
Rules:
1. ONLY use information from the context below
2. If context doesn't contain the answer, say "The uploaded documents don't contain information about this."
3. Be concise and cite the source document and page number when possible
4. Return ONLY valid JSON: {{"answer": "your answer", "sources": ["source1", "source2"]}}

Context:
{context}"""),
        
        ("human", "Question: {question}\n\nAnswer in JSON:")
    ])
    
    # Generate answer
    chain = prompt_template | llm | StrOutputParser()
    response = await chain.ainvoke({
        "context": context,
        "question": request.question
    })
    
    # Parse response
    try:
        import json
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            parsed = json.loads(json_str)
            answer = parsed.get('answer', response)
            sources = parsed.get('sources', [])
        else:
            answer = response
            sources = []
    except:
        answer = response
        sources = []
    
    # Format sources for frontend
    formatted_sources = []
    for doc in results:
        formatted_sources.append({
            "text": doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
            "source": doc['metadata'].get('source', 'Unknown'),
            "page": doc['metadata'].get('page', '?'),
            "score": doc['score']
        })
    
    return JSONResponse(content={
        "answer": answer,
        "sources": formatted_sources,
        "confidence": sum(d['score'] for d in results) / len(results) if results else 0
    })

@app.get("/session-info")
async def get_session_info(session_id: str):
    """Get information about a session"""
    info = vector_store.get_session_info(session_id)
    info["uploaded_files"] = session_documents.get(session_id, [])
    return info

@app.delete("/session")
async def delete_session(session_id: str):
    """Delete a session and all its data"""
    vector_store.delete_session(session_id)
    if session_id in session_documents:
        del session_documents[session_id]
    return {"success": True, "message": "Session deleted"}