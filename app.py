import re
import os
import json
import requests
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------
# --- Config --------------
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "student_handbook.pdf")

HPC_API_URL = "http://180.193.216.136:5000/ollama_chat"
OLLAMA_MODEL = "llama2"
TOTAL_PAGES = 131

# -------------------------
# --- PDF extraction ------
# -------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_pages(pdf_path):
    if not os.path.exists(pdf_path):
        return "", []

    pages = []
    full_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            clean = text.strip()
            pages.append(clean)
            if clean:
                full_parts.append(f"[Page {i}]\n{clean}")
    return "\n\n".join(full_parts), pages

full_handbook_text, pages = extract_pdf_pages(pdf_path)

# -------------------------
# --- Chunking & vectors --
# -------------------------
@st.cache_resource(show_spinner=False)
def build_index(pages, chunk_size=800, overlap=100):
    chunks = []
    for i, ptext in enumerate(pages, start=1):
        cleaned = re.sub(r"\s+", " ", ptext).strip()
        start = 0
        while start < len(cleaned):
            piece = cleaned[start:start + chunk_size]
            if piece.strip():
                chunks.append(f"[Page {i}] {piece}")
            start += chunk_size - overlap
        if not cleaned:
            chunks.append(f"[Page {i}] ")

    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, chunks

model, index, chunks = build_index(pages)

# -------------------------
# --- HPC API -------------
# -------------------------
def ollama_chat(prompt, model_name=OLLAMA_MODEL, timeout=60):
    try:
        response = requests.post(
            HPC_API_URL,
            json={"prompt": prompt, "model": model_name},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("output", "")
    except Exception as e:
        return json.dumps({
            "title": "Error",
            "answer": f"HPC backend request failed: {e}",
            "notes": "Running in fallback mode (retrieved context only).",
            "sources": []
        })

# -------------------------
# --- Retrieval helpers ---
# -------------------------
def extract_pages_from_chunks(retrieved_chunks):
    sources, seen = [], set()
    for chunk in retrieved_chunks:
        m = re.search(r"\[Page\s+(\d+)\]", chunk)
        if m:
            p = int(m.group(1))
            if p not in seen:
                seen.add(p)
                clean_text = re.sub(r"\[Page \d+\]", "", chunk).strip()
                title = clean_text[:80] if clean_text else f"Page {p}"
                sources.append(f"Page {p} of {len(pages)} – {title}")
    return sources

def fallback_keyword_search(query, pages, window_pages=1):
    q = query.lower()
    tokens = re.findall(r"[a-z0-9]+", q)
    if not tokens:
        return "", []

    page_scores = []
    for i, p in enumerate(pages, start=1):
        score = sum((p or "").lower().count(t) for t in tokens)
        if score > 0:
            page_scores.append((i, score))

    if not page_scores:
        return "", []

    top_page = sorted(page_scores, key=lambda x: x[1], reverse=True)[0][0]
    chosen = range(max(1, top_page - window_pages), min(len(pages), top_page + window_pages) + 1)

    ctx_parts, sources = [], []
    for pg in chosen:
        pg_text = pages[pg - 1] or ""
        ctx_parts.append(f"[Page {pg}]\n{pg_text}")
        sources.append(f"Page {pg} of {len(pages)}")
    return "\n\n".join(ctx_parts), sources

# -------------------------
# --- Core QA -------------
# -------------------------
def get_answer(query, use_hpc=True, k=6, min_faiss_score=0.15):
    # vector search
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k)

    retrieved_chunks = [
        chunks[idx] for score, idx in zip(D[0], I[0])
        if idx >= 0 and idx < len(chunks) and float(score) >= min_faiss_score
    ]

    sources = extract_pages_from_chunks(retrieved_chunks)
    context = "\n\n".join(retrieved_chunks)

    if not context:
        context, sources = fallback_keyword_search(query, pages)

    if not use_hpc:  # fallback mode
        return {
            "title": "Retrieved Context",
            "answer": context[:800] + ("..." if len(context) > 800 else ""),
            "notes": "No HPC backend used. This is raw context only.",
            "sources": sources
        }

    prompt = f"""
You are Handy, the CSPC Student Handbook Chatbot.
Respond strictly in JSON format:

{{
  "title": "short heading",
  "answer": "concise but complete answer from the handbook",
  "notes": "guidance or next steps"
}}

Context:
{context}

Question: {query}
"""
    raw = ollama_chat(prompt, model_name=OLLAMA_MODEL)
    try:
        parsed = json.loads(raw)
    except:
        parsed = {"title": "Parse Error", "answer": raw, "notes": "Could not parse JSON."}

    parsed["sources"] = parsed.get("sources", sources)
    return parsed

# -------------------------
# --- Streamlit UI --------
# -------------------------
st.set_page_config(page_title="Handy – CSPC Handbook Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Handy – CSPC Student Handbook Chatbot")
st.caption("Ask authoritative questions about the CSPC Student Handbook.")

query = st.text_input("Enter your question:", placeholder="e.g., What are the incentives for athletes?")
use_hpc = st.toggle("Use HPC backend", value=True)

if st.button("Ask") and query:
    with st.spinner("Searching handbook..."):
        response = get_answer(query, use_hpc=use_hpc)

    st.subheader(response["title"])
    st.write(response["answer"])
    if response.get("notes"):
        st.info(response["notes"])
    if response.get("sources"):
        st.write("📚 **Sources**")
        for src in response["sources"]:
            st.caption(src)
