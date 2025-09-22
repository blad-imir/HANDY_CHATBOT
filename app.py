import re
import os
import json
import subprocess
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "student_handbook.pdf")
OLLAMA_MODEL = "llama2"
TOTAL_PAGES = 131  # For display purposes only

# -------------------------
# --- PDF extraction ------
# -------------------------
def extract_pdf_pages(pdf_path):
    if not os.path.exists(pdf_path):
        st.warning(f"[WARN] PDF not found: {pdf_path}")
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
    full_text = "\n\n".join(full_parts)
    return full_text, pages

full_handbook_text, pages = extract_pdf_pages(pdf_path)

# -------------------------
# --- Chunking & vectors --
# -------------------------
def chunk_text_by_page(pages, chunk_size=900, overlap=150):
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
    return chunks

chunks = chunk_text_by_page(pages, chunk_size=800, overlap=100)

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
index = None
if chunks:
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

# -------------------------
# --- Ollama helper -------
# -------------------------
def ollama_chat(prompt, model_name=OLLAMA_MODEL, timeout=None):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=timeout
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return json.dumps({
            "title": "Error",
            "answer": f"Ollama failed: {e}",
            "notes": "Check your model setup locally.",
            "sources": []
        })

# -------------------------
# --- Utilities -----------
# -------------------------
def normalize_query_for_synonyms(query):
    q = query.lower()
    synonyms = {
        "failing grades": ["failing marks", "failed subject", "5.0 grade", "failed grade", "fail grade", "failed coursework"],
        "academic probation": ["probation", "on probation", "academic warning"],
        "dismissal": ["expulsion", "removal", "dismissed"],
        "attendance": ["absences", "attendance policy", "class attendance"]
    }
    for canonical, variants in synonyms.items():
        for v in variants:
            if v in q:
                return canonical
    return query

def fallback_keyword_page_search(query, pages, window_pages=2):
    q = query.lower()
    tokens = re.findall(r"[a-z0-9]+", q)
    if not tokens:
        return "", []

    page_scores = []
    for i, p in enumerate(pages, start=1):
        text_l = (p or "").lower()
        score = sum(text_l.count(t) for t in tokens)
        if score > 0:
            page_scores.append((i, score))

    if not page_scores:
        return "", []

    page_scores.sort(key=lambda x: x[1], reverse=True)
    top_pages = [pnum for pnum, s in page_scores[:5]]

    chosen = set()
    for pnum in top_pages:
        for pg in range(max(1, pnum - window_pages), min(len(pages), pnum + window_pages) + 1):
            chosen.add(pg)

    chosen_sorted = sorted(chosen)
    ctx_parts, sources = [], []
    for pg in chosen_sorted:
        pg_text = pages[pg - 1] or ""
        ctx_parts.append(f"[Page {pg}]\n{pg_text}")
        firstline = pg_text.splitlines()[0].strip() if pg_text.splitlines() else ""
        title = firstline[:80] if firstline else f"Page {pg}"
        sources.append(f"Page {pg} of {len(pages)} – {title}")

    return "\n\n".join(ctx_parts), sources

def extract_pages_from_chunks(retrieved_chunks):
    sources, seen = [], set()
    for chunk in retrieved_chunks:
        m = re.search(r"\[Page\s+(\d+)\]", chunk)
        if m:
            p = int(m.group(1))
            if p not in seen:
                seen.add(p)
                clean_text = re.sub(r"\[Page \d+\]", "", chunk).strip()
                heading_match = re.search(r"(CHAPTER [^\n]+|[A-Z][A-Z\s,;&\-]{5,})", clean_text)
                if heading_match:
                    title = heading_match.group(1).strip()
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
                    title = sentences[0][:80] if sentences else ""
                sources.append(f"Page {p} of {len(pages)} – {title}")
    return sources

def handle_special_queries(query):
    q = query.lower().strip()
    if q in ["who are you", "what are you", "introduce yourself"]:
        return {
            "title": "Official CSPC Student Handbook Chatbot",
            "answer": (
                "I am Handy — the Official CSPC Student Handbook Chatbot. "
                "I provide authoritative answers strictly from the CSPC Student Handbook. "
                "If I cannot find an answer in the handbook, I will tell you to consult the registrar."
            ),
            "sources": [],
            "notes": "I answer only using the official handbook content."
        }
    return None

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return {
            "title": "Parse Error",
            "answer": "The response could not be formatted into valid JSON.",
            "notes": "Please rephrase your question."
        }

# -------------------------
# --- Core: get_answer ----
# -------------------------
def get_answer(query, k=6, min_faiss_score=0.15):
    special = handle_special_queries(query)
    if special:
        return special

    norm_q = normalize_query_for_synonyms(query)
    retrieved_chunks, sources = [], []

    if index is not None:
        q_vec = model.encode([norm_q], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k)
        for score, idx in zip(D[0], I[0]):
            if idx >= 0 and idx < len(chunks) and float(score) >= min_faiss_score:
                retrieved_chunks.append(chunks[int(idx)])
        if retrieved_chunks:
            sources = extract_pages_from_chunks(retrieved_chunks)

    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""
    if not retrieved_chunks:
        fallback_ctx, fallback_sources = fallback_keyword_page_search(norm_q, pages, window_pages=1)
        if fallback_ctx:
            context, sources = fallback_ctx, fallback_sources
        else:
            return {
                "title": "Information Not Found",
                "answer": "I could not find information in the handbook.",
                "sources": [],
                "notes": "Consult registrar for confirmation."
            }

    # strict JSON prompt
    prompt = f"""
You are Handy, the Official Camarines Sur Polytechnic Colleges Student Handbook Chatbot. 
You MUST respond in valid JSON only, using this structure:

{{
  "title": "short heading (max 8 words)",
  "answer": "concise but complete answer from the handbook",
  "notes": "guidance or next steps"
}}

Context:
{context}

Question: {query}
"""

    raw = ollama_chat(prompt, model_name=OLLAMA_MODEL)
    try:
        parsed = safe_json_parse(raw)
    except:
        parsed = {
            "title": "Parse Error",
            "answer": raw,
            "notes": "Response could not be parsed into JSON."
        }

    parsed["sources"] = parsed.get("sources", sources)
    return parsed

# -------------------------
# --- Streamlit UI --------
# -------------------------
st.set_page_config(page_title="Handy – CSPC Handbook Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Handy – CSPC Student Handbook Chatbot")
st.caption("Ask authoritative questions about the CSPC Student Handbook.")

query = st.text_input("Enter your question:", placeholder="e.g., What are the incentives for athletes?")
if st.button("Ask") and query:
    with st.spinner("Searching handbook..."):
        response = get_answer(query)
    st.subheader(response["title"])
    st.write(response["answer"])
    if response.get("notes"):
        st.info(response["notes"])
    if response.get("sources"):
        st.write("📚 **Sources**")
        for src in response["sources"]:
            st.caption(src)
