import re
import os
import json
import subprocess
from flask import Flask, render_template, request, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "student_handbook.pdf")
OLLAMA_MODEL = "llama2"
TOTAL_PAGES = 131  # For display purposes only

# -------------------------
# --- PDF extraction ------
# -------------------------
def extract_pdf_pages(pdf_path):
    if not os.path.exists(pdf_path):
        print("[WARN] PDF not found:", pdf_path)
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
    print(f"[INFO] extracted {len(pages)} pages, text pages with content: {len(full_parts)}")
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
print(f"[INFO] total chunks: {len(chunks)}")

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
index = None
if chunks:
    print("[INFO] computing embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index ready with {index.ntotal} vectors")
else:
    print("[WARN] no chunks available, index not created")

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
        err = json.dumps({
            "title": "Error",
            "answer": f"Ollama failed: {e}",
            "notes": "Check your model setup locally.",
            "sources": []
        })
        return err

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
    ctx_parts = []
    sources = []
    for pg in chosen_sorted:
        pg_text = pages[pg - 1] or ""
        ctx_parts.append(f"[Page {pg}]\n{pg_text}")
        firstline = pg_text.splitlines()[0].strip() if pg_text.splitlines() else ""
        title = firstline[:80] if firstline else f"Page {pg}"
        sources.append(f"Page {pg} of {len(pages)} – {title}")

    context = "\n\n".join(ctx_parts)
    return context, sources

def extract_pages_from_chunks(retrieved_chunks):
    sources = []
    seen = set()
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

# -------------------------
# --- Special static QA ---
# -------------------------
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

# -------------------------
# --- JSON safety helper --
# -------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to "repair" common mistakes
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
        if not text.startswith("{"):
            text = "{" + text.split("{", 1)[-1]
        if not text.endswith("}"):
            text = text.rsplit("}", 1)[0] + "}"
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

    retrieved_chunks = []
    sources = []
    if index is not None:
        q_vec = model.encode([norm_q], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            if float(score) >= min_faiss_score:
                retrieved_chunks.append(chunks[int(idx)])
        if retrieved_chunks:
            sources = extract_pages_from_chunks(retrieved_chunks)

    context = ""
    if not retrieved_chunks:
        fallback_ctx, fallback_sources = fallback_keyword_page_search(norm_q, pages, window_pages=1)
        if fallback_ctx:
            context = fallback_ctx
            sources = fallback_sources
        else:
            return {
                "title": "Information Not Found",
                "answer": "I could not find information in the handbook regarding that question. Please consult the registrar or the academic office for confirmation.",
                "sources": [],
                "notes": "Handbook search returned no relevant passages."
            }
    else:
        context = "\n\n".join(retrieved_chunks)

    # ✅ Strict JSON-only prompt
    prompt = f"""
You are Handy, the Official Camarines Sur Polytechnic Colleges Student Handbook Chatbot. 
You MUST respond in valid JSON only. 
Do not include any text before or after the JSON. 
Use this structure exactly:

{{
  "title": "short heading (max 8 words)",
  "answer": "concise but complete answer from the handbook",
  "notes": "guidance or next steps, e.g., consult registrar"
}}

If no information is found in the context, return:
{{
  "title":"Information Not Found",
  "answer":"I could not find information in the handbook.",
  "notes":"Consult registrar for confirmation."
}}

Context:
{context}

Question: {query}
"""

    raw = ollama_chat(prompt, model_name=OLLAMA_MODEL)

    parsed = {
        "title": "Answer from CSPC Student Handbook",
        "answer": "",
        "notes": "",
        "sources": sources
    }
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            json_str = match.group(0)
            temp = safe_json_parse(json_str)
            parsed["title"] = temp.get("title", parsed["title"])
            parsed["answer"] = temp.get("answer", parsed["answer"] or raw)
            parsed["notes"] = temp.get("notes", parsed["notes"])
        else:
            parsed["answer"] = raw.strip()
            parsed["notes"] = "Response did not include valid JSON."
    except Exception as e:
        parsed["answer"] = raw.strip()
        parsed["notes"] = f"Response could not be parsed into strict JSON: {e}"

    if not parsed["sources"]:
        parsed["sources"] = sources

    return parsed

# -------------------------
# --- Flask endpoints -----
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("query", "")
    if not q or not q.strip():
        return jsonify({
            "title": "No Query",
            "answer": "Please enter a question.",
            "sources": [],
            "notes": "Provide a question about the CSPC Student Handbook."
        })
    try:
        answer = get_answer(q.strip())
        return jsonify(answer)
    except Exception as e:
        return jsonify({
            "title": "Error",
            "answer": f"An internal error occurred: {e}",
            "sources": [],
            "notes": "Check server logs for details."
        })

if __name__ == "__main__":
    app.run(debug=True)
