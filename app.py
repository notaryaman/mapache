#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Aryaman Singh. All rights reserved.

import os, json, requests, re, uuid
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import yaml

# Local helpers
from memory import remember, recall, all_ns, forget

# ---------------- CONFIG ----------------
INDEX_PATH = "index/faiss.index"
META_PATH  = "index/meta.json"
EMB_MODEL  = "BAAI/bge-small-en-v1.5"

# Ollama (local LLM)
LLM_MODEL  = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Optional reranker (CPU OK). Set env RERANK_MODEL="" to disable.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# History/context budgets
MAX_HISTORY_CHARS = 6000
MAX_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = """You are UCSB Campus Assistant. Be accurate and kind.
Policy:
- Prefer internal Notes (freeform facts) when available. If a Note answers the question, use it directly. Do not mention Notes explicitly.
- Otherwise, use the specified doc scope only (scoped PDFs/HTML from the local KB). Never fabricate policy or Legal Code.
- If neither provides a clear answer, say you don’t know and point to the correct UCSB office/site if obvious.
- Tone: talk like a normal UCSB student—clear, friendly, light (no role-play).
"""

# ---------------- CACHE-BUSTERS ----------------
def _notes_version():
    try: return os.path.getmtime("notes.yaml")
    except Exception: return 0.0

# ---------------- CACHED LOADERS ----------------
@st.cache_resource
def load_index_and_model():
    # FAISS (lazy); NumPy fallback
    faiss_index = None; faiss_ok = False
    try:
        import faiss
        faiss_index = faiss.read_index(INDEX_PATH)
        faiss_ok = True
    except Exception:
        pass

    meta = []
    if os.path.exists(META_PATH):
        try: meta = json.load(open(META_PATH, "r", encoding="utf-8"))
        except Exception: meta = []

    emb_mat = None
    try: emb_mat = np.load("index/embeddings.npy")
    except Exception: emb_mat = None

    emb_model = SentenceTransformer(EMB_MODEL)
    return {"faiss_ok": faiss_ok, "faiss_index": faiss_index, "meta": meta, "emb_mat": emb_mat, "emb_model": emb_model}

@st.cache_resource
def load_reranker():
    if not RERANK_MODEL: return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANK_MODEL)
    except Exception:
        return None

# ---------- NOTES (freeform text knowledge) ----------
@st.cache_resource
def load_notes(version: float) -> List[Dict[str, Any]]:
    """Load notes.yaml and return a list of dicts with id/title/tags/text/url."""
    if not os.path.exists("notes.yaml"):
        return []
    data = yaml.safe_load(open("notes.yaml", "r", encoding="utf-8")) or {}
    notes = data.get("notes", []) or []
    out = []
    for n in notes:
        out.append({
            "id": n.get("id") or "",
            "title": n.get("title") or (n.get("id") or "Note"),
            "tags": n.get("tags") or [],
            "text": n.get("text") or "",
            "url": n.get("url") or ""
        })
    return out

@st.cache_resource
def embed_notes(version: float, emb_model_name: str):
    """Embed all notes once and cache. Returns (notes_list, embeddings_matrix)."""
    notes = load_notes(version=_notes_version())
    if not notes:
        return [], None
    model = SentenceTransformer(EMB_MODEL)
    # Embed title + tags + text for better recall
    texts = [f"{n.get('title','')} {' '.join(n.get('tags',[]))}\n{n.get('text','')}" for n in notes]
    vecs = model.encode(texts, normalize_embeddings=True)
    mat = np.array(vecs, dtype="float32")
    return notes, mat

def retrieve_notes(query: str, k: int = 3, threshold: float = 0.30) -> List[Dict[str, Any]]:
    notes, mat = embed_notes(version=_notes_version(), emb_model_name=EMB_MODEL)
    if not notes or mat is None:
        return []
    emb_model = load_index_and_model()["emb_model"]
    qv = emb_model.encode([query], normalize_embeddings=True).astype("float32")[0]
    sims = (mat @ qv)
    idx = np.argsort(-sims)[:k]
    hits = []
    for i in idx:
        score = float(sims[i])
        if score < threshold: continue
        n = notes[i].copy()
        n["score"] = score
        hits.append(n)
    return hits

def render_note_sources(notes_hits: List[Dict[str,Any]]):
    if not notes_hits: return ""
    md = ["**Sources**"]
    for i,n in enumerate(notes_hits,1):
        title = n.get("title") or "Note"
        url = n.get("url") or ""
        if url: md.append(f"{i}. [{title}]({url})")
        else:   md.append(f"{i}. {title}")
    return "\n".join(md)

# ---------------- KB RETRIEVAL (scoped PDFs/HTML) ----------------
def embed_text(q: str, emb_model: SentenceTransformer) -> np.ndarray:
    return emb_model.encode([q], normalize_embeddings=True).astype("float32")  # (1, D)

def _topk_numpy(qv: np.ndarray, mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = (mat @ qv.ravel())  # (N,)
    if k >= sims.shape[0]: idx = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k)[:k]
        idx = part[np.argsort(-sims[part])]
    return sims[idx], idx

def _matches_scope(h: Dict[str,Any], scope_patterns: List[str]) -> bool:
    if not scope_patterns: return False
    src = (h.get("source") or "") + " " + (h.get("url") or "") + " " + (h.get("path") or "")
    for patt in scope_patterns:
        try:
            if re.search(patt, src, re.I): return True
        except re.error:
            if patt.lower() in src.lower(): return True
    return False

def retrieve_scoped(query: str, scope_patterns: List[str], k: int = 8) -> List[Dict[str,Any]]:
    state = load_index_and_model(); meta = state["meta"]
    if not meta: return []
    emb_model = state["emb_model"]; qv = embed_text(query, emb_model)

    # base candidate pool
    if state["faiss_ok"] and state["faiss_index"] is not None:
        D, I = state["faiss_index"].search(qv, max(k*8, 50))
        raw = [meta[i] for i in I[0] if 0 <= i < len(meta)]
    elif isinstance(state["emb_mat"], np.ndarray):
        D, I = _topk_numpy(qv, state["emb_mat"], max(k*8, 50))
        raw = [meta[i] for i in I if 0 <= i < len(meta)]
    else:
        return []

    # filter to scope
    scoped = [h for h in raw if _matches_scope(h, scope_patterns)]

    # optional rerank
    reranker = load_reranker()
    if reranker and scoped:
        pairs = [[query, h.get("text","")] for h in scoped]
        scores = reranker.predict(pairs, convert_to_numpy=True)
        scoped = [h for h,_ in sorted(zip(scoped, scores), key=lambda x: x[1], reverse=True)]

    return scoped[:k]

def render_sources(hits: List[Dict[str,Any]]):
    if not hits: return ""
    md = ["**Sources**"]
    for i,h in enumerate(hits,1):
        label = h.get("url") or h.get("path") or h.get("source") or "source"
        url   = h.get("url") or ""
        if url: md.append(f"{i}. [{label}]({url})")
        else:   md.append(f"{i}. {label}")
    return "\n".join(md)

# ---------------- LLM ----------------
def ollama_chat(system, messages, model=LLM_MODEL, temperature=0.3):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "options": {"temperature": temperature},
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ---------------- HISTORY / TONE / CONTEXT ----------------
def build_history_messages() -> list:
    msgs = []; total = 0
    for m in reversed(st.session_state.messages):
        total += len(m["content"])
        if total > MAX_HISTORY_CHARS: break
        if m["role"] in ("user","assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return list(reversed(msgs))

def _tone_instructions(tone: str) -> str:
    if tone == "Helpful Raccoon (casual)":
        return ("Write like a friendly UCSB student. Sound human and natural. "
                "Use 1–3 short paragraphs. Avoid bullet lists unless the user asks.")
    if tone == "Procedural (steps)":
        return ("Return a concise, numbered, step-by-step procedure a UCSB student can follow. "
                "If details are missing, say you don’t know and point to the right UCSB page.")
    return "Be concise and helpful. Prefer short paragraphs. Avoid lists unless needed."

def _pack_notes(notes_hits: List[Dict[str,Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    out, used = [], 0
    for n in notes_hits:
        block = (n.get("text","") or "").strip()
        if not block: continue
        if used + len(block) + 2 > max_chars: continue
        out.append(block); used += len(block) + 2
        if used > max_chars * 0.85: break
    return "\n\n".join(out)

def _pack_docs(hits: List[Dict[str,Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    scored = []
    for h in hits:
        txt = h.get("text","").strip()
        url = h.get("url") or h.get("path")
        score = max(0, 6 - (len(txt)//500))  # prefer shorter
        scored.append((score, txt, url))
    scored.sort(key=lambda x: x[0], reverse=True)
    out, used = [], 0
    for _, txt, url in scored:
        if not txt: continue
        block = f"{txt}\nSOURCE: {url}"
        if used + len(block) + 2 > max_chars: continue
        out.append(block); used += len(block) + 2
        if used > max_chars * 0.85: break
    if not out and hits:
        h = hits[0]
        out = [f"{h.get('text','').strip()}\nSOURCE: {h.get('url') or h.get('path')}"]
    return "\n\n".join(out)

# ---------------- ANSWER LOGIC ----------------
def notes_answer(query: str, tone: str):
    notes_hits = retrieve_notes(query, k=4, threshold=0.30)
    if not notes_hits:
        return None, []
    notes_ctx = _pack_notes(notes_hits, MAX_CONTEXT_CHARS)
    style_hint = _tone_instructions(tone)
    messages = build_history_messages() + [{
        "role": "user",
        "content": (
            "Use ONLY the following Notes. Do not speculate; quote the instruction plainly. "
            "If the exact answer is present, state it directly. If not, say you don’t know.\n\n"
            f"---\n{notes_ctx}\n\n"
            "VOICE & FORMAT:\n" + style_hint
        )
    }]
    temp = 0.35 if tone == "Helpful Raccoon (casual)" else 0.25
    raw = ollama_chat(SYSTEM_PROMPT, messages, temperature=temp)
    return raw, notes_hits

def doc_scoped_answer(query: str, tone: str):
    scope_patterns = recall("facts", "doc_scope", [])
    if not scope_patterns:
        return ("I don’t have a document scope set for that yet. "
                "Add one in the sidebar or type `scope: <filename or regex>`."), []
    scoped_hits = retrieve_scoped(query, scope_patterns, k=8)
    if not scoped_hits:
        return ("I didn’t find anything in the current document scope. "
                "You can adjust the scope or share Notes for this topic."), []
    doc_ctx = _pack_docs(scoped_hits, MAX_CONTEXT_CHARS)
    style_hint = _tone_instructions(tone)
    messages = build_history_messages() + [{
        "role": "user",
        "content": (
            "Answer using ONLY the following scoped document excerpts. If not present, say you don’t know.\n\n"
            f"---\n{doc_ctx}\n\n"
            "VOICE & FORMAT:\n" + style_hint
        )
    }]
    temp = 0.30 if tone == "Procedural (steps)" else 0.35
    raw = ollama_chat(SYSTEM_PROMPT, messages, temperature=temp)
    return raw, scoped_hits

def answer(query: str, tone: str = "Helpful Raccoon (casual)"):
    # 1) NOTES ALWAYS FIRST
    ntext, nhits = notes_answer(query, tone=tone)
    if ntext:
        return ntext, nhits, "notes"

    # 2) DOC SCOPE NEXT (and only fallback)
    dtext, dhits = doc_scoped_answer(query, tone=tone)
    if dhits or ("I don’t have a document scope" not in dtext and "didn’t find anything" not in dtext):
        return dtext, dhits, "docs"

    # 3) Nothing else allowed
    return ("I don’t have that in my Notes, and there isn’t a helpful match in the current document scope. "
            "You can add a Note, or set a scope to the specific PDF that covers this."), [], "none"

# ---------------- UI & SCOPE CONTROL ----------------
st.set_page_config(page_title="Mapache v1.1", page_icon="🎓", layout="wide")
st.title("🎓 UCSB Campus Assistant")


# Session init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.write(f"Model: `{LLM_MODEL}` (change via LLM_MODEL env var)")
    if load_reranker(): st.success("Reranker: enabled")
    else:               st.info("Reranker: disabled")

    # Voice preference
    default_voice = recall("prefs", "voice", "Helpful Raccoon (casual)")
    tone = st.radio(
        "Voice & tone",
        ["Helpful Raccoon (casual)", "Neutral (concise)", "Procedural (steps)"],
        index=["Helpful Raccoon (casual)", "Neutral (concise)", "Procedural (steps)"].index(default_voice),
        help="Conversational, human tone by default."
    )
    if st.button("Save voice as default"):
        remember("prefs", "voice", tone); st.success("Saved!")

    st.markdown("---")
    # Notes status
    n = load_notes(_notes_version())
    st.caption(f"Notes loaded: {len(n)}")

    st.markdown("---")
    # Document scope control
    state = load_index_and_model()
    meta = state["meta"] or []
    all_files = sorted({(m.get("source") or "") for m in meta if m.get("source")})
    st.subheader("Doc Scope")
    scope = recall("facts", "doc_scope", [])
    if scope:
        st.success("Active scope patterns:")
        for patt in scope: st.code(patt, language="text")
    else:
        st.info("No doc scope active.")

    add_pat = st.text_input("Add scope pattern (regex or filename substring)")
    if st.button("Add to scope") and add_pat.strip():
        new_scope = (scope or []) + [add_pat.strip()]
        remember("facts", "doc_scope", new_scope)
        st.experimental_rerun()

    # Quick-pick from file basenames
    pick = st.selectbox("Quick add from known files", ["(choose)"] + all_files, index=0)
    if pick != "(choose)":
        new_scope = (scope or []) + [re.escape(pick)]
        remember("facts", "doc_scope", new_scope)
        st.experimental_rerun()

    if st.button("Clear scope"):
        forget("facts", "doc_scope"); st.experimental_rerun()

    st.markdown("---")
    st.caption("Tip: In chat, you can type `scope: <regex or filename>` or `clear scope`.")

    st.markdown("---")
    dbg = st.checkbox("Debug: show which path answered", value=False)

# Chat UI
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"]=="user" else "assistant"):
        st.markdown(m["content"])

# Parse simple chat commands for scope
def _handle_chat_commands(msg: str) -> bool:
    m = msg.strip()
    if m.lower().startswith("scope:"):
        patt = m.split(":",1)[1].strip()
        if patt:
            cur = recall("facts","doc_scope",[]) or []
            remember("facts","doc_scope", cur + [patt])
            st.session_state.messages.append({"role":"assistant","content": f"Added to scope: `{patt}`"})
            return True
    if m.lower().strip() in ("clear scope","reset scope"):
        forget("facts","doc_scope")
        st.session_state.messages.append({"role":"assistant","content": "Scope cleared."})
        return True
    return False

user_msg = st.chat_input("Ask anything (Notes first). e.g., “Where do I submit the requisition form?”")
if user_msg:
    # Commands
    if _handle_chat_commands(user_msg):
        pass
    else:
        st.session_state.messages.append({"role":"user","content": user_msg})
        with st.spinner("Thinking..."):
            try:
                text, sources, path = answer(user_msg, tone=tone)
                st.session_state.messages.append({"role":"assistant","content": text})
                with st.chat_message("assistant"):
                    if dbg: st.caption(f"path: {path}")
                    st.markdown(text)
                    # Sources
                    if sources and isinstance(sources, list):
                        if path == "notes":
                            srcs = render_note_sources(sources)
                        else:
                            srcs = render_sources(sources)
                        if srcs:
                            st.divider(); st.markdown(srcs)
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(str(e))
