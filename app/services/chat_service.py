# app/services/chat_service.py
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from ..core.config import Settings
from ..core.inference.client import RouterRequestsClient
from ..core.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# --- Optional cross-encoder reranker (graceful fallback) ---
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

SYSTEM_PROMPT = (
    "You are MATRIX-AI, a concise assistant for the Matrix EcoSystem.\n"
    "Answer the user's question directly in 2–4 short sentences.\n"
    "Do NOT restate the question. Do NOT use labels like 'Question:' or 'Answer:'.\n"
    "Use the provided CONTEXT if present; if the answer is not supported by it, say you don't know.\n"
    "Do not ask follow-up questions unless the user explicitly asks you to."
)

# Thread-safe singleton retriever
_retriever_instance: Optional[Retriever] = None
_retriever_lock = threading.Lock()


def get_retriever(settings: Settings) -> Optional[Retriever]:
    """Initialize and return a single Retriever instance (double-checked locking)."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    kb_path = os.getenv("RAG_KB_PATH", "data/kb.jsonl")
    if not Path(kb_path).exists():
        logger.info("RAG KB not found at %s — running LLM-only.", kb_path)
        return None

    with _retriever_lock:
        if _retriever_instance is not None:
            return _retriever_instance
        try:
            _retriever_instance = Retriever(kb_path=kb_path, top_k=settings.rag.top_k)
            logger.info("RAG enabled with KB at %s (top_k=%d)", kb_path, settings.rag.top_k)
        except Exception as e:
            logger.warning("RAG disabled (failed to initialize Retriever: %s)", e)
            _retriever_instance = None
    return _retriever_instance


# ----------------------------
# Anti-repetition + de-label helpers
# ----------------------------
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')
_NORM = re.compile(r'[^a-z0-9\s]+')
_QA_LINE_RE = re.compile(r'^\s*(question|q|user)\s*:\s*', re.I)
_ANSWER_PREFIX_RE = re.compile(r'^\s*(answer|a)\s*:\s*', re.I)

def _norm_sentence(s: str) -> str:
    s = s.lower().strip()
    s = _NORM.sub(' ', s)
    return re.sub(r'\s+', ' ', s)

def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))

def _strip_qa_meta(text: str) -> str:
    """Drop lines like 'Question: ...' and leading 'Answer:' labels."""
    lines = text.splitlines()
    out: List[str] = []
    for i, l in enumerate(lines):
        if i == 0:
            l = _ANSWER_PREFIX_RE.sub('', l).strip()
        if _QA_LINE_RE.match(l):
            continue
        out.append(l)
    return "\n".join(out).strip()

def _remove_query_echo(text: str, query: str, sim_threshold: float = 0.9) -> str:
    """Remove sentences that are near-duplicates of the original query."""
    qn = _norm_sentence(query)
    parts = _SENT_SPLIT.split(re.sub(r'\s+', ' ', text).strip()) or [text]
    kept: List[str] = []
    for s in parts:
        sn = _norm_sentence(s)
        if _jaccard(qn, sn) >= sim_threshold:
            continue
        kept.append(s.strip())
    return ' '.join(kept).strip()

def _squash_repetition(text: str, max_sentences: int = 4, sim_threshold: float = 0.88) -> str:
    """Remove near-duplicate sentences while keeping order and cap total sentences."""
    t = re.sub(r'\s+', ' ', text).strip()
    if not t:
        return t
    parts = _SENT_SPLIT.split(t)
    out: List[str] = []
    norms: List[str] = []
    for s in parts:
        ns = _norm_sentence(s)
        if not ns:
            continue
        if any(_jaccard(prev, ns) >= sim_threshold for prev in norms):
            continue
        out.append(s.strip())
        norms.append(ns)
        if len(out) >= max_sentences:
            break
    return ' '.join(out).strip()

def _clean_answer(text: str, query: str) -> str:
    t = _strip_qa_meta(text)
    t = _remove_query_echo(t, query)
    t = _squash_repetition(t, max_sentences=4, sim_threshold=0.88)
    return t


# ----------------------------
# RAG helpers (query expansion, ranking, snippets)
# ----------------------------
_ALIAS_TABLE: Dict[str, List[str]] = {
    "matrixhub": ["matrix hub", "hub api", "catalog", "registry", "cas"],
    "mcp": ["model context protocol", "manifest", "server manifest", "admin api"],
    "agent-matrix": ["matrix agents", "matrix ecosystem", "matrix toolkit"],
}
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def _normalize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]

def _expand_query(q: str) -> str:
    ql = q.lower()
    extras: List[str] = []
    for canon, variants in _ALIAS_TABLE.items():
        if any(v in ql for v in ([canon] + variants)):
            extras.extend([canon] + variants)
    if extras:
        return q + " | " + " ".join(sorted(set(extras)))
    return q

def _keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_normalize(query))
    d_tokens = set(_normalize(text))
    if not q_tokens or not d_tokens:
        return 0.0
    inter = len(q_tokens & d_tokens)
    union = len(q_tokens | d_tokens)
    return inter / max(1, union)

def _domain_boost(text: str) -> float:
    t = text.lower()
    boost = 0.0
    for term in ("matrixhub", "hub api", "catalog", "mcp", "server manifest", "cas"):
        if term in t:
            boost += 0.05
    return min(boost, 0.25)

def _best_paragraphs(text: str, query: str, max_chars: int = 700) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return text[:max_chars]
    scored = [(p, _keyword_overlap_score(query, p)) for p in paras]
    scored.sort(key=lambda x: x[1], reverse=True)
    picked: List[str] = []
    used = 0
    for p, _s in scored[:4]:
        if used >= max_chars:
            break
        picked.append(p)
        used += len(p) + 2
        if used >= max_chars or len(picked) >= 2:
            break
    return "\n".join(picked)

def _cross_encoder_scores(
    model: Optional["CrossEncoder"], query: str, docs: List[Dict], max_pairs: int = 50
) -> Optional[List[float]]:
    if not model:
        return None
    try:
        pairs = [(query, d["text"][:1200]) for d in docs[:max_pairs]]
        return list(model.predict(pairs))
    except Exception as e:
        logger.warning("Cross-encoder scoring failed; continuing without it (%s)", e)
        return None

def _rerank_docs(
    docs: List[Dict], query: str, k_final: int, reranker: Optional["CrossEncoder"] = None
) -> List[Dict]:
    if not docs:
        return []
    vec_scores = [float(d.get("score", 0.0)) for d in docs]
    if vec_scores:
        vmin, vmax = min(vec_scores), max(vec_scores)
        rng = max(1e-6, (vmax - vmin))
        vec_norm = [(v - vmin) / rng for v in vec_scores]
    else:
        vec_norm = [0.0] * len(docs)

    lex_scores = [_keyword_overlap_score(query, d["text"]) for d in docs]
    boosts = [_domain_boost(d["text"]) for d in docs]

    ce_scores = _cross_encoder_scores(reranker, query, docs)
    if ce_scores:
        cmin, cmax = min(ce_scores), max(ce_scores)
        crng = max(1e-6, (cmax - cmin))
        ce_norm = [(c - cmin) / crng for c in ce_scores]
    else:
        ce_norm = None

    merged: List[Tuple[float, Dict]] = []
    for i, d in enumerate(docs):
        score = 0.55 * vec_norm[i] + 0.35 * lex_scores[i] + 0.10 * boosts[i]
        if ce_norm is not None:
            score = 0.80 * score + 0.20 * ce_norm[i]
        merged.append((score, d))
    merged.sort(key=lambda x: x[0], reverse=True)
    return [d for _s, d in merged[:k_final]]

def _build_context_from_docs(docs: List[Dict], query: str, max_blocks: int = 4) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    sources: List[str] = []
    for i, d in enumerate(docs[:max_blocks]):
        snip = _best_paragraphs(d["text"], query, max_chars=700)
        src = d.get("source", f"kb:{i}")
        blocks.append(f"[{i+1}] {snip}\n(source: {src})")
        sources.append(src)
    if not blocks:
        return "", []
    prelude = "CONTEXT (use only these facts; if missing, say you don't know):"
    return prelude + "\n\n" + "\n\n".join(blocks), sources


# ----------------------------
# Service
# ----------------------------
class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RouterRequestsClient(
            model=settings.model.name,
            fallback=settings.model.fallback,
            provider=getattr(settings.model, "provider", None),
            max_retries=2,
            connect_timeout=10.0,
            read_timeout=60.0,
        )
        self.retriever = get_retriever(settings)

        self.reranker = None
        use_rerank = os.getenv("RAG_RERANK", "true").lower() in ("1", "true", "yes")
        if use_rerank and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
                logger.info("RAG cross-encoder reranker enabled.")
            except Exception as e:
                logger.warning("Reranker disabled: %s", e)

        # default inference knobs to reduce repetition
        self._stop = ["\nQuestion:", "\nUser:", "\nQ:", "\nAnswer:", "\nA:"]
        self._extra = {"frequency_penalty": 0.2, "presence_penalty": 0.0}

    # ---------- RAG core ----------
    def _retrieve_best(self, query: str) -> Tuple[str, List[str]]:
        if not self.retriever:
            return "", []
        expanded = _expand_query(query)
        k_base = max(4, int(self.settings.rag.top_k) * 5)
        try:
            cands = self.retriever.retrieve(expanded, k=k_base)
        except Exception as e:
            logger.warning("Retriever failed (%s); falling back to LLM-only.", e)
            return "", []
        if not cands:
            return "", []
        top = _rerank_docs(cands, query, k_final=max(3, self.settings.rag.top_k), reranker=self.reranker)
        ctx, sources = _build_context_from_docs(top, query, max_blocks=max(3, self.settings.rag.top_k))
        return ctx, sources

    def _augment(self, query: str) -> Tuple[str, List[str]]:
        ctx, sources = self._retrieve_best(query)
        if ctx:
            # No Q:/A: labels — just a clear directive + the raw question
            user_msg = (
                f"{ctx}\n\n"
                "Using only the context above, respond concisely (2–4 sentences) to this query.\n"
                f"{query}"
            )
        else:
            user_msg = (
                "Respond concisely (2–4 sentences). Do not restate the question or add labels.\n"
                f"{query}"
            )
        return user_msg, sources

    # ---------- Non-stream ----------
    def answer_with_sources(self, query: str) -> Tuple[str, List[str]]:
        user_msg, sources = self._augment(query)
        text = self.client.chat_nonstream(
            SYSTEM_PROMPT,
            user_msg,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
            stop=self._stop,
            extra=self._extra,
        )
        text = _clean_answer(text, query)
        return text, sources

    # ---------- Stream ----------
    def stream_answer(self, query: str):
        """
        Stream while cleaning: suppress Q/A labels and near-duplicate lines as they appear.
        """
        user_msg, _ = self._augment(query)
        raw = self.client.chat_stream(
            SYSTEM_PROMPT,
            user_msg,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
            stop=self._stop,
            extra=self._extra,
        )

        buf = ""      # collected raw
        emitted = ""  # cleaned we already sent
        for token in raw:
            if not token:
                continue
            buf += token
            cleaned = _clean_answer(buf, query)
            if len(cleaned) < len(emitted):
                # parser got stricter; resync
                emitted = cleaned
                continue
            delta = cleaned[len(emitted):]
            if delta:
                emitted = cleaned
                yield delta
