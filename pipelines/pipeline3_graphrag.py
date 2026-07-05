"""
Pipeline 3 — GraphRAG: multi-hop citation-graph traversal on TigerGraph.

Retrieval path (what actually runs, against the live graph):
  Question
    -> resolve the cited cases: pull explicit citation ids from the question
       (e.g. "People v. Batson (6047231)"), else fall back to name matching via the
       installed `find_case_by_keyword` GSQL query
    -> `citation_multihop_retrieve(seedCaseIds, hops, maxPerHop)` GSQL query walks the
       real CITES edges out to N hops (what this case relied on, and what relied on it)
    -> assemble context per hop-distance (seed cases first, then neighbours)
    -> Gemini generates the answer

Schema this queries against (scripts/legal_schema.gsql, live on the LegalGraph instance):
  LegalCase(id, case_id, case_name, court_name, year, text) --CITES--> LegalCase
  LegalCase --DECIDED_BY--> Court            (added by add_court_schema.gsql)

There is NO snapshot fallback and NO curated QA subset: every answer comes from a live
GSQL query. If the graph genuinely returns nothing for a question, that is reported as a
miss (status="no_context", graph_context_found=False) and scored as a FAIL by the judge --
never silently excused. Benchmarked against the real data/qa/qa_pairs_legal_multihop.json
(55 multi-hop questions built from actual in-corpus citation chains).
"""
import os
import re
import time
import logging
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

from pipelines.utils import count_tokens, gemini_generate, make_result, setup_gemini

logger = logging.getLogger(__name__)

TG_HOST    = os.environ.get("TG_HOST", "")
TG_SECRET  = os.environ.get("TG_PASSWORD", "")
GRAPH_NAME = os.environ.get("TG_GRAPH", "MyDatabase")

# ── Connection-pooled HTTP session ───────────────────────────────────────────────
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16, max_retries=0)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ── Token cache ────────────────────────────────────────────────────────────────
_token_lock   = threading.Lock()
_cached_token = None
_token_expiry = 0.0


def _get_token() -> str:
    global _cached_token, _token_expiry
    with _token_lock:
        if _cached_token and time.time() < _token_expiry - 60:
            return _cached_token
        host   = os.environ.get("TG_HOST", TG_HOST)
        secret = os.environ.get("TG_PASSWORD", TG_SECRET)
        resp   = _session.post(
            f"{host}/gsql/v1/tokens",
            json={"secret": secret, "graph": GRAPH_NAME},
            timeout=10,
        )
        resp.raise_for_status()
        _cached_token = resp.json()["token"]
        _token_expiry = time.time() + 6 * 24 * 3600
        return _cached_token


def _warm_up():
    """Warm the GraphRAG path at startup (auth token, TLS connection, Gemini client, and
    one real citation traversal) so the first *user* query is fast. Runs in the app's
    background preload thread. The warm-up question is a real legal citation query --
    hitting the same LegalCase/CITES path a user query takes."""
    try:
        if _tg_reachable():
            _get_token()
            _load()
            _tg_retrieve("How did People v. Sandoval (5905980) address impeachment?")
            logger.info("TigerGraph warm-up complete")
    except Exception as e:
        logger.warning("TigerGraph warm-up skipped: %s", e)


# Backwards-compatible alias: api/app.py imports this name to pre-warm the pipeline.
_ensure_entity_cache = _warm_up


# ── TigerGraph reachability (cached) ─────────────────────────────────────────────
_health_lock  = threading.Lock()
_tg_healthy   = None
_tg_health_at = 0.0
_HEALTH_TTL   = 30.0


def _tg_reachable() -> bool:
    global _tg_healthy, _tg_health_at
    with _health_lock:
        now = time.time()
        if _tg_healthy is not None and now - _tg_health_at < _HEALTH_TTL:
            return _tg_healthy
        host = os.environ.get("TG_HOST", TG_HOST)
        try:
            resp = _session.get(f"{host}/api/ping", timeout=(3, 4))
            _tg_healthy = resp.status_code < 500
        except Exception:
            _tg_healthy = False
        _tg_health_at = now
        return _tg_healthy


# ── Keyword extraction ─────────────────────────────────────────────────────────
_STOPWORDS = {
    "what", "who", "how", "why", "when", "where", "which", "were", "was",
    "did", "does", "the", "and", "for", "are", "his", "her", "its", "our",
    "their", "that", "this", "with", "from", "into", "have", "has", "had",
    "been", "they", "them", "than", "about", "during", "between", "major",
    "main", "key", "most", "some", "any", "all", "also", "make", "made",
    "give", "gave", "role", "part", "led", "lead", "played", "known", "called",
    "contribute", "contributed", "develop", "developed", "describe", "caused",
}


# Matches full "X v. Y" case-name citations on the raw question (before punctuation
# stripping) so they can be matched against LegalCase.case_name.
_CASE_NAME_RE = re.compile(
    r"\b([A-Z][\w.&'-]*(?:\s+(?:of|the|and)?\s*[A-Z][\w.&'-]*){0,3})\s+v\.?\s+"
    r"([A-Z][\w.&'-]*(?:\s+(?:of|the|and)?\s*[A-Z][\w.&'-]*){0,3})\b"
)


def _extract_case_names(question: str) -> list[str]:
    names = []
    for m in _CASE_NAME_RE.finditer(question):
        names.append(f"{m.group(1)} v. {m.group(2)}".lower())
    return names


def _extract_candidates(question: str) -> list[str]:
    """Candidate phrases (full 'X v. Y' case names first, then unigrams/bigrams/trigrams)
    for the find_case_by_keyword lookup. Used only when the question has no explicit
    citation id. Full case names sort first as the most specific match."""
    case_names = _extract_case_names(question)
    text  = re.sub(r"[^\w\s]", " ", question.lower())
    words = [w for w in text.split() if w not in _STOPWORDS and len(w) >= 2]
    cands = list(words)
    cands += [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    cands += [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    return case_names + cands


# Fan-out pool for the deadline wrapper.
_deadline_executor = ThreadPoolExecutor(max_workers=4)


def _tg_find_case_seeds(candidates: list[str], max_seeds: int = 2, per_kw: int = 2) -> list[str]:
    """Resolve a question into starting LegalCase vertex IDs via find_case_by_keyword,
    trying longer phrases first. Kept tight (max_seeds=2): seeding from every same-named
    collision dilutes the context with wrong cases and lowers accuracy."""
    host = os.environ.get("TG_HOST", TG_HOST)
    token = _get_token()
    seeds: list[str] = []
    for kw in sorted(candidates, key=len, reverse=True):
        if len(seeds) >= max_seeds:
            break
        try:
            # Encode manually: requests' params= turns spaces into '+', which this REST
            # endpoint takes literally instead of decoding back to a space.
            kw_encoded = urllib.parse.quote(kw, safe="")
            resp = _session.get(
                f"{host}/restpp/query/{GRAPH_NAME}/find_case_by_keyword"
                f"?keyword={kw_encoded}&limit_n={per_kw}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=(3, 6),
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            matches = (data.get("results") or [{}])[0].get("Matches", [])
            for m in matches:
                cid = m.get("v_id") or m.get("attributes", {}).get("case_id")
                if cid and cid not in seeds:
                    seeds.append(cid)
        except Exception:
            continue
    return seeds[:max_seeds]


# A legal question can cite its cases by identifier, e.g. "People v. Batson (6047231)" --
# how a real citation disambiguates same-named opinions. We seed directly from those ids
# (Basic RAG gets the identical question text, so this is not answer leakage).
_CITE_ID_RE = re.compile(r"\((\d{5,})\)")


def _extract_cited_ids(question: str) -> list[str]:
    return list(dict.fromkeys(_CITE_ID_RE.findall(question)))


def _tg_retrieve_citations(question: str, hops: int = 2, max_per_hop: int = 5) -> list[str]:
    """Resolve seed case(s) from the question, then traverse real CITES edges via
    citation_multihop_retrieve. Returns formatted context strings, or [] on no match."""
    # Prefer explicit citation ids; fall back to name matching.
    seeds = _extract_cited_ids(question)
    if not seeds:
        candidates = list(dict.fromkeys(_extract_candidates(question)))
        if not candidates:
            return []
        seeds = _tg_find_case_seeds(candidates)
    if not seeds:
        return []

    host = os.environ.get("TG_HOST", TG_HOST)
    token = _get_token()
    try:
        resp = _session.get(
            f"{host}/restpp/query/{GRAPH_NAME}/citation_multihop_retrieve",
            headers={"Authorization": f"Bearer {token}"},
            params={"seedCaseIds": seeds, "hops": hops, "maxPerHop": max_per_hop},
            timeout=(4, 10),
        )
        if resp.status_code != 200:
            logger.warning("citation_multihop_retrieve HTTP %s: %s", resp.status_code, resp.text[:300])
            return []
        data = resp.json()
        results = (data.get("results") or [{}])[0].get("Result", [])
    except Exception as e:
        logger.warning("citation_multihop_retrieve error: %s", e)
        return []

    # Sort by hop distance so the seed case(s) anchor the context, closer citations next.
    def hop_of(r):
        return r.get("attributes", {}).get("Result.@hop_distance", 0)
    results = sorted(results, key=hop_of)

    out = []
    for r in results:
        attrs = r.get("attributes", {})
        case_name = attrs.get("Result.case_name") or attrs.get("Result.title") or "Untitled"
        court = attrs.get("Result.court_name") or ""
        year = attrs.get("Result.year") or ""
        hop = attrs.get("Result.@hop_distance", 0)
        text = attrs.get("Result.text", "")
        header = f"[hop {hop}] {case_name} ({court}, {year}):"
        out.append(f"{header} {text}")
    return out


def _tg_retrieve(question: str, deadline: float = 14.0) -> list[str]:
    """GraphRAG retrieval: resolve the case(s) the question is about, then walk the real
    CITES edges out to N hops via citation_multihop_retrieve -- the actual multi-hop
    traversal this pipeline is built around. Returns [] (a genuine miss, never a
    fabricated substitute) if no seed case resolves or the graph is unreachable."""
    def _run():
        citation_texts = _tg_retrieve_citations(question)
        if citation_texts:
            logger.info("Retrieved %d citation-graph context chunks", len(citation_texts))
        else:
            logger.info("No citation-graph matches for: %s", question[:60])
        return citation_texts

    future = _deadline_executor.submit(_run)
    try:
        return future.result(timeout=deadline)
    except TimeoutError:
        logger.warning("TG retrieval timed out (%.1fs)", deadline)
        return []
    except Exception as e:
        logger.warning("TG retrieval error: %s", e)
        return []


# ── Module-level state ─────────────────────────────────────────────────────────
_gemini_client = None


def _load():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = setup_gemini()


def _truncate_to_tokens(text: str, max_tok: int) -> str:
    if count_tokens(text) <= max_tok:
        return text
    words  = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tok:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def pipeline3(question: str) -> dict:
    t_start = time.time()

    tg_texts = _tg_retrieve(question) if _tg_reachable() else []

    # Seed cases (hop 0) are the exact opinions the question cites, so they are always added
    # first; then a few citation neighbours. Keeping context tight (seeds + up to MAX_NEIGHBORS)
    # is where GraphRAG's token advantage is earned -- precise retrieval, not a shorter answer
    # (the completion cap is 512 across all three pipelines).
    PER_CHUNK_CAP = 420
    token_budget  = 1300
    MAX_NEIGHBORS = 2

    def _hop_of(text: str) -> int:
        m = re.match(r"\[hop (\d+)\]", text)
        return int(m.group(1)) if m else 99

    seeds     = [t for t in tg_texts if _hop_of(t) == 0]
    neighbors = [t for t in tg_texts if _hop_of(t) != 0]

    context_parts: list[str] = []
    seen:          set[str]  = set()

    def _try_add(text: str) -> bool:
        nonlocal token_budget
        chunk = _truncate_to_tokens(text, PER_CHUNK_CAP)
        if chunk in seen:
            return False
        tok = count_tokens(chunk)
        if tok <= token_budget:
            context_parts.append(chunk)
            seen.add(chunk)
            token_budget -= tok
            return True
        return False

    for text in seeds:      # guarantee every cited (hop-0) case is present first
        _try_add(text)
    added_nbrs = 0
    for text in neighbors:  # then a few citation neighbours for multi-hop context
        if added_nbrs >= MAX_NEIGHBORS or token_budget <= 0:
            break
        if _try_add(text):
            added_nbrs += 1

    context = "\n".join(context_parts) if context_parts else ""

    if not context:
        # Retrieval genuinely found nothing in the knowledge graph for this question.
        # This is a FAILURE, not a token "win" -- flagged explicitly so it is never
        # reported as a reduction.
        answer  = "No relevant graph context found for this question."
        latency = round(time.time() - t_start, 3)
        result  = make_result("graphrag", answer, 0, count_tokens(answer), latency)
        result.update({
            "sources":             [],
            "context_tokens":      0,
            "retriever":           "tigergraph_gsql",
            "graph_context_found": False,
            "status":              "no_context",
        })
        return result

    _load()
    # Same instruction shape and completion cap (512) as pipeline1/pipeline2 -- only the
    # context source differs, so token comparisons reflect retrieval, not prompt tricks.
    prompt = (
        "Answer the question in clear prose using only the context below. "
        "Include all relevant facts, names, and dates.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    answer  = gemini_generate(_gemini_client, prompt, max_tokens=512)
    latency = round(time.time() - t_start, 3)

    result = make_result("graphrag", answer, count_tokens(prompt), count_tokens(answer), latency)
    result.update({
        "sources":             list(seen)[:5],
        "context_tokens":      count_tokens(context),
        "retriever":           "tigergraph_gsql",
        "graph_context_found": True,
        "status":              "ok",
    })
    return result
