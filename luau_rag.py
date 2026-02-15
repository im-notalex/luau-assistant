# luau_rag.py
import os
import glob
import subprocess
import tempfile
import shutil
import sys
from sentence_transformers import SentenceTransformer
import chromadb
from api_validator import validate_luau_code_blocks
import requests
try:
    from urllib3.util.retry import Retry  # noqa: F401
    from requests.adapters import HTTPAdapter  # noqa: F401
except Exception:
    Retry = None  # noqa: F401
    HTTPAdapter = None  # noqa: F401

# Configure subprocess encoding
def configure_subprocess():
    """Configure subprocess for proper encoding on Windows."""
    if sys.platform == "win32":
        # Set default subprocess parameters
        subprocess.Popen.default_params = {
            'encoding': 'utf-8',
            'errors': 'replace',
            'env': {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        }
        
        # For Windows, also configure subprocess window
        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen.default_params['startupinfo'] = startupinfo

configure_subprocess()

ROOT = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(ROOT, "chroma")
DOCS_PATH = os.path.join(ROOT, "Docs", "markdown")
MODEL_PATH = os.path.join(ROOT, "models", "deepseek-coder-7b-instruct-v1.5-Q6_K.gguf")
# Path to llama.cpp binary. If main.exe is in repo root, leave as "main.exe"
LLAMA_CPP_BIN = os.environ.get("LLAMA_CPP_BIN", "main.exe")  # Using system PATH by default
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mofanke/deepseek-coder:7b-instruct-v1.5-Q4_K_M")
FORCE_OLLAMA = os.environ.get("FORCE_OLLAMA", "").lower() in ("1", "true", "yes")
PREFERRED_BACKEND = os.environ.get("PREFERRED_BACKEND", "").lower()  # 'local'|'ollama'|'openai'|'openrouter'|'openai_compat'|'gemini'|'anthropic'|'xai'
# OpenAI configuration (only used if explicitly selected or preferred)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # UI labels it as "gpt5"; override here if desired
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "")
OPENAI_COMPAT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "")
OPENAI_COMPAT_AUTH_HEADER = os.environ.get("OPENAI_COMPAT_AUTH_HEADER", "Authorization")
OPENAI_COMPAT_AUTH_SCHEME = os.environ.get("OPENAI_COMPAT_AUTH_SCHEME", "Bearer")
OPENAI_COMPAT_VERIFY_SSL = os.environ.get("OPENAI_COMPAT_VERIFY_SSL", "1").lower() not in ("0", "false", "no")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2-latest")
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 6
DESIRED_CONTEXT_TOKENS = 3_000_000  # target tokens user wants (approx)
CHUNK_PROMPT_SIZE = 12_000_000  # coarse char approximation (~4 chars/token)
DEFAULT_TEMP = 0.5  # Default temperature for generation
APPROX_CHARS_PER_TOKEN = 4
PER_BACKEND_TOKEN_CAP = {
    'openai': 204_000,
    'openrouter': 204_000,
    'openai_compat': 204_000,
    'gemini': 204_000,
    'anthropic': 204_000,
    'xai': 204_000,
    'ollama': 12_000_000,
    'llama.cpp': 12_000_000,
}

def clamp_prompt_for_backend(prompt_text: str, backend: str, model: str = None) -> str:
    try:
        b = (backend or '').lower()
        cap_tokens = PER_BACKEND_TOKEN_CAP.get(b, 500_000)
        if not model:
            if b == 'openrouter':
                model = os.environ.get('OPENROUTER_MODEL')
            elif b == 'openai':
                model = os.environ.get('OPENAI_MODEL')
            elif b == 'openai_compat':
                model = os.environ.get('OPENAI_COMPAT_MODEL')
        m = (model or '').lower()
        if b in ('openai','openrouter','openai_compat') and 'deepseek' in m:
            cap_tokens = min(cap_tokens, 164_000)
        cap_chars = max(int(cap_tokens * APPROX_CHARS_PER_TOKEN), 1)
        if not isinstance(prompt_text, str):
            return prompt_text
        if len(prompt_text) <= cap_chars:
            return prompt_text
        return prompt_text[:cap_chars]
    except Exception:
        return prompt_text

def _make_session(verify: bool = True):
    """Return a plain requests session without automatic retries.
    Some providers (e.g., OpenRouter) aggressively rate-limit (429), and
    automatic retries can make it worse. The caller should handle 429s.
    """
    try:
        s = requests.Session()
        s.verify = verify
        return s
    except Exception:
        return requests

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection("luau_docs")
    except Exception:
        col = client.create_collection("luau_docs")
    return col

_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER

def get_context_and_sources(question, top_k=TOP_K, max_context_tokens=None):
    """Retrieve concatenated context text and a list of source identifiers."""
    embedder = get_embedder()
    q_emb = embedder.encode([question])[0]
    col = get_collection()
    docs = []
    metas = []
    # Retrieve more candidates for hybrid rerank (embedding+BM25+heuristics)
    n0 = max(int(top_k) * 3, 30)
    # Primary: embedding query
    try:
        res = col.query(query_embeddings=[q_emb], n_results=n0, include=["documents", "metadatas", "distances"])
        docs_e = res.get("documents", [[]])[0]
        metas_e = res.get("metadatas", [[]])[0]
        dists_e = res.get("distances", [[]])[0] if isinstance(res.get("distances", None), list) else [None]*len(docs_e)
    except Exception:
        docs_e, metas_e, dists_e = [], [], []
    # Secondary: text query (brings in lexical matches)
    try:
        res2 = col.query(query_texts=[question], n_results=n0, include=["documents", "metadatas"])
        docs_t = res2.get("documents", [[]])[0]
        metas_t = res2.get("metadatas", [[]])[0]
        dists_t = [None]*len(docs_t)
    except Exception:
        docs_t, metas_t, dists_t = [], [], []
    # Merge both candidate pools (dedupe by source+first 24 chars)
    def key_of(m, d):
        s = None
        if isinstance(m, dict):
            s = m.get("source")
        return (s or ""), (d or "")[:24]
    pool = []
    seen = set()
    for i in range(len(docs_e)):
        k = key_of(metas_e[i] if i < len(metas_e) else None, docs_e[i])
        if k in seen: continue
        seen.add(k)
        pool.append((docs_e[i], metas_e[i] if i < len(metas_e) else None, dists_e[i] if i < len(dists_e) else None))
    for i in range(len(docs_t)):
        k = key_of(metas_t[i] if i < len(metas_t) else None, docs_t[i])
        if k in seen: continue
        seen.add(k)
        pool.append((docs_t[i], metas_t[i] if i < len(metas_t) else None, dists_t[i] if i < len(dists_t) else None))
    docs = [p[0] for p in pool]
    metas = [p[1] for p in pool]
    dists = [p[2] for p in pool]

    # Lightweight rerank: prefer path relevance + keyword hits
    try:
        q = (question or "").lower()
        terms = [t for t in q.replace("/", " ").replace("_", " ").split() if len(t) > 2]
        # BM25 rerank on candidates (fallbacks gracefully if lib missing)
        bm25_scores = None
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [ (d or "").lower().split() for d in docs ]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(terms)
        except Exception:
            pass

        def score_item(doc_text, meta, dist, idx):
            s = 0.0
            lt = (doc_text or "").lower()
            for t in terms:
                if t in lt:
                    s += 2.0
            # Prefer Roblox engine terms strongly
            try:
                engine_terms = ['roblox', 'runservice', 'players', 'workspace', 'humanoid', 'server', 'client', 'profilingservice', 'replicatedstorage', 'remotes']
                if any(et in lt for et in engine_terms):
                    s += 3.0
            except Exception:
                pass
            src = None
            if isinstance(meta, dict):
                src = (meta.get("source") or "").lower()
            if src:
                if "reference/engine" in src:
                    s += 4.0
                if any(k in src for k in ['api', 'devhub', 'roblox']):
                    s += 3.0
                if src.endswith(".md"):
                    s += 1.0
            # embedding distance -> score (smaller distance better)
            if dist is not None:
                try:
                    s += max(0.0, 5.0 - float(dist))  # coarse normalization
                except Exception:
                    pass
            # BM25 score contribution
            if bm25_scores is not None:
                try:
                    # normalize bm25 by corpus mean
                    s += 0.3 * float(bm25_scores[idx])
                except Exception:
                    pass
            return s
        scored = [ (docs[i], metas[i], score_item(docs[i], metas[i], (dists[i] if i < len(dists) else None), i)) for i in range(len(docs)) ]
        scored.sort(key=lambda x: x[2], reverse=True)
        # trim to requested top_k
        if scored:
            docs = [x[0] for x in scored[:top_k]]
            metas = [x[1] for x in scored[:top_k]]
    except Exception:
        pass

    combined = "\n\n".join(docs)
    # trim to avoid extremely long prompts
    max_chars = CHUNK_PROMPT_SIZE
    effective_max_context_tokens = max_context_tokens
    if max_context_tokens and max_context_tokens < 10000:
        # For hosted backends, prevent context starvation; fallback to reasonable minimum
        effective_max_context_tokens = 50000  # 50k tokens ~200k chars
        print(f"Warning: max_context_tokens ({max_context_tokens}) too low for hosted backends; using {effective_max_context_tokens} to ensure context quality.")
    if effective_max_context_tokens and effective_max_context_tokens >= 10000:
        max_chars = min(max_chars, effective_max_context_tokens * APPROX_CHARS_PER_TOKEN)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n... (truncated) ..."

    # Collect readable source labels (deduped, preserve order)
    sources = []
    seen = set()
    for m in metas:
        src = None
        if isinstance(m, dict):
            src = m.get("source")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    return combined, sources

def retrieve_context(question, top_k=TOP_K):
    """Backward-compatible helper returning only the context text."""
    text, _ = get_context_and_sources(question, top_k=top_k)
    return text


def lint_luau_snippets(answer_text: str):
    """Lightweight lint to catch common API mistakes in generated LuaU code.
    Returns a list of notes with suggested fixes. Does nothing if no issues found.
    """
    try:
        import re
        notes = []
        # Find ```lua code blocks
        blocks = re.findall(r"```lua\s*(.*?)```", answer_text, flags=re.DOTALL | re.IGNORECASE)
        for b in blocks:
            # Humanoid:MoveDirection() is invalid; MoveDirection is a property or use Humanoid:Move()
            if re.search(r"Humanoid\s*:\s*MoveDirection\s*\(", b, flags=re.IGNORECASE):
                notes.append("Replace Humanoid:MoveDirection() (invalid) with Humanoid:MoveDirection property access or use Humanoid:Move(Vector3, relative) for movement.")
            # ParticleEmitter property types and usage
            if re.search(r"Instance\.new\(\"ParticleEmitter\"\)", b):
                if re.search(r"Acceleration\s*=\s*Vector2", b):
                    notes.append("ParticleEmitter.Acceleration expects Vector3, not Vector2.")
                if re.search(r"VelocitySpread\s*=\s*Vector\w+", b):
                    notes.append("ParticleEmitter.VelocitySpread expects a number (degrees).")
                if not re.search(r"Attachment", b):
                    notes.append("Create an Attachment inside the part and parent the ParticleEmitter to it for consistent positioning.")
                if ('rainbow' in answer_text.lower()) and re.search(r"ColorSequence\.new\s*\(\s*Color3", b) and not re.search(r"ColorSequenceKeypoint", b):
                    notes.append("For a true rainbow, use multiple ColorSequenceKeypoint entries (red, orange, yellow, green, blue, magenta) rather than a 2-color gradient.")
        # Cross-check with API dump if available
        try:
            root = ROOT
            api_notes = validate_luau_code_blocks(answer_text, root)
            notes.extend(api_notes)
        except Exception:
            pass
        return notes
    except Exception:
        return []

    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
def sanitize_answer(text: str) -> str:
    """
    A SAFE sanitizer: only removes *actual* unwanted meta lines.
    It will NEVER wipe regular content or accidentally match words like "no", "not", "none".
    """

    import re

    if not text:
        return text

    raw = text  # keep a copy for debugging

    # --------------------------
    # SAFE removal rules (strict)
    # --------------------------

    lines = raw.splitlines()
    cleaned = []

    # Patterns that are safe to remove ONLY when they match *the entire line*
    REMOVE_FULL_LINE = [
        r"^\s*as an ai\b.*$",
        r"^\s*i (?:do not|don't) have access\b.*$",
        r"^\s*i cannot browse\b.*$",
        r"^\s*i have no browsing\b.*$",
        r"^\s*cannot browse the internet\b.*$",
        r"^\s*no internet access\b.*$",
        r"^\s*sorry\b.*$",
        r"^\s*apologies\b.*$",
        r"^\s*assistant\s*$",
    ]

    rm = [re.compile(p, flags=re.IGNORECASE) for p in REMOVE_FULL_LINE]

    for ln in lines:
        stripped = ln.strip()

        # If line is empty or whitespace, just keep it
        if not stripped:
            cleaned.append(ln)
            continue

        # Only remove lines that match EXACT harmful meta sentences
        drop = any(rx.match(stripped) for rx in rm)
        if drop:
            continue

        cleaned.append(ln)

    out = "\n".join(cleaned)

    # Fix Luau/Lua formatting
    out = re.sub(r"```\s*luau", "```lua", out, flags=re.IGNORECASE)

    # Balance backticks (but never delete content)
    fence_count = len(re.findall(r"```", out))
    if fence_count % 2 == 1:
        out += "\n```"

    # Remove long external URLs (optional, but safe)
    out = re.sub(r"https?://[^\s)]+", "", out)

    return out.strip()

def select_backend():
    """Choose backend using environment preferences and availability."""
    if PREFERRED_BACKEND == 'local':
        return 'llama.cpp'
    if PREFERRED_BACKEND == 'ollama':
        return 'ollama'
    if PREFERRED_BACKEND == 'openai' and (OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY')):
        return 'openai'
    if PREFERRED_BACKEND == 'openrouter' and (OPENROUTER_API_KEY or os.environ.get('OPENROUTER_API_KEY')):
        return 'openrouter'
    if PREFERRED_BACKEND == 'openai_compat' and (OPENAI_COMPAT_API_KEY or os.environ.get('OPENAI_COMPAT_API_KEY')) and (OPENAI_COMPAT_BASE_URL or os.environ.get('OPENAI_COMPAT_BASE_URL')):
        return 'openai_compat'
    if PREFERRED_BACKEND == 'gemini' and (GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY')):
        return 'gemini'
    if PREFERRED_BACKEND == 'anthropic' and (ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY')):
        return 'anthropic'
    if PREFERRED_BACKEND == 'xai' and (XAI_API_KEY or os.environ.get('XAI_API_KEY')):
        return 'xai'
    if os.path.exists(MODEL_PATH) and (os.path.exists(LLAMA_CPP_BIN) or shutil.which(LLAMA_CPP_BIN)):
        return 'llama.cpp'
    if shutil.which('ollama'):
        return 'ollama'
    if OPENAI_API_KEY:
        return 'openai'
    if OPENROUTER_API_KEY:
        return 'openrouter'
    if OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL:
        return 'openai_compat'
    if GEMINI_API_KEY:
        return 'gemini'
    if ANTHROPIC_API_KEY:
        return 'anthropic'
    if XAI_API_KEY:
        return 'xai'
    return 'llama.cpp'


def build_guidelines() -> str:
    return """You are a careful reading companion. Explain and interpret the supplied documents with empathy and clarity.

Core principles:
- Prioritize the specific passages provided in Context; do not invent events or dialogue.
- When the user mentions a page (e.g., page 136), focus on that page range before anything else.
- Reference document names and page numbers directly in prose (e.g., 'Page 136 of Novel.md highlights...').
- If a passage is missing from Context, state that plainly and suggest how to locate it.
- Keep the tone approachable, as if tutoring someone who just asked for help understanding the reading.

Formatting:
- Respond in Markdown with sections such as Summary, Key Details, and References when helpful.
- References must cite internal paths and page numbers supplied in Context (e.g., `Docs/book.md (page 136)`).
- Avoid introductions like 'As an AI' or apologies unless information is genuinely unavailable.
- Use bullet lists for supporting details when it improves readability.
"""


# Reading reminders injected into the prompt
TRAINING_SNIPPETS = """Reading Assistant Reminders:
- Cite document names and page numbers from the Context.
- Summaries should stay faithful to the provided excerpts.
- If key details are missing, say so and explain what is needed.
"""

def _load_training_snippets():
    return

_load_training_snippets()

# Preload ALL markdown under Docs/ and build a compact catalog to keep in prompt
DOCS_CATALOG = ''
DOCS_COUNT = 0
DOCS_CATALOG_CAP = 100000  # characters to include in prompt for catalog

def _load_docs_catalog():
    global DOCS_CATALOG, DOCS_COUNT
    try:
        import os, re
        base = os.path.join(ROOT, 'Docs')
        lines = []
        count = 0
        for root, _, files in os.walk(base):
            for f in files:
                if not f.lower().endswith('.md'):
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, ROOT).replace(os.sep, '/')
                try:
                    txt = open(p, 'r', encoding='utf-8', errors='ignore').read()
                except Exception:
                    continue
                count += 1
                # extract up to three headings or first sentence
                heads = re.findall(r'^#{1,3}\s+(.+)$', txt, flags=re.MULTILINE)
                summary = '; '.join(heads[:3]) if heads else ''
                if not summary:
                    m = re.search(r'\b[^\n]{10,160}[\.!?]', txt)
                    summary = m.group(0).strip() if m else ''
                line = f"- {rel}: {summary}".strip()
                lines.append(line)
                # stop if catalog too large
                if sum(len(x)+1 for x in lines) > DOCS_CATALOG_CAP:
                    break
            if sum(len(x)+1 for x in lines) > DOCS_CATALOG_CAP:
                break
        DOCS_CATALOG = '\n'.join(lines)
        DOCS_COUNT = count
    except Exception:
        DOCS_CATALOG = ''
        DOCS_COUNT = 0

_load_docs_catalog()

def prepare_prompt(question, plan_mode=False, plan_iters=2, strict_mode=False, retr_top_k: int = TOP_K, max_context_tokens: int = None, pinned_path: str = None, pinned_paths: list = None, debug_no_guidelines: bool = False, pins_only: bool = False):
    """Retrieve context, optionally plan, and build the final prompt. Returns (prompt, sources, plan_steps)."""
    ctx = ''
    ctx_sources = []

    base = build_guidelines() if not debug_no_guidelines else ""

    # Engine anchor: if API dump is loaded, remind model this is Roblox engine work
    try:
        from api_validator import get_stats as _api_stats
        s = _api_stats()
        if s and s.get('loaded'):
            base += ("\nENGINE CONTEXT:\n"
                     "- This is Roblox LuaU in Roblox Studio/Engine, not generic Lua.\n"
                     "- Prefer RunService (Heartbeat/Stepped/RenderStepped), ProfilingService, task.wait, time(), and Roblox services via game:GetService.\n"
                     "- Use Instances, Humanoid, Players, Workspace appropriately.\n\n")
    except Exception:
        pass

    plan_steps = []
    if plan_mode:
        import json as _json
        def extract_steps(txt: str):
            try:
                obj = _json.loads(txt)
            except Exception:
                import re
                m = re.search(r"(\[.*\]|\{.*\})", txt, flags=re.DOTALL)
                if not m:
                    return []
                try:
                    obj = _json.loads(m.group(1))
                except Exception:
                    return []
            items = []
            if isinstance(obj, dict) and isinstance(obj.get('steps'), list):
                items = obj['steps']
            elif isinstance(obj, list):
                items = obj
            steps = []
            for it in items:
                if isinstance(it, dict) and 'step' in it:
                    steps.append(str(it['step']).strip())
                elif isinstance(it, str):
                    steps.append(it.strip())
            return [s for s in steps if s]

        planning_prompt_base = f"You are planning how to answer the user's question using the provided Context.\nOutput ONLY JSON with a 'steps' array of 3-6 short strings. No explanations.\n\nContext:\n{ctx}\n\nQuestion:\n{question}\n"

        backend = select_backend()
        prev = None
        for i in range(max(1, int(plan_iters))):
            if i == 0:
                ptxt = planning_prompt_base
            else:
                ptxt = planning_prompt_base + f"\nRefine the steps given the previous plan: {prev}. Output JSON only.\n"
            if backend == 'llama.cpp':
                out_plan, _ = call_llama_cpp(ptxt)
            else:
                out_plan, _ = call_ollama(ptxt, temperature=0.2)
            prev = out_plan or prev
            steps = extract_steps(out_plan or "")
            # Fallback: extract leading bullet lines if JSON missing
            if not steps and out_plan:
                import re
                bullets = re.findall(r"^\s*[-*]\s+(.*)$", out_plan, flags=re.MULTILINE)
                steps = [b.strip() for b in bullets[:6] if b.strip()]
            if steps:
                plan_steps = steps

    # Load pinned documents if provided (up to 8)
    pinned_block = ''
    pinned_sources = []
    try:
        paths = []
        if isinstance(pinned_paths, list):
            paths.extend([p for p in pinned_paths if isinstance(p, str)])
        if pinned_path and pinned_path not in paths:
            paths.append(pinned_path)
        paths = paths[:8]
        if paths:
            parts = []
            docs_root = os.path.abspath(os.path.join(ROOT, 'Docs'))
            for pp in paths:
                p = os.path.abspath(os.path.join(ROOT, pp.replace('\\', '/')))
                if p.startswith(docs_root) and p.lower().endswith('.md') and os.path.exists(p):
                    try:
                        txt = open(p, 'r', encoding='utf-8', errors='ignore').read()
                        relp = os.path.relpath(p, ROOT).replace('\\\\','/')
                        parts.append(f"Pinned Document: {relp}\n" + txt[:20000])
                        pinned_sources.append(relp)
                    except Exception:
                        continue
            pinned_block = "\n\n".join(parts)
    except Exception:
        pinned_block = ''
        pinned_sources = []

    # Retrieval: normal unless pins_only is enabled
    if pins_only and pinned_block:
        ctx = pinned_block
        ctx_sources = pinned_sources[:]
        # Do not perform retrieval when pins_only
    else:
        ctx, ctx_sources = get_context_and_sources(question, top_k=retr_top_k, max_context_tokens=max_context_tokens)

    # Nudge towards Roblox-correct benchmarking if query suggests it
    try:
        ql = (question or '').lower()
        bench_hint = ''
        if any(k in ql for k in ['benchmark', 'profil', 'perf', 'performance', 'microprofiler']):
            bench_hint = ("\nROBLOX BENCHMARK HINTS:\n"
                          "- Use RunService (Heartbeat/Stepped/RenderStepped) for frame-time context; avoid plain tight loops that block.\n"
                          "- Use os.clock()/time() for measurement, and consider ProfilingService or MicroProfiler.\n"
                          "- Benchmark realistic workloads (Instances, Humanoids, Services) rather than pure Lua math only.\n\n")
            base += bench_hint
    except Exception:
        pass

    prompt = base + (
        f"Using the following context, provide a structured response:\n\nContext:\n{ctx}\n\n"
        f"Training Data:\n{TRAINING_SNIPPETS}\n\n"
        f"Docs Catalog (headings):\n{DOCS_CATALOG}\n\n"
        + (f"{pinned_block}\n\n" if (pinned_block and not pins_only) else "") +
        f"Sources List:\n" + ("\n".join(f"- {s}" for s in ctx_sources) if ctx_sources else "(none)") + "\n\n"
        f"Question:\n{question}\n\n"
        "Remember:\n"
        "- Never put prose inside code fences.\n"
        "- Always end with a References section.\n"
        "- Format all code with ```lua blocks and proper indentation.\n"
        "- Start with minimal, runnable ```lua code before longer prose.\n"
        "- Add a top-of-code comment stating script type and placement (e.g., LocalScript in StarterPlayerScripts, ModuleScript in ServerScriptService).\n"
    )

    if plan_steps:
        prompt += ("\nINTERNAL PLAN (do not include in the final answer):\n" + "\n".join(f"- {s}" for s in plan_steps) +
                   "\nDo NOT output the plan. Produce only the final Overview/Code/Notes/References sections.\n")
    elif strict_mode:
        prompt += ("\nINTERNAL PLAN (do not include in the final answer):\n- Summarize the goal\n- Provide a short example\n- Cite relevant docs\n")

    # Strict mode disabled by design: extensive memory + catalog in context

    return prompt, ctx_sources, plan_steps

def call_llama_cpp(prompt, max_tokens=512):
    diagnostics = {
        'tool': 'llama.cpp',
        'cmd': None,
        'stdout': None,
        'stderr': None,
        'returncode': None,
        'error': None,
    }

    # Ensure model exists
    if not os.path.exists(MODEL_PATH):
        diagnostics['error'] = f"Model not found at {MODEL_PATH}. Place the .gguf there."
        return (f"ERROR: Model not found at {MODEL_PATH}. Place the .gguf there.", diagnostics)

    # If a llama.cpp binary is available try to use it
    if os.path.exists(LLAMA_CPP_BIN) or shutil.which(LLAMA_CPP_BIN):
        # We write prompt to a temp file to avoid quoting issues for very long prompts
        prompt_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as tf:
                tf.write(prompt)
                tf.flush()
                prompt_file = tf.name

            cmd = [
                LLAMA_CPP_BIN,
                "-m", MODEL_PATH,
                "-p", open(prompt_file, "r", encoding="utf-8").read(),
                "-n", str(max_tokens)
            ]

            diagnostics['cmd'] = cmd

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=120,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
                )
                diagnostics['stdout'] = (proc.stdout or "").strip()
                diagnostics['stderr'] = (proc.stderr or "").strip()
                diagnostics['returncode'] = proc.returncode
                out = diagnostics['stdout']
            except Exception:
                # Fallback: try feeding prompt via stdin
                try:
                    diagnostics['cmd'] = [LLAMA_CPP_BIN, "-m", MODEL_PATH, "-n", str(max_tokens)]
                    proc = subprocess.run(
                        [LLAMA_CPP_BIN, "-m", MODEL_PATH, "-n", str(max_tokens)],
                        input=prompt,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=120,
                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
                    )
                    diagnostics['stdout'] = (proc.stdout or "").strip()
                    diagnostics['stderr'] = (proc.stderr or "").strip()
                    diagnostics['returncode'] = proc.returncode
                    out = diagnostics['stdout']
                except Exception as e2:
                    diagnostics['error'] = str(e2)
                    out = f"Failed to run llama.cpp: {e2}\nEnsure llama.cpp binary supports -p or stdin usage and that LLAMA_CPP_BIN path is correct."

        finally:
            # cleanup temp file
            try:
                if prompt_file and os.path.exists(prompt_file):
                    os.remove(prompt_file)
            except Exception:
                pass
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
        return (out, diagnostics)

    # If llama.cpp not available, return instruction to use Ollama or let caller handle fallback
    diagnostics['error'] = f"llama.cpp binary not found at {LLAMA_CPP_BIN}."
    return (f"llama.cpp binary not found at {LLAMA_CPP_BIN}.\nYou can set LLAMA_CPP_BIN environment variable to the full path of your llama.cpp main executable, or install/build llama.cpp. Alternatively, ensure 'ollama' is installed and available - the system can fallback to Ollama if configured.", diagnostics)


def ansi_strip(s):
    import re
    ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_re.sub('', s)

    def check_model_availability():
        """Check if the model is available in Ollama."""
        try:
            check_proc = subprocess.run(
                [ollama_path, "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            # also log for diagnostics
            stdout = (check_proc.stdout or "").strip()
            stderr = (check_proc.stderr or "").strip()
            # write a small diagnostic file
            try:
                with open(os.path.join(ROOT, "logs", "ollama_list_debug.txt"), "w", encoding="utf-8") as df:
                    df.write("OLLAMA LIST STDOUT:\n")
                    df.write(stdout + "\n\n")
                    df.write("OLLAMA LIST STDERR:\n")
                    df.write(stderr + "\n")
            except Exception:
                pass
            return model_name.lower() in stdout.lower()
        except Exception:
            return False

    # Validate model availability
    if not check_model_availability():
        # Attempt to pull the model automatically (useful for UI startup)
        try:
            pull_proc = subprocess.run(
                [ollama_path, "pull", model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            # If pull succeeded (returncode 0) re-check availability
            if pull_proc.returncode == 0:
                if check_model_availability():
                    # proceed
                    pass
                else:
                    diagnostics['error'] = f"Model '{model_name}' not found after pulling."
                    return (f"Model '{model_name}' not found after pulling. Please run: ollama pull {model_name}", diagnostics)
            else:
                # Pull failed; surface stderr to user
                stderr = (pull_proc.stderr or "").strip()
                diagnostics['stderr'] = stderr
                diagnostics['returncode'] = pull_proc.returncode
                diagnostics['error'] = f"Failed to pull model '{model_name}': {stderr}"
                return (f"Failed to pull model '{model_name}': {stderr}\nPlease run: ollama pull {model_name}", diagnostics)
        except subprocess.TimeoutExpired:
            diagnostics['error'] = 'timeout'
            return (f"Model pull timed out for '{model_name}'. Please run: ollama pull {model_name}", diagnostics)
        except Exception as e:
            diagnostics['error'] = str(e)
            return (f"Failed to pull model '{model_name}': {e}\nPlease run: ollama pull {model_name}", diagnostics)

    try:
        # Try running with --temperature flag first (newer Ollama versions)
        cmd = [ollama_path, "run", model_name, "--temperature", str(temperature)]
        if isinstance(gen_options, dict):
            if gen_options.get("top_k") is not None:
                cmd += ["--top-k", str(gen_options.get("top_k"))]
            if gen_options.get("top_p") is not None:
                cmd += ["--top-p", str(gen_options.get("top_p"))]
            if gen_options.get("repeat_penalty") is not None:
                cmd += ["--repeat-penalty", str(gen_options.get("repeat_penalty"))]
            if gen_options.get("frequency_penalty") is not None:
                cmd += ["--frequency-penalty", str(gen_options.get("frequency_penalty"))]
        diagnostics['cmd'] = cmd

        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        # Process the output
        diagnostics['stdout'] = (proc.stdout or "").strip()
        diagnostics['stderr'] = (proc.stderr or "").strip()
        diagnostics['returncode'] = proc.returncode
        out = diagnostics['stdout']
        err = diagnostics['stderr']

        # If there's an error, write diagnostics to file for debugging
        if proc.returncode != 0 or err:
            try:
                with open(os.path.join(ROOT, "logs", "ollama_run_debug.txt"), "a", encoding="utf-8") as df:
                    df.write(f"CMD: {cmd}\nRETURN: {proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}\n----\n")
            except Exception:
                pass

        # If the CLI complains about an unknown flag (older Ollama), retry without the flag
        if err and "unknown flag" in err.lower():
            # Retry without temperature flag (older syntax)
            cmd2 = [ollama_path, "run", model_name]
            proc2 = subprocess.run(
                cmd2,
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            out2 = (proc2.stdout or "").strip()
            err2 = (proc2.stderr or "").strip()
            if proc2.returncode != 0 or err2:
                try:
                    with open(os.path.join(ROOT, "logs", "ollama_run_debug.txt"), "a", encoding="utf-8") as df:
                        df.write(f"CMD: {cmd2}\nRETURN: {proc2.returncode}\nSTDOUT:\n{out2}\nSTDERR:\n{err2}\n----\n")
                except Exception:
                    pass
            if proc2.returncode == 0 and out2:
                diagnostics['cmd'] = cmd2
                diagnostics['stdout'] = out2
                diagnostics['stderr'] = err2
                diagnostics['returncode'] = proc2.returncode
                return (out2, diagnostics)
            elif err2:
                diagnostics['error'] = err2
                diagnostics['cmd'] = cmd2
                diagnostics['stdout'] = out2
                diagnostics['stderr'] = err2
                diagnostics['returncode'] = proc2.returncode
                return (f"Ollama error: {err2}", diagnostics)
            else:
                diagnostics['error'] = 'Ollama ran but produced no output.'
                return ("Ollama ran but produced no output. Please try again.", diagnostics)

        # Handle different cases for the first attempt
        if proc.returncode == 0 and out:
            return (out, diagnostics)
        elif err:
            diagnostics['error'] = err
            return (f"Ollama error: {err}", diagnostics)
        else:
            diagnostics['error'] = 'Ollama ran but produced no output.'
            return ("Ollama ran but produced no output. Please try again.", diagnostics)

    except subprocess.TimeoutExpired:
        diagnostics['error'] = 'timeout'
        return ("Ollama request timed out after 120 seconds. Please try again.", diagnostics)
    except Exception as e:
        diagnostics['error'] = str(e)
        return (f"Failed to run Ollama: {str(e)}\nEnsure Ollama service is running: ollama serve", diagnostics)

def call_ollama_stream(prompt, model_name=None, temperature=0.7, gen_options=None):
    """Yield output from Ollama incrementally (best-effort)."""
    ollama_path = shutil.which('ollama')
    if not ollama_path:
        yield ""
        return
    if model_name is None:
        model_name = OLLAMA_MODEL
    cmd = [ollama_path, "run", model_name, "--temperature", str(temperature)]
    if isinstance(gen_options, dict):
        if gen_options.get("top_k") is not None:
            cmd += ["--top-k", str(gen_options.get("top_k"))]
        if gen_options.get("top_p") is not None:
            cmd += ["--top-p", str(gen_options.get("top_p"))]
        if gen_options.get("repeat_penalty") is not None:
            cmd += ["--repeat-penalty", str(gen_options.get("repeat_penalty"))]
        if gen_options.get("frequency_penalty") is not None:
            cmd += ["--frequency-penalty", str(gen_options.get("frequency_penalty"))]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except Exception:
            pass
        # Read line by line (Ollama usually outputs buffered lines)
        for line in proc.stdout:
            if not line:
                break
            yield line
    except Exception:
        yield ""


def call_ollama(prompt, model_name=None, temperature=0.7, gen_options=None):
    """Run Ollama non-streaming and return (text, diagnostics)."""
    diagnostics = {'tool': 'ollama'}
    ollama_path = shutil.which('ollama')
    if not ollama_path:
        diagnostics['error'] = 'ollama_not_found'
        return ("Ollama is not installed or not on PATH.", diagnostics)
    if model_name is None:
        model_name = OLLAMA_MODEL
    diagnostics['model'] = model_name
    cmd = [ollama_path, "run", model_name, "--temperature", str(temperature)]
    if isinstance(gen_options, dict):
        if gen_options.get("top_k") is not None:
            cmd += ["--top-k", str(gen_options.get("top_k"))]
        if gen_options.get("top_p") is not None:
            cmd += ["--top-p", str(gen_options.get("top_p"))]
        if gen_options.get("repeat_penalty") is not None:
            cmd += ["--repeat-penalty", str(gen_options.get("repeat_penalty"))]
        if gen_options.get("frequency_penalty") is not None:
            cmd += ["--frequency-penalty", str(gen_options.get("frequency_penalty"))]
    diagnostics['cmd'] = cmd
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        diagnostics['stdout'] = out
        diagnostics['stderr'] = err
        diagnostics['returncode'] = proc.returncode
        if proc.returncode == 0 and out:
            return (out, diagnostics)
        if err:
            diagnostics['error'] = err
            return (f"Ollama error: {err}", diagnostics)
        diagnostics['error'] = 'empty_output'
        return ("Ollama ran but produced no output.", diagnostics)
    except subprocess.TimeoutExpired:
        diagnostics['error'] = 'timeout'
        return ("Ollama request timed out.", diagnostics)
    except Exception as e:
        diagnostics['error'] = str(e)
        return (f"Failed to run Ollama: {e}", diagnostics)

def call_openai(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None, max_tokens: int = 800):
    """Call OpenAI Chat Completions API and return (text, diagnostics)."""
    diagnostics = {'tool': 'openai', 'endpoint': 'v1/chat/completions', 'model': model or OPENAI_MODEL}
    key = api_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY', '')
    mdl = model or OPENAI_MODEL
    if not key:
        return ("OpenAI API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    try:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {
            "model": mdl,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
                {"role": "user", "content": prompt}
            ]
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = resp.status_code
        if resp.status_code != 200:
            try:
                with open(os.path.join(ROOT, 'logs', 'openai_debug.txt'), 'a', encoding='utf-8') as df:
                    df.write(f"STATUS: {resp.status_code}\nBODY: {resp.text}\n---\n")
            except Exception:
                pass
            return (f"OpenAI error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt = _extract_text_from_chat_payload(data)
        return (txt or '', diagnostics)
    except requests.Timeout:
        return ("OpenAI request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"OpenAI error: {e}", {**diagnostics, 'error': str(e)})


def call_openai_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    """Yield chunks from OpenAI stream (best effort)."""
    key = api_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY', '')
    mdl = model or OPENAI_MODEL
    if not key:
        yield ""
        return
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": mdl,
        "temperature": float(temperature),
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        with requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[len('data: '):].strip()
                if line == '[DONE]':
                    break
                try:
                    import json as _json
                    obj = _json.loads(line)
                    delta = ((obj.get('choices') or [{}])[0].get('delta') or {}).get('content', '')
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception:
        yield ""
        return

def call_openrouter(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None, max_tokens: int = 800):
    """Call OpenRouter Chat Completions and return (text, diagnostics)."""
    diagnostics = {'tool': 'openrouter', 'endpoint': 'api/v1/chat/completions', 'model': model or OPENROUTER_MODEL}
    key = api_key or OPENROUTER_API_KEY or os.environ.get('OPENROUTER_API_KEY', '')
    mdl = model or OPENROUTER_MODEL
    if not key:
        return ("OpenRouter API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "User-Agent": "luau-rag/1.0",
        }
        body = {
            "model": mdl,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
                {"role": "user", "content": prompt}
            ]
        }
        verify_flag = (os.environ.get('OPENROUTER_VERIFY_SSL', '1').lower() not in ('0','false','no'))
        sess = _make_session(verify=verify_flag)
        resp = sess.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = resp.status_code
        if resp.status_code != 200:
            return (f"OpenRouter error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt = _extract_text_from_chat_payload(data)
        return (txt or '', diagnostics)
    except requests.Timeout:
        return ("OpenRouter request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"OpenRouter error: {e}", {**diagnostics, 'error': str(e)})

def call_openrouter_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    key = api_key or OPENROUTER_API_KEY or os.environ.get('OPENROUTER_API_KEY', '')
    mdl = model or OPENROUTER_MODEL
    if not key:
        yield ""
        return
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": "luau-rag/1.0",
    }
    body = {
        "model": mdl,
        "temperature": float(temperature),
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        verify_flag = (os.environ.get('OPENROUTER_VERIFY_SSL', '1').lower() not in ('0','false','no'))
        sess = _make_session(verify=verify_flag)
        with sess.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[len('data: '):].strip()
                if line == '[DONE]':
                    break
                try:
                    import json as _json
                    obj = _json.loads(line)
                    delta = ((obj.get('choices') or [{}])[0].get('delta') or {}).get('content', '')
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception:
        yield ""
        return

def call_openai_compat_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None, base_url: str = None):
    """Yield chunks from OpenAI-compatible streaming endpoint."""
    key = api_key or OPENAI_COMPAT_API_KEY or os.environ.get('OPENAI_COMPAT_API_KEY', '')
    mdl = model or OPENAI_COMPAT_MODEL
    endpoint = _build_openai_compat_endpoint(base_url or OPENAI_COMPAT_BASE_URL or os.environ.get('OPENAI_COMPAT_BASE_URL', ''))
    if not key or not endpoint:
        yield ""
        return
    headers = {"Content-Type": "application/json"}
    header_name = (OPENAI_COMPAT_AUTH_HEADER or "Authorization").strip()
    auth_scheme = (OPENAI_COMPAT_AUTH_SCHEME or "Bearer").strip()
    if header_name:
        if auth_scheme:
            headers[header_name] = f"{auth_scheme} {key}".strip()
        else:
            headers[header_name] = key
    body = {
        "model": mdl,
        "temperature": float(temperature),
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        sess = _make_session(verify=OPENAI_COMPAT_VERIFY_SSL)
        with sess.post(endpoint, json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[len('data: '):].strip()
                if line == '[DONE]':
                    break
                try:
                    import json as _json
                    obj = _json.loads(line)
                    delta = ((obj.get('choices') or [{}])[0].get('delta') or {}).get('content', '')
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception:
        yield ""
        return


def call_gemini_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    """Yield chunks from Gemini streamGenerateContent (best effort)."""
    key = api_key or GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY', '')
    mdl = model or GEMINI_MODEL
    base = (GEMINI_BASE_URL or os.environ.get('GEMINI_BASE_URL') or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if not key:
        yield ""
        return
    endpoint = f"{base}/models/{mdl}:streamGenerateContent"
    headers = {"Content-Type": "application/json"}
    body = {
        "system_instruction": {"parts": [{"text": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature)},
    }
    try:
        last_full = ""
        with requests.post(endpoint, params={"key": key, "alt": "sse"}, json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    import json as _json
                    obj = _json.loads(payload)
                except Exception:
                    continue
                full = ""
                for candidate in (obj.get("candidates") or []):
                    parts = ((candidate.get("content") or {}).get("parts") or [])
                    segs = []
                    for part in parts:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            segs.append(part["text"])
                    if segs:
                        full = "".join(segs)
                        break
                if not full:
                    continue
                if last_full and full.startswith(last_full):
                    delta = full[len(last_full):]
                else:
                    delta = full
                last_full = full
                if delta:
                    yield delta
    except Exception:
        yield ""
        return


def call_anthropic_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    """Yield chunks from Anthropic SSE stream (best effort)."""
    key = api_key or ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY', '')
    mdl = model or ANTHROPIC_MODEL
    if not key:
        yield ""
        return
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": mdl,
        "temperature": float(temperature),
        "max_tokens": 2048,
        "stream": True,
        "system": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly.",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        with requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    import json as _json
                    obj = _json.loads(payload)
                except Exception:
                    continue
                delta = ((obj.get("delta") or {}).get("text")) or ""
                if not delta and obj.get("type") == "content_block_delta":
                    delta = (((obj.get("delta") or {}).get("text")) or "")
                if delta:
                    yield delta
    except Exception:
        yield ""
        return


def call_xai_stream(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    """Yield chunks from xAI chat/completions stream (OpenAI-compatible SSE)."""
    key = api_key or XAI_API_KEY or os.environ.get('XAI_API_KEY', '')
    mdl = model or XAI_MODEL
    base = (XAI_BASE_URL or os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1").rstrip("/")
    if not key:
        yield ""
        return
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": mdl,
        "temperature": float(temperature),
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        with requests.post(f"{base}/chat/completions", json=body, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[len('data: '):].strip()
                if line == '[DONE]':
                    break
                try:
                    import json as _json
                    obj = _json.loads(line)
                    delta = ((obj.get('choices') or [{}])[0].get('delta') or {}).get('content', '')
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception:
        yield ""
        return

def _extract_text_from_chat_payload(data):
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "".join(parts).strip()
        if isinstance(content, str):
            return content
    output = data.get("output") or []
    if output and isinstance(output, list):
        text_parts = []
        for item in output:
            for c in (item or {}).get("content", []):
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    text_parts.append(c["text"])
        if text_parts:
            return "".join(text_parts).strip()
    return ""

def _build_openai_compat_endpoint(base_url: str) -> str:
    if not base_url:
        return ""
    base = base_url.strip()
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    return f"{base.rstrip('/')}/chat/completions"


def call_openai_compat(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None,
                       base_url: str = None, max_tokens: int = 800):
    """Call a generic OpenAI-compatible endpoint and return (text, diagnostics)."""
    diagnostics = {
        'tool': 'openai_compat',
        'model': model or OPENAI_COMPAT_MODEL,
        'base_url': base_url or OPENAI_COMPAT_BASE_URL,
    }
    key = api_key or OPENAI_COMPAT_API_KEY or os.environ.get('OPENAI_COMPAT_API_KEY', '')
    mdl = model or OPENAI_COMPAT_MODEL
    endpoint = _build_openai_compat_endpoint(base_url or OPENAI_COMPAT_BASE_URL or os.environ.get('OPENAI_COMPAT_BASE_URL', ''))
    if not key:
        return ("OpenAI-compatible API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    if not endpoint:
        return ("OpenAI-compatible base URL missing.", {**diagnostics, 'error': 'missing_base_url'})
    diagnostics['endpoint'] = endpoint
    try:
        headers = {"Content-Type": "application/json"}
        header_name = (OPENAI_COMPAT_AUTH_HEADER or "Authorization").strip()
        auth_scheme = (OPENAI_COMPAT_AUTH_SCHEME or "Bearer").strip()
        if header_name:
            if auth_scheme:
                headers[header_name] = f"{auth_scheme} {key}".strip()
            else:
                headers[header_name] = key
        body = {
            "model": mdl,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
                {"role": "user", "content": prompt}
            ]
        }
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        sess = _make_session(verify=OPENAI_COMPAT_VERIFY_SSL)
        resp = sess.post(endpoint, json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = getattr(resp, 'status_code', None)
        if getattr(resp, 'status_code', 200) != 200:
            return (f"OpenAI-compatible error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt = _extract_text_from_chat_payload(data)
        return (txt or '', diagnostics)
    except requests.Timeout:
        return ("OpenAI-compatible request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"OpenAI-compatible error: {e}", {**diagnostics, 'error': str(e)})


def call_gemini(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    diagnostics = {'tool': 'gemini', 'endpoint': 'models/*:generateContent', 'model': model or GEMINI_MODEL}
    key = api_key or GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY', '')
    mdl = model or GEMINI_MODEL
    base = (GEMINI_BASE_URL or os.environ.get('GEMINI_BASE_URL') or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if not key:
        return ("Gemini API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    endpoint = f"{base}/models/{mdl}:generateContent"
    diagnostics['endpoint'] = endpoint
    try:
        headers = {"Content-Type": "application/json"}
        body = {
            "system_instruction": {"parts": [{"text": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature)},
        }
        resp = requests.post(endpoint, params={"key": key}, json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = resp.status_code
        if resp.status_code != 200:
            return (f"Gemini error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt = ""
        for candidate in (data.get("candidates") or []):
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            chunks = []
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            if chunks:
                txt = "".join(chunks).strip()
                break
        if not txt:
            for key_name in ("output_text", "text"):
                value = data.get(key_name)
                if isinstance(value, str) and value.strip():
                    txt = value.strip()
                    break
        if not txt:
            fr = ((data.get("candidates") or [{}])[0].get("finishReason") if data.get("candidates") else None)
            diagnostics["finish_reason"] = fr
            return ("Gemini returned no text content.", {**diagnostics, 'error': 'empty_content'})
        return (txt, diagnostics)
    except requests.Timeout:
        return ("Gemini request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"Gemini error: {e}", {**diagnostics, 'error': str(e)})


def call_anthropic(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    diagnostics = {'tool': 'anthropic', 'endpoint': '/v1/messages', 'model': model or ANTHROPIC_MODEL}
    key = api_key or ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY', '')
    mdl = model or ANTHROPIC_MODEL
    if not key:
        return ("Anthropic API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    try:
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": mdl,
            "temperature": float(temperature),
            "max_tokens": 2048,
            "system": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = resp.status_code
        if resp.status_code != 200:
            return (f"Anthropic error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt_parts = []
        for part in (data.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                txt_parts.append(part["text"])
        txt = "".join(txt_parts).strip()
        if not txt:
            return ("Anthropic returned no text content.", {**diagnostics, 'error': 'empty_content'})
        return (txt, diagnostics)
    except requests.Timeout:
        return ("Anthropic request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"Anthropic error: {e}", {**diagnostics, 'error': str(e)})


def call_xai(prompt: str, temperature: float = 0.7, api_key: str = None, model: str = None):
    diagnostics = {'tool': 'xai', 'endpoint': '/chat/completions', 'model': model or XAI_MODEL}
    key = api_key or XAI_API_KEY or os.environ.get('XAI_API_KEY', '')
    mdl = model or XAI_MODEL
    base = (XAI_BASE_URL or os.environ.get('XAI_BASE_URL') or "https://api.x.ai/v1").rstrip("/")
    if not key:
        return ("xAI API key missing.", {**diagnostics, 'error': 'missing_api_key'})
    endpoint = f"{base}/chat/completions"
    diagnostics['endpoint'] = endpoint
    try:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {
            "model": mdl,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": "You are a precise LuaU/Roblox engineer. Follow the user's prompt strictly."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(endpoint, json=body, headers=headers, timeout=120)
        diagnostics['status_code'] = resp.status_code
        if resp.status_code != 200:
            return (f"xAI error: {resp.status_code} {resp.text}", {**diagnostics, 'error': resp.text})
        data = resp.json()
        txt = _extract_text_from_chat_payload(data)
        if not txt:
            return ("xAI returned no text content.", {**diagnostics, 'error': 'empty_content'})
        return (txt, diagnostics)
    except requests.Timeout:
        return ("xAI request timed out.", {**diagnostics, 'error': 'timeout'})
    except Exception as e:
        return (f"xAI error: {e}", {**diagnostics, 'error': str(e)})


def ask(
    question,
    temperature=0.7,
    plan_mode=False,
    review_mode=False,
    plan_iters=2,
    strict_mode=False,
    gen_options=None,
    retr_top_k: int = TOP_K,
    max_context_tokens: int = None,
    pinned_path: str = None,
    pinned_paths: list = None,
    pins_only: bool = False,
    backend_override: str = None,
    openai_key: str = None,
    openrouter_key: str = None,
    openrouter_model: str = None,
    openai_model: str = None,
    openai_compat_key: str = None,
    openai_compat_model: str = None,
    openai_compat_base_url: str = None,
    gemini_key: str = None,
    gemini_model: str = None,
    anthropic_key: str = None,
    anthropic_model: str = None,
    xai_key: str = None,
    xai_model: str = None,
):
    """Generate a response using either llama.cpp or Ollama.
    plan_mode: include a concise plan section at the top (3-5 bullets).
    review_mode: append a short quality check section based on the context.
    """
    # Get relevant documentation context and sources (or pins-only)
    ctx = ''
    ctx_sources = []
    
    # Construct the prompt and pinned document blocks
    # Load pinned documents if provided (up to 8)
    pinned_block = ''
    pinned_sources = []
    try:
        paths = []
        if isinstance(pinned_paths, list):
            paths.extend([p for p in pinned_paths if isinstance(p, str)])
        if pinned_path and pinned_path not in paths:
            paths.append(pinned_path)
        paths = paths[:8]
        if paths:
            parts = []
            docs_root = os.path.abspath(os.path.join(ROOT, 'Docs'))
            for pp in paths:
                p = os.path.abspath(os.path.join(ROOT, pp.replace('\\', '/')))
                if p.startswith(docs_root) and p.lower().endswith('.md') and os.path.exists(p):
                    try:
                        txt = open(p, 'r', encoding='utf-8', errors='ignore').read()
                        relp = os.path.relpath(p, ROOT).replace('\\\\','/')
                        parts.append(f"Pinned Document: {relp}\n" + txt[:20000])
                        pinned_sources.append(relp)
                    except Exception:
                        continue
            pinned_block = "\n\n".join(parts)
    except Exception:
        pinned_block = ''
        pinned_sources = []

    if pins_only and pinned_block:
        # Use only pinned content as context
        ctx = pinned_block
        ctx_sources = pinned_sources[:]
    else:
        ctx, ctx_sources = get_context_and_sources(question, top_k=retr_top_k, max_context_tokens=max_context_tokens)

    prompt = build_guidelines() + f"""

Using the following context, provide a structured response:

Context:
{ctx}

Training Data:
{TRAINING_SNIPPETS}

Docs Catalog (headings):
{DOCS_CATALOG}

Sources List:
""" + ("\n".join(f"- {s}" for s in ctx_sources) if ctx_sources else "(none)") + f"""

""" + (f"{pinned_block}\n" if (pinned_block and not pins_only) else "") + f"""

Question:
{question}

Remember:
- Never put prose inside code fences.
- Always end with a References section.
- Format all code with ```lua blocks and proper indentation.
- If asked to create or implement something, start with minimal ```lua code blocks before longer prose.
"""

    # placeholder; filled after backend selection
    plan_steps = []
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
    # Strict mode: tighten behavior
    if strict_mode:
        prompt += ("\nSTRICT MODE:\n- Keep the response concise and strictly relevant.\n"
                   "- If Context support is weak or missing, state that and stop.\n"
                   "- Always include References using provided sources.\n")
    

    def has_openai_credentials():
        return bool(openai_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY'))

    def has_openrouter_credentials():
        return bool(openrouter_key or OPENROUTER_API_KEY or os.environ.get('OPENROUTER_API_KEY'))

    def has_openai_compat_credentials():
        key_present = (openai_compat_key or OPENAI_COMPAT_API_KEY or os.environ.get('OPENAI_COMPAT_API_KEY'))
        model_present = (openai_compat_model or OPENAI_COMPAT_MODEL or os.environ.get('OPENAI_COMPAT_MODEL'))
        url_present = (openai_compat_base_url or OPENAI_COMPAT_BASE_URL or os.environ.get('OPENAI_COMPAT_BASE_URL'))
        return bool(key_present and model_present and url_present)

    def has_gemini_credentials():
        return bool(gemini_key or GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY'))

    def has_anthropic_credentials():
        return bool(anthropic_key or ANTHROPIC_API_KEY or os.environ.get('ANTHROPIC_API_KEY'))

    def has_xai_credentials():
        return bool(xai_key or XAI_API_KEY or os.environ.get('XAI_API_KEY'))

    def choose_backend():
        """Choose backend: prefer PREFERRED_BACKEND/env overrides, then local gguf (llama.cpp), then Ollama, then hosted keys."""
        # explicit UI override
        bo = (backend_override or '').lower()
        if bo == 'openai':
            return 'openai'
        if bo == 'openrouter':
            return 'openrouter'
        if bo in ('openai_compat', 'custom'):
            return 'openai_compat'
        if bo == 'gemini':
            return 'gemini'
        if bo == 'anthropic':
            return 'anthropic'
        if bo == 'xai':
            return 'xai'
        if bo == 'ollama':
            return 'ollama'
        if bo == 'local':
            return 'llama.cpp'

        # env preference
        if PREFERRED_BACKEND == 'local':
            return 'llama.cpp'
        if PREFERRED_BACKEND == 'ollama':
            return 'ollama'
        if PREFERRED_BACKEND == 'openai' and has_openai_credentials():
            return 'openai'
        if PREFERRED_BACKEND == 'openrouter' and has_openrouter_credentials():
            return 'openrouter'
        if PREFERRED_BACKEND == 'openai_compat' and has_openai_compat_credentials():
            return 'openai_compat'
        if PREFERRED_BACKEND == 'gemini' and has_gemini_credentials():
            return 'gemini'
        if PREFERRED_BACKEND == 'anthropic' and has_anthropic_credentials():
            return 'anthropic'
        if PREFERRED_BACKEND == 'xai' and has_xai_credentials():
            return 'xai'

        # prefer local if model file exists and llama.cpp binary seems available
        if os.path.exists(MODEL_PATH) and (os.path.exists(LLAMA_CPP_BIN) or shutil.which(LLAMA_CPP_BIN)):
            return 'llama.cpp'

        # otherwise prefer Ollama if available
        if shutil.which('ollama'):
            return 'ollama'

        # fallback prefer hosted if keys present, else llama.cpp
        if has_openai_credentials():
            return 'openai'
        if has_openrouter_credentials():
            return 'openrouter'
        if has_openai_compat_credentials():
            return 'openai_compat'
        if has_gemini_credentials():
            return 'gemini'
        if has_anthropic_credentials():
            return 'anthropic'
        if has_xai_credentials():
            return 'xai'
        return 'llama.cpp'

    backend = choose_backend()

    # Internal planning loop (not included in final answer). Run using selected backend.
    if plan_mode:
        import json as _json

        def extract_steps(txt):
            try:
                obj = _json.loads(txt)
            except Exception:
                import re
                m = re.search(r"(\[.*\]|\{.*\})", txt, flags=re.DOTALL)
                if not m:
                    return []
                try:
                    obj = _json.loads(m.group(1))
                except Exception:
                    return []
            steps = []
            if isinstance(obj, dict) and 'steps' in obj and isinstance(obj['steps'], list):
                items = obj['steps']
            elif isinstance(obj, list):
                items = obj
            else:
                items = []
            for it in items:
                if isinstance(it, dict) and 'step' in it:
                    steps.append(str(it['step']).strip())
                elif isinstance(it, str):
                    steps.append(it.strip())
            return [s for s in steps if s]

        planning_prompt_base = f"""You are planning how to answer the user's question using the provided Context.
Output ONLY JSON with a 'steps' array of 3-6 short strings. No explanations.

Context:\n{ctx}\n\nQuestion:\n{question}\n"""

        prev = None
        for i in range(max(1, int(plan_iters))):
            if i == 0:
                ptxt = planning_prompt_base
            else:
                ptxt = planning_prompt_base + f"\nRefine the steps given the previous plan: {prev}. Output JSON only.\n"
            if backend == 'llama.cpp':
                out_plan, _ = call_llama_cpp(ptxt)
            elif backend == 'ollama':
                out_plan, _ = call_ollama(ptxt, temperature=0.2, gen_options=gen_options)
            elif backend == 'openrouter':
                if (openrouter_model or OPENROUTER_MODEL or '').lower().find('qwen') != -1:
                    out_plan = ""
                else:
                    out_plan, _ = call_openrouter(ptxt, temperature=0.2, api_key=openrouter_key, model=openrouter_model)
            elif backend == 'openai_compat':
                out_plan, _ = call_openai_compat(
                    ptxt,
                    temperature=0.2,
                    api_key=openai_compat_key,
                    model=openai_compat_model,
                    base_url=openai_compat_base_url,
                )
            elif backend == 'gemini':
                out_plan, _ = call_gemini(ptxt, temperature=0.2, api_key=gemini_key, model=gemini_model)
            elif backend == 'anthropic':
                out_plan, _ = call_anthropic(ptxt, temperature=0.2, api_key=anthropic_key, model=anthropic_model)
            elif backend == 'xai':
                out_plan, _ = call_xai(ptxt, temperature=0.2, api_key=xai_key, model=xai_model)
            else:
                out_plan, _ = call_openai(ptxt, temperature=0.2, api_key=openai_key, model=(openai_model or None))
            prev = out_plan or prev
            steps = extract_steps(out_plan or "")
            if steps:
                plan_steps = steps

        if plan_steps:
            prompt += ("\nINTERNAL PLAN (do not include in the final answer):\n" + "\n".join(f"- {s}" for s in plan_steps) +
                       "\nDo NOT output the plan. Produce only the final Overview/Code/Notes/References sections.\n")
        else:
            # Fallback guard: add a minimal internal hint to still structure the answer
            prompt += ("\nINTERNAL PLAN (do not include in the final answer):\n- Summarize the goal\n- Provide a short example\n- Cite relevant docs\n")

    # Clamp prompt for backend capacity.
    model_hint = None
    if backend == 'openrouter':
        model_hint = openrouter_model or OPENROUTER_MODEL
    elif backend == 'openai':
        model_hint = openai_model or OPENAI_MODEL
    elif backend == 'openai_compat':
        model_hint = openai_compat_model or OPENAI_COMPAT_MODEL
    elif backend == 'gemini':
        model_hint = gemini_model or GEMINI_MODEL
    elif backend == 'anthropic':
        model_hint = anthropic_model or ANTHROPIC_MODEL
    elif backend == 'xai':
        model_hint = xai_model or XAI_MODEL
    prompt = clamp_prompt_for_backend(prompt, backend, model_hint)

    def ensure_references(out_text):
        text = out_text or ""
        try:
            if isinstance(ctx_sources, list) and ctx_sources and "references:" not in text.lower():
                refs = "\n".join(f"- {s}" for s in ctx_sources[:8])
                text = f"{text}\n\nReferences:\n{refs}"
        except Exception:
            pass
        return text

    def maybe_append_review(backend_name, draft):
        if not review_mode or not draft:
            return draft, None
        review_prompt = f"""You are a strict reviewer. In 3-5 short bullets, assess the following draft for:
- factual correctness vs Context
- missing/weak references
- security/performance caveats
If no issues, say 'Looks good'.

Context:\n{ctx}\n\nQuestion:\n{question}\n\nDraft Answer:\n{draft}\n"""
        try:
            if backend_name == 'llama.cpp':
                rev_out, _ = call_llama_cpp(review_prompt)
            elif backend_name == 'ollama':
                rev_out, _ = call_ollama(review_prompt, temperature=0.3)
            elif backend_name == 'openai':
                rev_out, _ = call_openai(review_prompt, temperature=0.3, api_key=openai_key, model=(openai_model or None))
            elif backend_name == 'openrouter':
                rev_out, _ = call_openrouter(review_prompt, temperature=0.3, api_key=openrouter_key, model=openrouter_model)
            elif backend_name == 'openai_compat':
                rev_out, _ = call_openai_compat(review_prompt, temperature=0.3, api_key=openai_compat_key, model=openai_compat_model, base_url=openai_compat_base_url)
            elif backend_name == 'gemini':
                rev_out, _ = call_gemini(review_prompt, temperature=0.3, api_key=gemini_key, model=gemini_model)
            elif backend_name == 'anthropic':
                rev_out, _ = call_anthropic(review_prompt, temperature=0.3, api_key=anthropic_key, model=anthropic_model)
            elif backend_name == 'xai':
                rev_out, _ = call_xai(review_prompt, temperature=0.3, api_key=xai_key, model=xai_model)
            else:
                rev_out = None
        except Exception:
            rev_out = None
        if rev_out and ('timed out' not in rev_out.lower()) and ('error' not in rev_out.lower()):
            return f"{draft}\n\nQuality Check:\n{rev_out.strip()}", rev_out.strip()
        return draft, None

    def maybe_lint(text):
        if not review_mode:
            return text
        try:
            lint_notes = lint_luau_snippets(text or "")
            if lint_notes:
                return f"{text}\n\nNotes:\n- " + "\n- ".join(lint_notes[:3])
        except Exception:
            pass
        return text

    if backend == 'llama.cpp':
        out, diag = call_llama_cpp(prompt)
        if (not out) or any(s in str(out) for s in ("ERROR: Model not found", "llama.cpp binary not found", "Failed to run llama.cpp:")):
            if shutil.which('ollama'):
                out2, diag2 = call_ollama(prompt, temperature=temperature, gen_options=gen_options)
                merged = {'preferred': 'ollama', 'llama': diag, 'ollama': diag2}
                out2, review = maybe_append_review('ollama', out2)
                if review:
                    merged['review'] = review
                out2 = ensure_references(maybe_lint(sanitize_answer(out2)))
                return ("ollama", out2, merged)
        out, review = maybe_append_review('llama.cpp', out)
        extra = {'llama': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("llama.cpp", out, extra)

    if backend == 'openai':
        out, diag = call_openai(prompt, temperature=temperature, api_key=openai_key, model=(openai_model or None))
        out, review = maybe_append_review('openai', out)
        extra = {'openai': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("openai", out, extra)

    if backend == 'openai_compat':
        out, diag = call_openai_compat(
            prompt,
            temperature=temperature,
            api_key=openai_compat_key,
            model=openai_compat_model,
            base_url=openai_compat_base_url,
        )
        out, review = maybe_append_review('openai_compat', out)
        extra = {'openai_compat': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("openai_compat", out, extra)

    if backend == 'openrouter':
        if (openrouter_model or OPENROUTER_MODEL or '').lower().find('qwen') != -1:
            out = "Selected OpenRouter Qwen model is disabled for now. Please choose another model."
            diag = {'error': 'qwen_blocked'}
        else:
            out, diag = call_openrouter(prompt, temperature=temperature, api_key=openrouter_key, model=openrouter_model)
        out, review = maybe_append_review('openrouter', out)
        extra = {'openrouter': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("openrouter", out, extra)

    if backend == 'gemini':
        out, diag = call_gemini(prompt, temperature=temperature, api_key=gemini_key, model=gemini_model)
        out, review = maybe_append_review('gemini', out)
        extra = {'gemini': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("gemini", out, extra)

    if backend == 'anthropic':
        out, diag = call_anthropic(prompt, temperature=temperature, api_key=anthropic_key, model=anthropic_model)
        out, review = maybe_append_review('anthropic', out)
        extra = {'anthropic': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("anthropic", out, extra)

    if backend == 'xai':
        out, diag = call_xai(prompt, temperature=temperature, api_key=xai_key, model=xai_model)
        out, review = maybe_append_review('xai', out)
        extra = {'xai': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("xai", out, extra)

    if backend == 'ollama':
        out, diag = call_ollama(prompt, temperature=temperature, gen_options=gen_options)
        if isinstance(out, str) and out.lower().startswith("model '") and os.path.exists(MODEL_PATH) and (os.path.exists(LLAMA_CPP_BIN) or shutil.which(LLAMA_CPP_BIN)):
            out2, diag2 = call_llama_cpp(prompt)
            merged = {'preferred': 'llama.cpp', 'ollama': diag, 'llama': diag2}
            out2, review = maybe_append_review('llama.cpp', out2)
            if review:
                merged['review'] = review
            out2 = ensure_references(maybe_lint(sanitize_answer(out2)))
            return ("llama.cpp", out2, merged)
        out, review = maybe_append_review('ollama', out)
        extra = {'ollama': diag, 'sources': ctx_sources}
        if review:
            extra['review'] = review
        if plan_steps:
            extra['plan_steps'] = plan_steps
        out = ensure_references(maybe_lint(sanitize_answer(out)))
        return ("ollama", out, extra)

    return (backend, sanitize_answer(out if 'out' in locals() else ""), {'sources': ctx_sources})

def main():
    print("LuaU RAG (local) - type 'exit' to quit.")
    print("Context length for Ollama models is 12000000 tokens, but external models are capped at 204000 tokens.")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "status":
            print(f"Model path: {MODEL_PATH}")
            llama_exists = os.path.exists(LLAMA_CPP_BIN) or shutil.which(LLAMA_CPP_BIN) is not None
            print(f"llama.cpp bin: {LLAMA_CPP_BIN} (exists: {llama_exists})")
            ollama_path = shutil.which("ollama")
            print(f"ollama: {ollama_path if ollama_path else 'not found on PATH'}")
            if ollama_path:
                try:
                    proc = subprocess.run([ollama_path, "list"], capture_output=True, text=True, timeout=10)
                    models_list = (proc.stdout or "").lower()
                    model_present = OLLAMA_MODEL.lower() in models_list
                    print(f"ollama model present: {model_present}")
                    if not model_present:
                        print(f"To pull the model, run: ollama pull {OLLAMA_MODEL}")
                except Exception as e:
                    print(f"Could not check ollama models: {e}")
                    print("Ensure the Ollama service is running")
            continue
        print("Retrieving docs and querying model...")
        result = ask(q)
        # support both old and new return shapes
        if isinstance(result, tuple) and len(result) == 3:
            backend, resp, diag = result
        elif isinstance(result, tuple) and len(result) == 2:
            backend, resp = result
            diag = None
        else:
            backend = 'unknown'
            resp = str(result)
            diag = None
        print(f"\n[backend: {backend}]\n---\n{resp}\n---\n")
        print(f"\n[backend: {backend}]\n---\n{resp}\n---\n")

if __name__ == "__main__":
    main()
