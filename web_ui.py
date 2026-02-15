"""
web_ui.py (HTTP UI)
Serves the modern HTML UI (FastAPI) and JSON endpoints; replaces the old Gradio UI.
Run: python web_ui.py and open the printed URL.
oh also, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
"""

import os
import json
import socket
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from json import dumps as json_dumps

from luau_rag import ask, OLLAMA_MODEL, prepare_prompt, call_ollama_stream, select_backend, CHUNK_PROMPT_SIZE, DESIRED_CONTEXT_TOKENS, clamp_prompt_for_backend, sanitize_answer
from api_validator import get_stats as api_stats, load_api_dump

ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
UI_DIR = ROOT / "ui"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)


def find_free_port(start_port: int = 7861, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise OSError("Could not find a free port")


def save_chat(messages: List[Dict[str, Any]], chat_id: Optional[str] = None) -> str:
    if not chat_id:
        chat_id = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
    data = {"id": chat_id, "timestamp": datetime.now().isoformat(), "messages": messages}
    with open(LOGS_DIR / f"{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return chat_id


def load_chat(chat_id: str) -> List[Dict[str, Any]]:
    p = LOGS_DIR / f"{chat_id}.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except Exception:
        return []

    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
def list_chats() -> List[Dict[str, Any]]:
    items = []
    for fp in LOGS_DIR.glob("chat_*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            items.append({
                "id": data.get("id", fp.stem),
                "timestamp": data.get("timestamp", datetime.fromtimestamp(fp.stat().st_mtime).isoformat()),
            })
        except Exception:
            # If JSON is corrupt, still surface the file using mtime
            try:
                items.append({
                    "id": fp.stem,
                    "timestamp": datetime.fromtimestamp(fp.stat().st_mtime).isoformat(),
                })
            except Exception:
                continue
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return items


def mount_static_ui():
    """Ensure the front-end assets are mounted so / serves index.html."""
    if getattr(app.state, "_ui_mounted", False):
        return
    if (UI_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
        app.state._ui_mounted = True


@asynccontextmanager
async def lifespan(_app):
    mount_static_ui()
    yield


app = FastAPI(title="LuaU RAG HTTP UI", lifespan=lifespan)

# Streaming cancellation flags (best-effort)
_cancel_flags: Dict[str, bool] = {}


@app.get("/api/status")
def api_status():
    # Try to load API dump status for UI display
    try:
        _ = load_api_dump(str(ROOT))
        dump = api_stats()
    except Exception:
        dump = {"loaded": False, "property_count": 0}
    # Docs preload stats (from luau_rag globals)
    try:
        from luau_rag import DOCS_COUNT, DOCS_CATALOG
        docs = {"count": int(DOCS_COUNT), "catalog_chars": len(DOCS_CATALOG or '')}
    except Exception:
        docs = {"count": 0, "catalog_chars": 0}
    return {"model": OLLAMA_MODEL, "context_cap": CHUNK_PROMPT_SIZE, "context_cap_tokens": DESIRED_CONTEXT_TOKENS, "api_dump": dump, "docs": docs}


@app.get("/api/backends")
def api_backends():
    supported = ["local", "ollama", "openai", "openrouter", "openai_compat", "gemini", "anthropic", "xai"]
    env_ready = {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "openai_compat": bool(os.environ.get("OPENAI_COMPAT_API_KEY") and os.environ.get("OPENAI_COMPAT_BASE_URL")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "xai": bool(os.environ.get("XAI_API_KEY")),
    }
    return {"supported": supported, "env_ready": env_ready}


@app.post("/api/ensure_api_dump")
def api_ensure_api_dump():
    try:
        from api_validator import ensure_api_dump as _ensure
        ensured = _ensure(str(ROOT))
        from api_validator import get_stats as _stats
        s = _stats()
        return {"ok": True, "api_dump": s}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats")
def api_list_chats():
    return {"chats": list_chats()}


@app.get("/api/chats/{chat_id}")
def api_get_chat(chat_id: str):
    return {"id": chat_id, "messages": load_chat(chat_id)}


@app.post("/api/new_chat")
def api_new_chat():
    cid = save_chat([])
    return {"id": cid, "messages": []}


@app.post("/api/chat")
def api_chat(
    message: str = Body(..., embed=True),
    chat_id: Optional[str] = Body(None),
    temperature: float = Body(0.3),
    plan_mode: bool = Body(False),
    self_check: bool = Body(False),
    strict_mode: bool = Body(False),
    plan_depth: int = Body(2),
    pinned_path: Optional[str] = Body(None),
    pinned_paths: Optional[List[str]] = Body(None),
    pins_only: Optional[bool] = Body(False),
    # backend selection
    backend: Optional[str] = Body(None),
    openai_key: Optional[str] = Body(None),
    openai_model: Optional[str] = Body(None),
    openrouter_key: Optional[str] = Body(None),
    openrouter_model: Optional[str] = Body(None),
    openai_compat_key: Optional[str] = Body(None),
    openai_compat_model: Optional[str] = Body(None),
    openai_compat_base_url: Optional[str] = Body(None),
    gemini_key: Optional[str] = Body(None),
    gemini_model: Optional[str] = Body(None),
    anthropic_key: Optional[str] = Body(None),
    anthropic_model: Optional[str] = Body(None),
    xai_key: Optional[str] = Body(None),
    xai_model: Optional[str] = Body(None),
    top_k: int = Body(None),
    top_p: float = Body(None),
    repeat_penalty: float = Body(None),
    freq_penalty: float = Body(None),
    retr_k: int = Body(None),
    max_context_tokens: int = Body(None),
 ):
    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    history = load_chat(chat_id) if chat_id else []

    gen_opts = {
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": freq_penalty,
    }

    pins_only_flag = bool(pins_only)
    chosen = (backend or '').lower().strip()
    hosted = chosen in ('openai', 'openrouter', 'openai_compat', 'gemini', 'anthropic', 'xai')
    eff_retr_k = (retr_k if retr_k else 5)
    if hosted and not pins_only_flag:
        eff_retr_k = min(eff_retr_k, 4)

    # Build prompt once to estimate tokens_in (approx) and to precompute sources if needed
    prompt, sources, plan_steps = prepare_prompt(
        message,
        plan_mode=plan_mode,
        plan_iters=plan_depth,
        strict_mode=strict_mode,
        retr_top_k=eff_retr_k,
        pinned_path=pinned_path,
        pinned_paths=pinned_paths,
        pins_only=pins_only_flag,
        max_context_tokens=max_context_tokens,
    )

    import time
    t0 = time.perf_counter()
    result = ask(
        message,
        temperature=temperature,
        plan_mode=plan_mode,
        review_mode=self_check,
        plan_iters=plan_depth,
        strict_mode=strict_mode,
        pinned_path=pinned_path,
        pinned_paths=pinned_paths,
        pins_only=pins_only_flag,
        gen_options=gen_opts,
        retr_top_k=eff_retr_k or None,
        max_context_tokens=max_context_tokens,
        backend_override=(backend or None),
        openai_key=(openai_key or None),
        openai_model=(openai_model or None),
        openrouter_key=(openrouter_key or None),
        openrouter_model=(openrouter_model or None),
        openai_compat_key=(openai_compat_key or None),
        openai_compat_model=(openai_compat_model or None),
        openai_compat_base_url=(openai_compat_base_url or None),
        gemini_key=(gemini_key or None),
        gemini_model=(gemini_model or None),
        anthropic_key=(anthropic_key or None),
        anthropic_model=(anthropic_model or None),
        xai_key=(xai_key or None),
        xai_model=(xai_model or None),
    )

    if isinstance(result, tuple) and len(result) == 3:
        backend, answer, diagnostics = result
    elif isinstance(result, tuple) and len(result) == 2:
        backend, answer = result
        diagnostics = None
    else:
        backend = "unknown"
        answer = str(result)
        diagnostics = None

    import math
    t1 = time.perf_counter()
    ms = int((t1 - t0) * 1000)
    tin = int(len(prompt) / 4)
    tout = int(len(answer or '') / 4)
    tps = (tout / max(0.001, (t1 - t0)))

    assistant = answer

    now = datetime.now().isoformat()
    history = history + [
        {"role": "user", "content": message, "time": now},
        {"role": "assistant", "content": assistant, "time": datetime.now().isoformat()},
    ]
    new_id = save_chat(history, chat_id)

    if diagnostics is None:
        diagnostics = {}
    diagnostics.setdefault('sources', sources)
    diagnostics['perf'] = {"ms": ms, "tin": tin, "tout": tout, "tps": tps}
    payload = {"id": new_id, "messages": history, "diagnostics": diagnostics}
    return JSONResponse(payload)
@app.post("/api/chat_stream")
def api_chat_stream(
    message: str = Body(..., embed=True),
    chat_id: Optional[str] = Body(None),
    temperature: float = Body(0.3),
    plan_mode: bool = Body(False),
    self_check: bool = Body(False),
    strict_mode: bool = Body(False),
    plan_depth: int = Body(2),
    pinned_path: Optional[str] = Body(None),
    pinned_paths: Optional[List[str]] = Body(None),
    pins_only: Optional[bool] = Body(False),
    backend: Optional[str] = Body(None),
    openai_key: Optional[str] = Body(None),
    openai_model: Optional[str] = Body(None),
    openrouter_key: Optional[str] = Body(None),
    openrouter_model: Optional[str] = Body(None),
    openai_compat_key: Optional[str] = Body(None),
    openai_compat_model: Optional[str] = Body(None),
    openai_compat_base_url: Optional[str] = Body(None),
    gemini_key: Optional[str] = Body(None),
    gemini_model: Optional[str] = Body(None),
    anthropic_key: Optional[str] = Body(None),
    anthropic_model: Optional[str] = Body(None),
    xai_key: Optional[str] = Body(None),
    xai_model: Optional[str] = Body(None),
    top_k: int = Body(None),
    top_p: float = Body(None),
    repeat_penalty: float = Body(None),
    freq_penalty: float = Body(None),
    retr_k: int = Body(None),
    max_context_tokens: int = Body(None),
 ):
    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    gen_opts = {
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "frequency_penalty": freq_penalty,
    }

    chosen = (backend or '').lower().strip()
    pins_only_flag = bool(pins_only) if pins_only is not None else False
    hosted = chosen in ('openai', 'openrouter', 'openai_compat', 'gemini', 'anthropic', 'xai')
    eff_retr_k = (retr_k or 5)
    if hosted and not pins_only_flag:
        eff_retr_k = min(eff_retr_k, 4)
    prompt, sources, plan_steps = prepare_prompt(
        message,
        plan_mode=plan_mode,
        plan_iters=plan_depth,
        strict_mode=strict_mode,
        retr_top_k=eff_retr_k,
        pinned_path=pinned_path,
        pinned_paths=pinned_paths,
        pins_only=pins_only_flag,
        max_context_tokens=max_context_tokens,
    )

    if not chosen:
        chosen = select_backend()
    use_stream = chosen in ('ollama', 'openai', 'openrouter', 'openai_compat', 'gemini', 'anthropic', 'xai')

    def gen():
        acc = []
        sent_done = False
        import time
        t0 = time.perf_counter()
        try:
            if use_stream and chosen == 'ollama':
                bounded = clamp_prompt_for_backend(prompt, 'ollama')
                for chunk in call_ollama_stream(bounded, temperature=temperature, gen_options=gen_opts):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                            ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'openai':
                from luau_rag import call_openai_stream as _oa_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'openai')
                for chunk in _oa_stream(bounded, temperature=temperature, api_key=(openai_key or None), model=(openai_model or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_openai as _oa_call
                    txt, _diag = _oa_call(bounded, temperature=temperature, api_key=(openai_key or None), model=(openai_model or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'openrouter':
                from luau_rag import call_openrouter_stream as _or_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'openai')
                for chunk in _or_stream(bounded, temperature=temperature, api_key=(openrouter_key or None), model=(openrouter_model or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_openrouter as _or_call
                    txt, _diag = _or_call(bounded, temperature=temperature, api_key=(openrouter_key or None), model=(openrouter_model or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'openai_compat':
                from luau_rag import call_openai_compat_stream as _oc_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'openai')
                for chunk in _oc_stream(bounded, temperature=temperature, api_key=(openai_compat_key or None), model=(openai_compat_model or None), base_url=(openai_compat_base_url or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_openai_compat as _oc_call
                    txt, _diag = _oc_call(bounded, temperature=temperature, api_key=(openai_compat_key or None), model=(openai_compat_model or None), base_url=(openai_compat_base_url or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'gemini':
                from luau_rag import call_gemini_stream as _gm_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'gemini')
                for chunk in _gm_stream(bounded, temperature=temperature, api_key=(gemini_key or None), model=(gemini_model or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_gemini as _gm_call
                    txt, _diag = _gm_call(bounded, temperature=temperature, api_key=(gemini_key or None), model=(gemini_model or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'anthropic':
                from luau_rag import call_anthropic_stream as _an_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'anthropic')
                for chunk in _an_stream(bounded, temperature=temperature, api_key=(anthropic_key or None), model=(anthropic_model or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_anthropic as _an_call
                    txt, _diag = _an_call(bounded, temperature=temperature, api_key=(anthropic_key or None), model=(anthropic_model or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            elif use_stream and chosen == 'xai':
                from luau_rag import call_xai_stream as _xa_stream
                had_chunk = False
                bounded = clamp_prompt_for_backend(prompt, 'xai')
                for chunk in _xa_stream(bounded, temperature=temperature, api_key=(xai_key or None), model=(xai_model or None)):
                    if chat_id and _cancel_flags.get(chat_id):
                        _cancel_flags.pop(chat_id, None)
                        break
                    if chunk:
                        acc.append(chunk)
                        had_chunk = True
                        yield json_dumps({"type":"chunk","text":chunk}, ensure_ascii=False) + "\n"
                if not had_chunk:
                    from luau_rag import call_xai as _xa_call
                    txt, _diag = _xa_call(bounded, temperature=temperature, api_key=(xai_key or None), model=(xai_model or None))
                    if txt:
                        acc.append(txt)
                        yield json_dumps({"type":"chunk","text":txt}, ensure_ascii=False) + "\n"
            else:
                # For non-stream, allow any backend as chosen; be robust to return shape
                res = ask(
                    message,
                    temperature=temperature,
                    plan_mode=plan_mode,
                    review_mode=self_check,
                    plan_iters=plan_depth,
                    strict_mode=strict_mode,
                    pinned_path=pinned_path,
                    pinned_paths=pinned_paths,
                    pins_only=pins_only_flag,
                    gen_options=gen_opts,
                    retr_top_k=eff_retr_k,
                    max_context_tokens=max_context_tokens,
                    backend_override=chosen,
                    openai_key=(openai_key or None),
                    openai_model=(openai_model or None),
                    openrouter_key=(openrouter_key or None),
                    openrouter_model=(openrouter_model or None),
                    openai_compat_key=(openai_compat_key or None),
                    openai_compat_model=(openai_compat_model or None),
                    openai_compat_base_url=(openai_compat_base_url or None),
                    gemini_key=(gemini_key or None),
                    gemini_model=(gemini_model or None),
                    anthropic_key=(anthropic_key or None),
                    anthropic_model=(anthropic_model or None),
                    xai_key=(xai_key or None),
                    xai_model=(xai_model or None),
                )
                try:
                    if isinstance(res, tuple) and len(res) == 3:
                        backend_used, text, _ = res
                    elif isinstance(res, tuple) and len(res) == 2:
                        backend_used, text = res
                    else:
                        backend_used, text = (chosen or 'unknown'), (str(res or ''))
                except Exception:
                    backend_used, text = (chosen or 'unknown'), ''
                acc.append(text)
                yield json_dumps({"type":"chunk","text":text}, ensure_ascii=False) + "\n"
        except Exception as e:
            # Surface a minimal error chunk so UI can close typing
            err = f"[error] {str(e)}"
            acc.append(err)
            yield json_dumps({"type":"chunk","text":err}, ensure_ascii=False) + "\n"
        finally:
            try:
                full = "".join(acc).strip()
                try:
                    sanitized = sanitize_answer(full)
                    # if sanitizer removed everything, keep original so UI shows something
                    full = sanitized if sanitized.strip() else (full or "No response generated. Check your context settings or backend configuration.")
                except Exception:
                    if not full:
                        full = "No response generated. Check your context settings or backend configuration."
                hist = load_chat(chat_id) if chat_id else []
                hist = hist + [{"role": "user", "content": message}, {"role": "assistant", "content": full}]
                cid = save_chat(hist, chat_id)
                t1 = time.perf_counter()
                ms = int((t1 - t0) * 1000)
                tin = int(len(prompt) / 4)
                tout = int(len(full) / 4)
                tps = (tout / max(0.001, (t1 - t0)))
                final = {"type": "done", "id": cid, "diagnostics": {"sources": sources, "plan_steps": plan_steps}, "perf": {"ms": ms, "tin": tin, "tout": tout, "tps": tps}}
                sent_done = True
                yield json_dumps_line(final)
            except Exception:
                # last resort: still send a done with a generated id
                try:
                    fallback_id = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
                    final = {"type": "done", "id": fallback_id, "diagnostics": {"sources": sources, "plan_steps": plan_steps}}
                    yield json_dumps_line(final)
                except Exception:
                    pass

    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/api/cancel")
def api_cancel(chat_id: Optional[str] = Body(None)):
    if chat_id:
        _cancel_flags[chat_id] = True
    return {"ok": True}

@app.post("/api/chat_edit")
def api_chat_edit(chat_id: str = Body(...), index: int = Body(...), content: str = Body(...)):
    msgs = load_chat(chat_id)
    if index < 0 or index >= len(msgs):
        raise HTTPException(status_code=400, detail="index out of range")
    msgs[index] = {**msgs[index], 'content': content}
    cid = save_chat(msgs, chat_id)
    return {"id": cid, "messages": msgs}

@app.post("/api/chat_delete_from")
def api_chat_delete_from(chat_id: str = Body(...), index: int = Body(...)):
    msgs = load_chat(chat_id)
    if index < 0 or index >= len(msgs):
        raise HTTPException(status_code=400, detail="index out of range")
    msgs = msgs[:index]
    cid = save_chat(msgs, chat_id)
    return {"id": cid, "messages": msgs}
@app.post("/api/clear_chat")
def api_clear_chat(chat_id: Optional[str] = Body(None)):
    cid = chat_id or datetime.now().strftime("chat_%Y%m%d_%H%M%S")
    messages: List[Dict[str, Any]] = []
    save_chat(messages, cid)
    return {"id": cid, "messages": messages}


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    fp = LOGS_DIR / f"{chat_id}.json"
    if fp.exists():
        fp.unlink()
    return {"ok": True}


@app.get("/api/source")
def api_source(path: str = Query(...)):
    """Return the content of a referenced source in a safe manner.
    Path must be relative to the repository root; traversal outside is blocked.
    """
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    # Normalize separators and resolve
    rel = Path(path.replace("\\", "/").lstrip("/"))
    abs_path = (ROOT / rel).resolve()
    try:
        # ensure within repo
        ROOT.resolve()
    except Exception:
        pass
    if not str(abs_path).startswith(str(ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    try:
        text = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    suffix = abs_path.suffix.lower()
    kind = "text"
    language = None
    if suffix == ".md":
        kind = "markdown"
    elif suffix == ".lua":
        kind = "code"; language = "lua"
    elif suffix in (".json", ".toml", ".py", ".txt"):
        kind = "code"; language = suffix.lstrip(".")

    return JSONResponse({
        "path": str(rel),
        "kind": kind,
        "language": language,
        "content": text,
    })


@app.get("/api/docs_list")
def api_docs_list():
    """Return a recursive list of markdown files under Docs/."""
    base = ROOT / "Docs"
    items = []
    if not base.exists():
        return {"base": str(base), "files": []}
    for fp in base.rglob("*.md"):
        try:
            rel = fp.relative_to(ROOT).as_posix()
            items.append(rel)
        except Exception:
            continue
    items.sort()
    return {"base": str(base), "files": items}


@app.get("/api/docs_read")
def api_docs_read(path: str = Query(...)):
    """Read a markdown file under Docs/. Blocks HTML or other types."""
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    rel = Path(path.replace("\\", "/").lstrip("/"))
    abs_path = (ROOT / rel).resolve()
    docs_root = (ROOT / "Docs").resolve()
    if not str(abs_path).startswith(str(docs_root)):
        raise HTTPException(status_code=400, detail="Path must be under Docs/")
    if abs_path.suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="Only .md files are allowed")
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        text = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
    return JSONResponse({"path": rel.as_posix(), "content": text})


@app.get("/api/tooltips")
def api_tooltips():
    """Return tooltips.json content with light sanitization to strip mojibake."""
    fp = ROOT / "tooltips.json"
    if not fp.exists():
        return JSONResponse({"_meta": {"version": "0", "about": "no tooltips.json present"}}, status_code=200)
    try:
        raw = fp.read_text(encoding="utf-8-sig", errors="replace")
        # Strip any residual BOM if present
        raw = raw.lstrip("\ufeff")
        # Basic cleanups for common mojibake seen in this repo
        cleaned = (raw
                   .replace("â€¦", "...")
                   .replace("â€“", "-")
                   .replace("Â", "")
                   .replace("�", ""))
        data = json.loads(cleaned)
        # Also scrub any stray replacement chars inside strings
        def scrub(obj):
            if isinstance(obj, dict):
                return {k: scrub(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [scrub(v) for v in obj]
            if isinstance(obj, str):
                return obj.replace("�", "").replace("Â", "")
            return obj
        data = scrub(data)
        return JSONResponse(data)
    except Exception as e:
        # Be resilient: return a small stub instead of a 500 to keep UI usable
        print(f"tooltips.json load error: {e}")
        return JSONResponse({
            "_meta": {"version": "0", "about": "tooltips load error"},
            "error": str(e),
            "services": {}, "classes": {}, "datatypes": {}, "functions": {},
            "globals": {}, "keywords": {}, "enums": {}, "properties": {}, "symbols": {}
        }, status_code=200)

## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
def json_dumps_line(obj):
    return json_dumps(obj, ensure_ascii=False) + "\n"


def main():
    # Mount static UI last so that /api routes take precedence
    mount_static_ui()
    # Attempt to load/ensure API dump at startup so status shows immediately
    try:
        from api_validator import ensure_api_dump as _ensure
        ensured = _ensure(str(ROOT))
        if ensured or load_api_dump(str(ROOT)):
            from api_validator import get_stats as _stats
            s = _stats()
            print(f"API dump loaded: {s.get('path','?')} ({s.get('property_count',0)} properties)")
        else:
            print("API dump not detected in Docs/ (place ApiDump.json or api_dump.json under Docs)")
    except Exception as e:
        print(f"API dump load check failed: {e}")
    import uvicorn
    port = int(os.environ.get("PORT", find_free_port(7861)))
    print(f"HTTP UI available at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
