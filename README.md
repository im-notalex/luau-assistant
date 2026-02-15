# LuaU RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-FF6F00)](https://www.sbert.net/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-7C3AED)](https://www.trychroma.com/)
[![UI: Vanilla JS](https://img.shields.io/badge/UI-Vanilla%20JS-F7DF1E?logo=javascript&logoColor=111)](ui/)
[![Optional: Ollama](https://img.shields.io/badge/Optional-Ollama-111827)](https://ollama.com/)
[![Providers](https://img.shields.io/badge/Providers-OpenAI%20%7C%20OpenRouter%20%7C%20Gemini%20%7C%20Anthropic%20%7C%20xAI-2563EB)](#backends)

Version: `v2.4.2`

Local-first RAG assistant for Roblox LuaU development. It indexes docs under `Docs/` and serves a web UI for chat, retrieval, and API lookup.

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Quick Start

Run setup steps:

```bash
python setup_verify.py
python api_validator.py
```

Start the UI:

```bash
python web_ui.py
```

Then open the printed local URL in your browser.

## Backends

The UI supports:

- Local (`llama.cpp`)
- Ollama
- OpenAI
- OpenRouter
- OpenAI-compatible endpoints
- Gemini
- Anthropic
- xAI

Configure provider keys/models in the Settings drawer. Keys are stored locally in the browser.

## Project Layout

- `web_ui.py` FastAPI app + API routes
- `luau_rag.py` retrieval/prompting/backend calls
- `ui/` frontend assets
- `Docs/` markdown sources for indexing
- `chroma/` local vector store (generated at runtime)
- `logs/` chat/session logs (generated at runtime)

## Troubleshooting

- Missing dependencies:
  - Re-run `pip install -r requirements.txt`
- Empty/weak retrieval:
  - Ensure docs are in `Docs/markdown`
  - Re-run setup scripts
- Hosted API auth errors:
  - Verify key/model/base URL fields in UI settings
- Port in use:
  - Stop the other process or set a different port via `PORT`

## Notes

- This is a personal project and is not affiliated with Roblox.
- Internet is only needed for dependency install and hosted model endpoints.

## License

MIT. See `LICENSE`.
