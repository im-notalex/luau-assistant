# embeddings_index.py
import os
import glob
import math
import hashlib
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

ROOT = os.path.dirname(__file__)
MARKDOWN_PATH = os.path.join(ROOT, "Docs", "markdown")
CHROMA_PATH = os.path.join(ROOT, "chroma")
CHUNK_SIZE = 1000  # characters per chunk (tuneable)
CHUNK_OVERLAP = 200  # overlap to keep continuity across chunks

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunking with overlap for better retrieval continuity."""
    if size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0
    chunks = []
    start = 0
    L = len(text)
    step = max(1, size - overlap)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start += step
    return chunks


def make_doc_id(path: str, idx: int, content: str) -> str:
    """Create a globally unique, stable ID for a chunk.
    Uses file path (relative) + chunk index + content hash to avoid collisions
    when multiple files share the same name (e.g., init.lua.md) across repos.
    """
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    # slugify path (keep alnum, dot, dash, underscore, slash)
    slug = re.sub(r"[^A-Za-z0-9._/\-]", "-", rel).strip("-")
    h = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()[:8]
    # Replace slashes to keep Chroma IDs simple
    slug = slug.replace("/", "__")
    return f"{slug}__{idx}__{h}"

def main():
    if not os.path.exists(MARKDOWN_PATH):
        print(f"No folder found at {MARKDOWN_PATH}. Run the downloader first or place .md files there.")
        return

    md_files = []
    for root, _, files in os.walk(MARKDOWN_PATH):
        for f in files:
            if f.lower().endswith(".md"):
                md_files.append(os.path.join(root, f))

    if len(md_files) == 0:
        print(f"No .md files found in {MARKDOWN_PATH}. Please populate it (run download_luau_docs.bat) and try again.")
        return

    print(f"Found {len(md_files)} markdown files. Preparing embeddings...")

    # sentence-transformers model (higher recall for code/doc): BAAI/bge-small-en-v1.5
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
    # Chroma client (persistent)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection("luau_docs")
    except Exception:
        collection = client.create_collection("luau_docs")

    ids = []
    docs = []
    metadatas = []

    total_chunks = 0
    used_ids = set()
    for path in md_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read().strip()
            if not text:
                continue
            chunks = chunk_text(text, CHUNK_SIZE)
            for i, c in enumerate(chunks):
                uid = make_doc_id(path, i, c)
                # Ensure uniqueness within this batch as well
                if uid in used_ids:
                    # Rare: adjust with a different short hash suffix
                    h2 = hashlib.sha1(f"{uid}-{len(used_ids)}".encode()).hexdigest()[:6]
                    uid = f"{uid}__{h2}"
                used_ids.add(uid)
                ids.append(uid)
                docs.append(c)
                metadatas.append({"source": os.path.relpath(path, ROOT)})
            total_chunks += len(chunks)

    if total_chunks == 0:
        print("No non-empty chunks extracted. Check your markdown files.")
        return

    print(f"Embedding {total_chunks} chunks with {getattr(embed_model, 'model_card', None) or 'bge-small-en-v1.5'}...")
    embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    # Add to chroma (can raise if duplicate ids exist - avoid duplicates by upserting)
    try:
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings.tolist())
    except Exception as e:
        # fallback: upsert (safer on re-runs)
        print("Add failed (maybe duplicates). Trying upsert...")
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings.tolist())
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
    print(f"✅ Indexed {total_chunks} chunks into ChromaDB at '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()
