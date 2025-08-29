import os, glob, re, pickle, time
import numpy as np
from PyPDF2 import PdfReader
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "text-embedding-004"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 100  # <= лимит API

def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        pdf = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in pdf.pages])
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(t: str):
    t = re.sub(r"\s+", " ", t).strip()
    if not t: return []
    out, i, step = [], 0, CHUNK_SIZE - CHUNK_OVERLAP
    while i < len(t):
        out.append(t[i:i+CHUNK_SIZE])
        i += step
    return out

def embed_batch(texts):
    """Встраиваем списком, батчами по 100."""
    embs_all = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[{"parts":[{"text":x}]} for x in batch]
        )
        embs = [e.values for e in resp.embeddings]
        embs_all.extend(embs)
        # микропаузa, чтобы не упереться в rate limit
        time.sleep(0.2)
    embs_all = np.array(embs_all, dtype="float32")
    embs_all /= (np.linalg.norm(embs_all, axis=1, keepdims=True) + 1e-12)
    return embs_all

def main():
    files = []
    for ext in ("*.txt","*.md","*.pdf"):
        files += glob.glob(os.path.join("kb", ext))
    assert files, "Папка kb пуста."

    docs, meta = [], []
    for path in files:
        chs = chunk_text(read_file(path))
        for j, ch in enumerate(chs):
            docs.append(ch)
            meta.append({"source": os.path.basename(path), "chunk_id": j})
    assert docs, "Файлы пустые после парсинга."

    print(f"[prepare] Чанков: {len(docs)} из {len(files)} файлов… (встраиваю)")
    embs = embed_batch(docs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    with open("kb.pkl","wb") as f:
        pickle.dump({"docs": docs, "meta": meta}, f)
    faiss.write_index(index, "kb.index")
    print(f"[prepare] OK: {len(docs)} чанков. Файлы: kb.index, kb.pkl")

if __name__ == "__main__":
    main()
