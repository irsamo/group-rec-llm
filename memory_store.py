# memory_store.py
import os, json, time
from typing import List, Dict, Optional, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim
_DIM = 384
_INDEX_PATH = "user_prefs.faiss"
_META_PATH  = "user_prefs_meta.json"
_KV_PATH    = "user_prefs_kv.json"

def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

_index = faiss.read_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else faiss.IndexFlatIP(_DIM)
_meta  = _load_json(_META_PATH, {})
_kv    = _load_json(_KV_PATH, {})

def _save():
    faiss.write_index(_index, _INDEX_PATH)
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f)
    with open(_KV_PATH, "w", encoding="utf-8") as f:
        json.dump(_kv, f)

def _embed_texts(texts: List[str]) -> np.ndarray:
    vecs = _MODEL.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def upsert_user_prefs(user_id: str, prefs: List[str], importance: int = 3) -> None:
    text = "; ".join(prefs)
    ts = int(time.time())
    _kv[user_id] = {"text": text, "ts": ts, "importance": int(importance)}
    vec = _embed_texts([text])
    row_id = len(_meta)
    _index.add(vec)
    _meta[row_id] = {"user_id": user_id, "ts": ts, "importance": int(importance)}
    _save()

def upsert_user_prefs_batch(records: Iterable[Dict[str, object]], importance: int = 3, save_every: int = 5000) -> None:
    texts, ids, ts = [], [], int(time.time())
    for rec in records:
        uid = str(rec["user_id"])
        prefs = [t for t in rec["prefs"] if t]
        text = "; ".join(prefs) if prefs else ""
        if not text:
            continue
        _kv[uid] = {"text": text, "ts": ts, "importance": int(importance)}
        texts.append(text); ids.append(uid)
        if len(texts) == save_every:
            vecs = _embed_texts(texts)
            start = len(_meta)
            _index.add(vecs)
            for j, uid_j in enumerate(ids):
                _meta[start + j] = {"user_id": uid_j, "ts": ts, "importance": int(importance)}
            _save(); texts, ids = [], []
    if texts:
        vecs = _embed_texts(texts)
        start = len(_meta)
        _index.add(vecs)
        for j, uid_j in enumerate(ids):
            _meta[start + j] = {"user_id": uid_j, "ts": ts, "importance": int(importance)}
        _save()

def get_prefs_exact(user_id: str) -> Optional[List[str]]:
    rec = _kv.get(str(user_id))
    if not rec: return None
    return [t.strip() for t in rec["text"].split(";") if t.strip()]

def retrieve_similar_prefs(seed_prefs: List[str], top_k: int = 5) -> List[Dict]:
    if _index.ntotal == 0: return []
    text = "; ".join(seed_prefs)
    q = _embed_texts([text])
    scores, idxs = _index.search(q, top_k)
    now = time.time()
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        m = _meta.get(int(i))
        if not m: continue
        rec = _kv.get(m["user_id"], {})
        recency = max(0.0, 1.0 - (now - m.get("ts", now)) / (30*24*3600))
        imp = float(rec.get("importance", 3))
        boost = 0.05*recency + 0.02*imp
        out.append({"user_id": m["user_id"], "text": rec.get("text",""), "score": float(s + boost)})
    return sorted(out, key=lambda d: d["score"], reverse=True)
