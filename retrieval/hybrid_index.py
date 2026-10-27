#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk legal source files and build hybrid indices (BM25 + FAISS).
- Reads cleaned, section-marked text files from a supplied directory.
- Produces:
  - indices/chunks.jsonl               (all chunks + metadata)
  - indices/sparse/bm25.pkl            (BM25Okapi + tokenized docs)
  - indices/sparse/chunk_ids.json      (chunk id order)
  - indices/vector/index.faiss         (FAISS index, inner product)
  - indices/vector/meta.json           (chunk id → minimal metadata)
- Optional: --query "text" to test hybrid retrieval via RRF
"""

import argparse, json, re, pickle
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Sparse
from rank_bm25 import BM25Okapi

# Dense
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CLEAN_DIR = Path("cases_clean")
OUT_DIR = Path("indices")
HEADER_RE = re.compile(r"^\s*===\s*([A-Z ]+)\s*===\s*$")
PIN_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(\"])")

KEEP_SECTIONS = {"BODY", "FACTS", "EXTRACTS", "RESULT"}  # META is for metadata only

def parse_sections(text: str) -> List[Tuple[str, List[str]]]:
    lines = text.splitlines()
    sections = []
    current_name = "BODY"
    current = []
    for ln in lines:
        m = HEADER_RE.match(ln)
        if m:
            sections.append((current_name, current))
            current_name = m.group(1).strip().upper()
            current = []
        else:
            current.append(ln)
    sections.append((current_name, current))
    return sections

def extract_meta(sections) -> Dict[str, str]:
    meta = {}
    for name, lines in sections:
        if name == "META":
            for ln in lines:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            break
    # fallback case name from filename happens later
    return meta

def is_statute(meta_case: str, filename: str) -> bool:
    nm = (meta_case or "").lower()
    if any(x in nm for x in ["act", "regulation", "regulations", "code"]):
        return True
    if " v " not in filename.replace("_", " ").lower():
        return True
    return False

def paragraphs(lines: List[str]) -> List[str]:
    """Group lines into paragraphs (blank-line separated)."""
    paras, buf = [], []
    for ln in lines:
        if ln.strip() == "":
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(ln.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    # filter empties
    return [p for p in paras if p]

def detect_section_labels_for_statute(lines: List[str]) -> List[Tuple[str, str]]:
    """
    For statutes, try to detect section headers like '13A Title...' and attach them.
    Returns list of (section_label, paragraph_text)
    """
    out = []
    current_label = None
    paras = paragraphs(lines)
    for p in paras:
        m = re.match(r"^\s*(\d+[A-Z]?)\b", p)
        if m and len(p) < 400:  # short header-like line
            current_label = m.group(1)
            continue
        out.append((current_label or "", p))
    # If no headers detected, just return with empty labels
    if any(lbl for lbl,_ in out):
        return out
    return [("", p) for p in paras]

def pins_in_text(txt: str) -> List[str]:
    return PIN_RE.findall(txt)

def chunk_paragraphs(paras: List[str], target_words=(120, 220)) -> List[str]:
    """
    Merge complete sentences into chunks roughly in the 120–220 word range.
    """
    lo, hi = target_words
    units = [sentence.strip() for paragraph in paras
             for sentence in SENTENCE_BOUNDARY_RE.split(paragraph)
             if sentence.strip()]
    chunks, buf, wc = [], [], 0
    for p in units:
        w = len(p.split())
        if wc + w <= hi:
            buf.append(p); wc += w
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf, wc = [p], w
    if buf:
        chunks.append(" ".join(buf))
    # If any chunk is very short, try to merge with previous
    merged = []
    for c in chunks:
        if merged and len(c.split()) < lo:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged

def build_chunks_for_file(fp: Path) -> List[Dict]:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    sections = parse_sections(text)
    meta = extract_meta(sections)
    case_name = meta.get("case", "") or fp.stem.replace("_", " ")
    citation = meta.get("citation", "")
    ttype = "statute" if is_statute(case_name, fp.name) else "case"

    chunks = []
    seq = 0

    for name, lines in sections:
        name = name.upper()
        if name not in KEEP_SECTIONS:
            continue

        if ttype == "statute":
            labeled_paras = detect_section_labels_for_statute(lines)
            # chunk per paragraph (statutes tend to be terse)
            for sec_label, para in labeled_paras:
                if not para.strip():
                    continue
                seq += 1
                chunks.append({
                    "id": f"{fp.stem}::#{seq}",
                    "text": para,
                    "meta": {
                        "file": fp.name,
                        "case": case_name,
                        "citation": citation,
                        "type": ttype,
                        "section": name,
                        "statute_section": sec_label
                    },
                    "pins": pins_in_text(para),
                    "priority": 1.0
                })
        else:
            paras = paragraphs(lines)
            # Give EXTRACTS a slight priority and smaller chunk size
            if name == "EXTRACTS":
                target = (60, 140)
                prio = 1.25
            else:
                target = (120, 220)
                prio = 1.0
            for chunk in chunk_paragraphs(paras, target_words=target):
                seq += 1
                chunks.append({
                    "id": f"{fp.stem}::#{seq}",
                    "text": chunk,
                    "meta": {
                        "file": fp.name,
                        "case": case_name,
                        "citation": citation,
                        "type": ttype,
                        "section": name
                    },
                    "pins": pins_in_text(chunk),
                    "priority": prio
                })

    return chunks

def tokenise(s: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(s)]

def save_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def build_indices(clean_dir: Path, out_dir: Path, model_name: str, show_progress=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = out_dir / "vector"
    sparse_dir = out_dir / "sparse"
    chunks_path = out_dir / "chunks.jsonl"
    vector_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build chunks
    all_chunks = []
    files = sorted(clean_dir.glob("*.txt"))
    for fp in tqdm(files, desc="Chunking", disable=not show_progress):
        cs = build_chunks_for_file(fp)
        all_chunks.extend(cs)

    if not all_chunks:
        raise SystemExit("No chunks produced. Did you run the cleaner?")

    save_jsonl(chunks_path, all_chunks)

    # 2) Sparse index (BM25)
    tokenised_docs = [tokenise(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenised_docs)
    with (sparse_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    with (sparse_dir / "chunk_ids.json").open("w", encoding="utf-8") as f:
        json.dump([c["id"] for c in all_chunks], f)

    # 3) Dense index (FAISS) with cosine (normalised vectors + inner product)
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in all_chunks]

    embs = []
    bs = 64
    for i in tqdm(range(0, len(texts), bs), desc="Embedding", disable=not show_progress):
        vecs = model.encode(texts[i:i+bs], normalize_embeddings=True, show_progress_bar=False)
        embs.append(vecs.astype("float32"))
    embs = np.vstack(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(vector_dir / "index.faiss"))

    # Meta needed to map vector rows back to chunks
    meta = {
        "model_name": model_name,
        "chunk_ids": [c["id"] for c in all_chunks],
        "meta": [{"case": c["meta"]["case"], "citation": c["meta"].get("citation",""), "file": c["meta"]["file"]} for c in all_chunks]
    }
    (vector_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {len(all_chunks)} chunks | BM25 and FAISS saved under {out_dir}")

def hybrid_search(query: str, topk: int = 10, index_dir: Path = OUT_DIR) -> List[Tuple[float, Dict]]:
    """Quick test search using RRF fusion of BM25 and FAISS (no reranker yet)."""
    index_dir = Path(index_dir)
    chunks = load_jsonl(index_dir / "chunks.jsonl")
    # load sparse
    with (index_dir / "sparse" / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    # load dense
    index = faiss.read_index(str(index_dir / "vector" / "index.faiss"))
    meta = json.loads((index_dir / "vector" / "meta.json").read_text(encoding="utf-8"))

    # sparse scores
    q_tok = tokenise(query)
    sp_scores = bm25.get_scores(q_tok)
    sp_rank = np.argsort(-sp_scores)[:50]  # indices

    # dense scores
    model = SentenceTransformer(meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    qv = model.encode([query], normalize_embeddings=True)
    D, I = index.search(qv.astype("float32"), 50)  # distances (inner product)
    dn_rank = [int(idx) for idx in I[0] if idx >= 0]

    # RRF
    ranks = {}
    for r, idx in enumerate(sp_rank, start=1):
        ranks.setdefault(idx, 0.0)
        ranks[idx] += 1.0 / (60 + r)
    for r, idx in enumerate(dn_rank, start=1):
        ranks.setdefault(idx, 0.0)
        ranks[idx] += 1.0 / (60 + r)

    fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = []
    for idx, score in fused:
        ch = chunks[idx]
        results.append((score, ch))
    return results

def main():
    ap = argparse.ArgumentParser(description="Chunk cleaned cases and build hybrid indices.")
    ap.add_argument("--clean_dir", default=str(CLEAN_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--query", default=None, help="Test a query after indexing")
    args = ap.parse_args()

    build_indices(Path(args.clean_dir), Path(args.out_dir), args.model)

    if args.query:
        print(f"\n[hybrid search] {args.query}\n")
        for score, ch in hybrid_search(args.query, topk=10, index_dir=Path(args.out_dir)):
            meta = ch["meta"]
            pin = f" pins={','.join(ch.get('pins', [])[:3])}" if ch.get("pins") else ""
            snippet = (ch["text"][:300] + "…") if len(ch["text"]) > 300 else ch["text"]
            print(f"{score:.4f}  | {meta.get('case','?')}  {meta.get('citation','')}  [{meta.get('section','')}] {pin}\n  {snippet}\n")

if __name__ == "__main__":
    main()
