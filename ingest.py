#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCSB Assistant - Ingest
Builds a local knowledge base from:
  A) Local files you specify (PDF/TXT/HTML)  <-- default / recommended to start
  B) Optional web crawl over UCSB/A.S. sites <-- only runs if --crawl + --seed specified

Outputs:
  - data/corpus.jsonl         (chunks & metadata)
  - index/meta.json           (metas used by the app)
  - index/faiss.index         (if faiss is available)
  - index/embeddings.npy      (always; NumPy fallback)

Examples
--------
# Only two local PDFs (recommended start)
python ingest.py --files "/path/AS_Legal_Code.pdf" "/path/AS_Financial_Policies.pdf"

# Add a folder of PDFs
python ingest.py --dir ./docs

# Later, enable crawling (explicit)
python ingest.py --crawl --seed https://www.as.ucsb.edu/legal-resource-center/ \
                 --seed https://www.as.ucsb.edu/documents/governing-documents/ \
                 --max-pages 200 --max-depth 2 --ignore-robots
"""

import os, re, io, json, time, hashlib, argparse
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ------------------- PATHS -------------------

DATA_DIR  = Path("data")
RAW_DIR   = DATA_DIR / "raw"
INDEX_DIR = Path("index")
MANIFEST  = RAW_DIR / "manifest.jsonl"

# ------------------- CHUNKING / EMBEDDING -------------------

EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS       = 650       # friendly for big policy PDFs
CHUNK_OVERLAP     = 120
MIN_CHARS_PER_CH  = 400       # drop tiny/noisy fragments

# ------------------- CRAWL (opt-in) -------------------

CRAWL_DELAY       = 0.15
BLOCKLIST = [
    r"\?page=\d+", r"[\?&]p=\d+", r"/tag/", r"/category/",
    r"/feed/?$", r"/wp-json", r"/search\?", r"/\?s=",
    r"my\.sa\.ucsb\.edu/.+aspx",
    r"map\.ucsb\.edu/.+layer",
    r"catalog\.ucsb\.edu/.+/archive",
    # enable these later if crawl explodes:
    # r"/events?", r"/event/", r"/calendar",
]
ALLOWED_CONTENT_TYPES = ("text/html",)
HTML_ENDINGS = (".html", "/", "", ".htm")
USER_AGENT   = "UCSB-Assistant-Ingest/2.0 (+local)"

# ------------------- LABEL HEURISTICS -------------------

TOPIC_RULES = [
    ("legal",   [r"legal", r"judicial", r"governing[- ]documents", r"legal[- ]code",
                 r"landlord", r"tenant", r"lease", r"evict", r"security[- ]deposit", r"habitability"]),
    ("finance", [r"finance", r"requisition", r"purchase", r"vendor", r"invoice", r"reimbursement"]),
    ("food",    [r"food[- ]bank", r"calfresh", r"basic needs"]),
    ("bike",    [r"bike[- ]shop"]),
    ("housing", [r"housing", r"residence[- ]hall", r"apartment"]),
]

# ------------------- UTILS -------------------

def safe_name_from_url(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    path = urlparse(url).path.replace("/", "_") or "index"
    return f"{path}_{h}"

def safe_name_from_local(path: Path) -> str:
    h = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:10]
    base = path.stem[:60].replace(" ", "_")
    return f"{base}_{h}{path.suffix.lower()}"

def get(url: str, headers=None):
    hh = {"User-Agent": USER_AGENT}
    if headers: hh.update(headers)
    return requests.get(url, headers=hh, timeout=25)

def url_allowed(url: str, allowed_hosts: set) -> bool:
    host = urlparse(url).netloc.split(":")[0].lower()
    return any(host == d or host.endswith("." + d) for d in allowed_hosts) if allowed_hosts else True

def url_blocked(url: str) -> bool:
    for pat in BLOCKLIST:
        if re.search(pat, url, flags=re.IGNORECASE):
            return True
    return False

def is_html_like(url: str) -> bool:
    p = urlparse(url).path
    return p.endswith(HTML_ENDINGS)

def chunk_text(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    i = 0
    while i < len(words):
        yield " ".join(words[i:i+size])
        i += max(1, size - overlap)

def infer_labels(url: str, text: str):
    text_snip = (text[:2000] or "").lower()
    url_l = (url or "").lower()
    labs = set()
    for lab, pats in TOPIC_RULES:
        for p in pats:
            if re.search(p, url_l) or re.search(p, text_snip):
                labs.add(lab); break
    return sorted(labs)

def write_manifest(url_or_path, saved_as, ctype, depth, origin, extra=None):
    rec = {
        "origin": origin,           # "local" or "web"
        "url_or_path": str(url_or_path),
        "file": str(saved_as),
        "content_type": ctype,
        "depth": depth,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    if extra: rec.update(extra)
    with MANIFEST.open("a", encoding="utf-8") as w:
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------- LOCAL FILE INGEST -------------------

def pdf_bytes_to_text(data: bytes) -> str:
    # Try pdfplumber (better), fallback to pypdf
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""

def html_to_text_str(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","nav"]):
        tag.decompose()
    sections, cur = [], []
    for el in soup.find_all(["h1","h2","h3","p","li"]):
        if el.name in ("h1","h2","h3"):
            if cur: sections.append("\n".join(cur)); cur = []
            sections.append(el.get_text(" ", strip=True))
        else:
            cur.append(el.get_text(" ", strip=True))
    if cur: sections.append("\n".join(cur))
    txt = "\n\n".join([s for s in sections if s.strip()])
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def import_local_files(files: list[Path]) -> list[Path]:
    """Copy selected local files into data/raw and write manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for fp in files:
        if not fp.exists() or not fp.is_file():
            print(f"skip (missing): {fp}")
            continue
        ext = fp.suffix.lower()
        if ext not in (".pdf", ".txt", ".md", ".html", ".htm"):
            print(f"skip (unsupported): {fp}")
            continue
        out_name = safe_name_from_local(fp)
        out_path = RAW_DIR / out_name
        # copy bytes
        data = fp.read_bytes()
        out_path.write_bytes(data)
        ctype = "application/pdf" if ext==".pdf" else ("text/html" if ext in (".html",".htm") else "text/plain")
        write_manifest(fp, out_path, ctype, depth=0, origin="local")
        saved.append(out_path)
    print(f"Local files imported: {len(saved)}")
    return saved

# ------------------- OPTIONAL CRAWLER -------------------

def get_robot_parser(base):
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        pass
    return rp

def sitemap_urls(base):
    try:
        r = get(urljoin(base, "/robots.txt"))
        if r.status_code != 200: return []
        return [ln.split(": ",1)[1].strip()
                for ln in r.text.splitlines() if ln.lower().startswith("sitemap:")]
    except Exception:
        return []

def crawl(seeds: list[str], allowed_hosts: set, max_pages: int, max_depth: int, respect_robots: bool):
    if not seeds:
        print("No seeds provided; skipping crawl.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen, enqueued = set(), set()
    saved_pdfs = saved_html = 0

    queue = deque([(u, 0) for u in seeds])

    # Add sitemap links (best-effort)
    for seed in seeds:
        for sm in sitemap_urls(seed):
            try:
                sr = get(sm)
                if sr.status_code == 200:
                    for ln in re.findall(r"<loc>(.*?)</loc>", sr.text, flags=re.I):
                        ln = ln.strip()
                        if url_allowed(ln, allowed_hosts) and not url_blocked(ln) and is_html_like(ln):
                            queue.append((ln, 1))
            except Exception:
                pass

    robots = {}
    pbar = tqdm(total=max_pages, desc="Crawling (opt-in)", unit="pg")
    while queue and len(seen) < max_pages:
        url, depth = queue.popleft()

        if url in seen:
            continue
        if not url_allowed(url, allowed_hosts):
            # print("skip: host not allowed", url)
            continue
        if url_blocked(url) and url not in seeds:
            # print("skip: blocklist", url)
            continue

        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if respect_robots:
            if host not in robots:
                robots[host] = get_robot_parser(host)
            rp = robots[host]
            if rp and hasattr(rp,"can_fetch") and not rp.can_fetch(USER_AGENT, url):
                # print("skip: robots disallow", url)
                continue

        seen.add(url)
        try:
            r = get(url)
            if r.status_code != 200:
                # print("skip: status", r.status_code, url)
                pbar.update(1); time.sleep(CRAWL_DELAY); continue

            ctype = r.headers.get("Content-Type","").split(";")[0].lower()
            etag = r.headers.get("ETag"); lastmod = r.headers.get("Last-Modified")

            if "application/pdf" in ctype or url.lower().endswith(".pdf"):
                fn = RAW_DIR / f"{safe_name_from_url(url)}.pdf"
                fn.write_bytes(r.content)
                write_manifest(url, fn, "application/pdf", depth, origin="web",
                               extra={"etag": etag, "lastmod": lastmod})
                saved_pdfs += 1

            elif any(t in ctype for t in ALLOWED_CONTENT_TYPES):
                fn = RAW_DIR / f"{safe_name_from_url(url)}.html"
                fn.write_text(r.text, encoding="utf-8", errors="ignore")
                write_manifest(url, fn, "text/html", depth, origin="web",
                               extra={"etag": etag, "lastmod": lastmod})
                saved_html += 1

                if depth < max_depth:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for a in soup.select("a[href]"):
                        href = urljoin(url, a["href"]).split("#")[0]
                        if (url_allowed(href, allowed_hosts) and not url_blocked(href)
                            and is_html_like(href)
                            and href not in seen and href not in enqueued):
                            queue.append((href, depth+1))
                            enqueued.add(href)

        except Exception as e:
            print("err:", url, e)

        pbar.update(1); time.sleep(CRAWL_DELAY)

    pbar.close()
    print(f"Done crawl: pages={len(seen)}, html={saved_html}, pdfs={saved_pdfs}")

# ------------------- PARSE & BUILD CORPUS -------------------

def parse_and_chunk_all() -> Path:
    """Parse everything in data/raw into corpus.jsonl with heading-aware chunking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records, seen_hashes = [], set()

    # Build quick map from manifest for url attribution
    url_by_file = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            url_by_file[Path(rec["file"]).name] = rec["url_or_path"]

    files = list(RAW_DIR.glob("*"))
    for f in tqdm(files, desc="Parsing & chunking", unit="file"):
        ext = f.suffix.lower()
        text = ""
        try:
            if ext == ".pdf":
                text = pdf_bytes_to_text(f.read_bytes())
            elif ext in (".html", ".htm"):
                text = html_to_text_str(f.read_text(encoding="utf-8", errors="ignore"))
            elif ext in (".txt", ".md"):
                text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print("parse err:", f, e)
            continue

        if not text or len(text.strip()) < 200:
            continue

        for i, ch in enumerate(chunk_text(text)):
            ch_norm = re.sub(r"\s+", " ", ch).strip()
            if len(ch_norm) < MIN_CHARS_PER_CH:
                continue
            h = hashlib.md5(ch_norm.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            url = url_by_file.get(f.name, f.name)
            labels = infer_labels(url, ch_norm)
            records.append({
                "source": f.name,
                "url": url if isinstance(url, str) else str(url),
                "path": str(f),
                "type": ext.lstrip("."),
                "chunk": i,
                "labels": labels,
                "text": ch_norm
            })

    out = DATA_DIR / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} chunks → {out}")
    return out

# ------------------- INDEX -------------------

def build_index(corpus_path: Path):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Missing dependency 'sentence-transformers'. Install it first.")
    import numpy as np

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    metas, texts = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            metas.append(r)
            texts.append(r["text"])

    if not texts:
        print("No texts found in corpus; skipping index build.")
        return

    # Embed with progress
    from tqdm import tqdm as _tqdm
    model = SentenceTransformer(EMBED_MODEL)
    vecs = []
    for t in _tqdm(texts, desc="Embedding chunks", unit="chunk"):
        v = model.encode(t, normalize_embeddings=True)
        vecs.append(v)

    mat = np.vstack(vecs).astype("float32")

    # Try FAISS; always save embeddings.npy for fallback
    faiss_ok = False
    try:
        import faiss
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        faiss_ok = True
        print("FAISS index written.")
    except Exception as e:
        print("FAISS unavailable; using NumPy fallback only.", e)

    with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as w:
        json.dump(metas, w)
    np.save(INDEX_DIR / "embeddings.npy", mat)
    print(f"Index built: vectors={len(metas)} dim={mat.shape[1]} (faiss_ok={faiss_ok})")

# ------------------- MAIN -------------------

def main():
    parser = argparse.ArgumentParser(description="UCSB Assistant - Ingest")

    # Local files (default path)
    parser.add_argument("--files", nargs="*", default=[], help="Explicit file paths to ingest (PDF/TXT/HTML).")
    parser.add_argument("--dir", dest="dirs", nargs="*", default=[], help="Directories whose PDFs/TXT/HTML to ingest.")

    # Crawl (opt-in)
    parser.add_argument("--crawl", action="store_true", help="Enable web crawling (off by default).")
    parser.add_argument("--seed", dest="seeds", nargs="*", default=[], help="Seed URLs to crawl (required with --crawl).")
    parser.add_argument("--allow-host", dest="allow_hosts", nargs="*", default=[], help="Additional allowed hostnames.")
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (local dev/testing).")

    # Housekeeping
    parser.add_argument("--fresh", action="store_true", help="Delete previous manifest/index before running.")

    args = parser.parse_args()

    # Ensure dirs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        if MANIFEST.exists(): MANIFEST.unlink()
        for p in RAW_DIR.glob("*"): p.unlink(missing_ok=True)
        for p in INDEX_DIR.glob("*"): p.unlink(missing_ok=True)

    # ---- 1) LOCAL FILES (default) ----
    local_paths = [Path(p) for p in args.files]
    for d in args.dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            # pick supported types
            for ext in ("*.pdf","*.txt","*.md","*.html","*.htm"):
                local_paths.extend(p.rglob(ext))
    if local_paths:
        import_local_files(local_paths)

    # ---- 2) OPTIONAL CRAWL (only if --crawl AND seeds) ----
    if args.crawl and args.seeds:
        # Build allowed host set from seeds + any explicit allow list
        hosts = set(args.allow_hosts) if args.allow_hosts else set()
        for s in args.seeds:
            net = urlparse(s).netloc.split(":")[0].lower()
            if net: hosts.add(net)
        print(f"[crawl] hosts allowed: {sorted(hosts)}")
        crawl(
            seeds=args.seeds,
            allowed_hosts=hosts,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            respect_robots=not args.ignore_robots
        )
    else:
        print("Crawling disabled (run with --crawl and one or more --seed URLs to enable).")

    # ---- 3) PARSE → CORPUS → INDEX ----
    corpus_path = parse_and_chunk_all()
    build_index(corpus_path)

    print("\nDone. You can now run:  streamlit run app.py")

if __name__ == "__main__":
    main()
