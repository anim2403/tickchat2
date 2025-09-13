import os
import re
import sys
import math
import time
import json
import argparse
import asyncio
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
import async_timeout
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()


SKIP_EXT = re.compile(r".*\.(png|jpg|jpeg|gif|svg|webp|pdf|zip|tar|gz|css|js)$", re.I)

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "nav", "footer", "noscript"]):
        s.extract()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

def url_hash(u: str) -> str:
    import hashlib
    return hashlib.md5(u.encode("utf-8")).hexdigest()


class Crawler:
    def __init__(self, roots, max_pages, concurrency, rps):
        self.roots = roots
        self.max_pages = max_pages
        self.sem = asyncio.Semaphore(concurrency)
        self.delay = 1.0 / rps if rps > 0 else 0
        self.visited = set()
        self.to_visit = list(roots)
        self.robot_parsers = {}

    async def fetch(self, session, url):
        async with self.sem:
            await asyncio.sleep(self.delay)
            try:
                async with async_timeout.timeout(20):
                    async with session.get(url) as resp:
                        if resp.status == 200 and resp.content_type in ("text/html", "application/xhtml+xml"):
                            return await resp.text()
            except Exception as e:
                print(f"[ERROR] {url} {e}")
        return None

    async def get_robot(self, session, root):
        parsed = urlparse(root)
        base = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if base in self.robot_parsers:
            return self.robot_parsers[base]
        try:
            async with session.get(base) as resp:
                text = await resp.text()
                rp = robotparser.RobotFileParser()
                rp.parse(text.splitlines())
                self.robot_parsers[base] = rp
                return rp
        except:
            return None

    async def crawl(self):
        results = {}
        async with aiohttp.ClientSession() as session:
            while self.to_visit and len(results) < self.max_pages:
                url = self.to_visit.pop(0)
                if url in self.visited or SKIP_EXT.match(url):
                    continue
                self.visited.add(url)

                
                rp = await self.get_robot(session, url)
                if rp and not rp.can_fetch("*", url):
                    continue

                html = await self.fetch(session, url)
                if not html:
                    continue

                text = clean_text(html)
                results[url] = text

               
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    link, _ = urldefrag(link)
                    if any(link.startswith(root) for root in self.roots):
                        if link not in self.visited:
                            self.to_visit.append(link)

        return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--pinecone-index", required=True)
    ap.add_argument("--namespace", default="atlan")
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    
    print(f"[INFO] Loading model {args.embedding_model} ...")
    embedder = SentenceTransformer(args.embedding_model)

    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dim = embedder.get_sentence_embedding_dimension()
    if args.pinecone_index not in pc.list_indexes().names():
        pc.create_index(
            name=args.pinecone_index,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(args.pinecone_index)

    
    crawler = Crawler(args.roots, args.max_pages, args.concurrency, args.rps)
    pages = asyncio.run(crawler.crawl())
    print(f"[INFO] Crawled {len(pages)} pages.")

    
    batch = []
    for url, text in pages.items():
        for chunk in chunk_text(text):
            vec = embedder.encode(chunk).tolist()
            batch.append({
                "id": url_hash(url + chunk),
                "values": vec,
                "metadata": {"url": url, "text": chunk}
            })
            if len(batch) >= 100:
                index.upsert(vectors=batch, namespace=args.namespace)
                batch = []
    if batch:
        index.upsert(vectors=batch, namespace=args.namespace)

    print(f"[INFO] Done. Stored embeddings in Pinecone index '{args.pinecone_index}'.")

if __name__ == "__main__":
    main()
