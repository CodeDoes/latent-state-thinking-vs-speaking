#!/usr/bin/env python3
"""Query the arXiv API for papers.

Hits the public export.arxiv.org endpoint with `search_query`, returns
parsed entries with title/abstract/authors/pdf-link.

Usage:
    python ./src/arxiv_query.py "<query>"
    python ./src/arxiv_query.py -n 10 "all:state+space"
    python ./src/arxiv_query.py -cat cs.LG "all:adapter+memory"
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ENDPOINT = "http://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def parse_feed(xml_text):
    root = ET.fromstring(xml_text)
    total = root.find("atom:opensearch:totalResults", NS)
    total = int(total.text) if total is not None else 0

    out = []
    for entry in root.findall("atom:entry", NS):
        e = {}
        e["id"] = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
        e["title"] = re_clean(entry.findtext("atom:title", default="", namespaces=NS))
        e["summary"] = re_clean(entry.findtext("atom:summary", default="", namespaces=NS))
        e["published"] = entry.findtext("atom:published", default="", namespaces=NS) or ""

        # Find PDF link
        for link in entry.findall("atom:link", NS):
            href = link.attrib.get("href", "")
            if link.attrib.get("title", "") == "pdf" or href.endswith(".pdf"):
                e["pdf"] = href
                break

        authors = []
        for a in entry.findall("atom:author", NS):
            name = a.findtext("atom:name", default="", namespaces=NS) or ""
            if name:
                authors.append(name)
        e["authors"] = authors

        cats = []
        for c in entry.findall("atom:category", NS):
            t = c.attrib.get("term")
            if t:
                cats.append(t)
        e["categories"] = cats

        prim = entry.find("arxiv:primary_category", NS)
        if prim is not None:
            e["primary_category"] = prim.attrib.get("term", "")

        comm = entry.findtext("arxiv:comment", default="", namespaces=NS)
        if comm:
            e["comment"] = re_clean(comm)

        out.append(e)
    return total, out


def re_clean(text):
    if text is None:
        return ""
    return " ".join(text.split())


def query(q, max_results=10, wait_seconds=3.1):
    """Returns (total, results). Be polite: 1 request per 3 sec."""
    params = {
        "search_query": q,
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"fetch failed: {e}", file=sys.stderr)
        return 0, []

    total, results = parse_feed(xml_text)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    return total, results


def main():
    ap = argparse.ArgumentParser(description="Query the arXiv API.")
    ap.add_argument("query", help="search_query string (arXiv syntax)")
    ap.add_argument("-n", "--max", type=int, default=10, help="max results")
    ap.add_argument("-o", "--out", help="write JSON file (else print summary)")
    args = ap.parse_args()

    total, results = query(args.query, args.max)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "query": args.query, "total": total, "results": results
        }, indent=2))
        print(f"wrote {args.out}")
    else:
        print(f"Total matching: {total}")
        print(f"Showing {len(results)}:")
        for i, e in enumerate(results, 1):
            print()
            print(f"{i}. {e['title']}")
            print(f"   ID: {e['id']}")
            print(f"   Authors: {', '.join(e['authors'][:4])}{' et al.' if len(e['authors']) > 4 else ''}")
            print(f"   Primary: {e.get('primary_category', '')}")
            print(f"   Published: {e.get('published', '')[:10]}")
            if e.get("pdf"):
                print(f"   PDF: {e['pdf']}")
            sm = e["summary"]
            if len(sm) > 400:
                sm = sm[:400] + "..."
            print(f"   > {sm}")


if __name__ == "__main__":
    main()
