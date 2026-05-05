#!/usr/bin/env python3
"""Eval runner — run the dataset against a live backend and report pass/fail.

Usage
-----
    python scripts/run_eval.py [--backend-url URL] [--dataset PATH] [--langsmith]

Defaults
    --backend-url   http://127.0.0.1:8000
    --dataset       eval/dataset.json

Checks per case
    1. HTTP 200 status
    2. Response has a non-empty answer
    3. At least expected_min_citations citations are returned
    4. At least one expected_topic appears in the answer or citation text

LangSmith upload (--langsmith)
    Requires LANGSMITH_API_KEY env var.  Creates (or updates) a dataset named
    after dataset["name"] and adds example outputs from this run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DATASET = _REPO_ROOT / "eval" / "dataset.json"


def _call_ask(backend_url: str, video_id: str, question: str) -> dict:
    """POST /ask and return the parsed JSON body, or a synthetic error dict."""
    import urllib.request
    import urllib.error

    payload = json.dumps({"video_id": video_id, "question": question, "k": 5}).encode()
    req = urllib.request.Request(
        f"{backend_url}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {}
        return {"_http_error": exc.code, "error": body.get("error", {})}
    except Exception as exc:
        return {"_connection_error": str(exc)}


def _check_case(case: dict, response: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []

    if "_http_error" in response or "_connection_error" in response:
        failures.append(f"request failed: {response}")
        return False, failures

    answer = response.get("answer", "")
    citations = response.get("citations", [])

    if not answer:
        failures.append("answer is empty")

    min_cit = case.get("expected_min_citations", 1)
    if len(citations) < min_cit:
        failures.append(f"expected >= {min_cit} citations, got {len(citations)}")

    # At least one expected topic must appear somewhere in answer + citation texts
    topics = [t.lower() for t in case.get("expected_topics", [])]
    haystack = answer.lower() + " " + " ".join(c.get("text", "").lower() for c in citations)
    matched = [t for t in topics if t in haystack]
    if topics and not matched:
        failures.append(f"none of the expected topics found: {topics}")

    return len(failures) == 0, failures


def _upload_to_langsmith(dataset_meta: dict, results: list[dict]) -> None:
    """Push results to a LangSmith dataset (best-effort)."""
    try:
        from langsmith import Client

        client = Client()
        ds_name = dataset_meta.get("name", "eval-dataset")
        try:
            ds = client.read_dataset(dataset_name=ds_name)
        except Exception:
            ds = client.create_dataset(dataset_name=ds_name, description=dataset_meta.get("description", ""))

        for r in results:
            try:
                client.create_example(
                    inputs={"question": r["question"], "video_id": r["video_id"]},
                    outputs={
                        "answer": r.get("answer", ""),
                        "citations": r.get("citations", []),
                        "passed": r["passed"],
                    },
                    dataset_id=ds.id,
                )
            except Exception:
                pass
        print(f"[langsmith] uploaded {len(results)} examples to dataset '{ds_name}'")
    except Exception as exc:
        print(f"[langsmith] upload failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval dataset against the backend.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000", metavar="URL")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET), metavar="PATH")
    parser.add_argument("--langsmith", action="store_true", help="Upload results to LangSmith")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    dataset = json.loads(dataset_path.read_text())
    cases = dataset["cases"]
    video_id = dataset["video_id"]

    print(f"Backend : {args.backend_url}")
    print(f"Dataset : {dataset_path.name}  ({len(cases)} cases)")
    print(f"Video   : {video_id}")
    print()

    col_w = max(len(c["id"]) for c in cases) + 2
    results: list[dict] = []
    passed = 0

    for case in cases:
        t0 = time.monotonic()
        response = _call_ask(args.backend_url, video_id, case["question"])
        elapsed = time.monotonic() - t0
        ok, failures = _check_case(case, response)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {case['id']:<{col_w}}  {elapsed:.2f}s  {case['question'][:60]}")
        if failures:
            for f in failures:
                print(f"          ↳ {f}")
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "video_id": video_id,
                "passed": ok,
                "failures": failures,
                "latency_s": round(elapsed, 3),
                "answer": response.get("answer", ""),
                "citations": response.get("citations", []),
            }
        )

    total = len(cases)
    print()
    print(f"Results: {passed}/{total} passed  ({'%.0f' % (passed / total * 100)}%)")

    if args.langsmith:
        _upload_to_langsmith(dataset, results)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
