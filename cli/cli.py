from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

REQUIRED_DATASET_FILES = ("corpus.jsonl", "queries.jsonl", "qrels_debug.jsonl", "qrels/test.tsv")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def looks_like_dataset_dir(path: Path) -> bool:
    return all((path / rel).exists() for rel in REQUIRED_DATASET_FILES)


def find_dataset_candidates(base_dir: Path) -> list[Path]:
    out: list[Path] = []
    if looks_like_dataset_dir(base_dir):
        out.append(base_dir.resolve())
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and looks_like_dataset_dir(child):
            out.append(child.resolve())
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def choose_dataset(console: Console, base_dir: Path) -> Path:
    candidates = find_dataset_candidates(base_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No dataset directories found in {base_dir}. Expected: {', '.join(REQUIRED_DATASET_FILES)}"
        )
    if len(candidates) == 1:
        return candidates[0]

    table = Table(title="Choose Dataset", box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Path")
    for i, path in enumerate(candidates, start=1):
        table.add_row(str(i), str(path))
    console.print(table)

    while True:
        raw = Prompt.ask("Dataset index", default="1").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        console.print(f"[red]Choose a value between 1 and {len(candidates)}[/red]")


def load_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    missing = [rel for rel in REQUIRED_DATASET_FILES if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(str(root / rel) for rel in missing))

    corpus = {str(row["_id"]): row for row in read_jsonl(root / "corpus.jsonl")}
    queries = {str(row["_id"]): str(row.get("text", "")) for row in read_jsonl(root / "queries.jsonl")}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (root / "qrels" / "test.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    debug_rows = read_jsonl(root / "qrels_debug.jsonl")
    debug_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in debug_rows:
        debug_by_query[str(row.get("query_id", ""))].append(row)

    return {
        "root": root,
        "corpus": corpus,
        "queries": queries,
        "qrels": dict(qrels),
        "debug_by_query": dict(debug_by_query),
        "debug_rows": debug_rows,
    }


def first_sentence(text: str, width: int = 100) -> str:
    return textwrap.shorten(re.sub(r"\s+", " ", text).strip(), width=width, placeholder="...")


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["pbcopy"], ["clip"]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            return True, cmd[0]
        except Exception:
            continue
    return False, "no clipboard backend"


def copy_payload(console: Console, payload: dict[str, Any], label: str) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    ok, backend = copy_to_clipboard(text)
    if ok:
        console.print(f"[green]Copied {label} via {backend}[/green]")
        return
    out = Path.cwd() / f"copied_{label}_{payload.get('query_id', 'unknown')}.json"
    out.write_text(text, encoding="utf-8")
    console.print(f"[yellow]Clipboard unavailable ({backend}); wrote {out}[/yellow]")


def show_summary(console: Console, ds: dict[str, Any]) -> None:
    qrel_count = sum(len(rows) for rows in ds["qrels"].values())
    debug_scores = Counter(int(row.get("score", 0)) for row in ds["debug_rows"])
    qrel_scores = Counter(score for rows in ds["qrels"].values() for score in rows.values())

    table = Table(title="Dataset Summary", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Dataset", str(ds["root"]))
    table.add_row("Corpus", f"{len(ds['corpus']):,}")
    table.add_row("Queries", f"{len(ds['queries']):,}")
    table.add_row("Qrels", f"{qrel_count:,}")
    table.add_row("Debug rows", f"{len(ds['debug_rows']):,}")
    table.add_row("Qrels scores", ", ".join(f"{k}:{v}" for k, v in sorted(qrel_scores.items())) or "-")
    table.add_row("Debug scores", ", ".join(f"{k}:{v}" for k, v in sorted(debug_scores.items())) or "-")
    console.print(table)


def show_queries(console: Console, ds: dict[str, Any], needle: str = "") -> None:
    needle = needle.lower().strip()
    rows = [
        (qid, text)
        for qid, text in ds["queries"].items()
        if not needle or needle in qid.lower() or needle in text.lower()
    ]
    rows = rows[:100]

    table = Table(title=f"Queries ({len(rows)} shown)", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Qrels", justify="right")
    table.add_column("Debug", justify="right")
    table.add_column("Query")
    for qid, text in rows:
        table.add_row(qid, str(len(ds["qrels"].get(qid, {}))), str(len(ds["debug_by_query"].get(qid, []))), first_sentence(text, 90))
    console.print(table)


def show_query(console: Console, ds: dict[str, Any], qid: str) -> None:
    if qid not in ds["queries"]:
        console.print(f"[red]Unknown query id:[/red] {qid}")
        return

    console.print(Panel(ds["queries"][qid], title=f"Query {qid}", border_style="cyan"))
    debug_rows = sorted(ds["debug_by_query"].get(qid, []), key=lambda r: int(r.get("score", 0)), reverse=True)
    qrels = ds["qrels"].get(qid, {})

    meta = Table(title="Query Debug Metadata", box=box.ROUNDED, style="magenta",)
    meta.add_column("Field", style="bold")
    meta.add_column("Value")
    first = debug_rows[0] if debug_rows else {}
    meta.add_row("Query source", str(first.get("query_source") or "-"))
    meta.add_row("Query parent id", str(first.get("query_parent_id") or "-"))
    meta.add_row("Score histogram", json.dumps(first.get("score_histogram", {}), ensure_ascii=False))
    console.print(meta)

    docs_payload = build_documents_payload(ds, qid)["documents"]

    explanation_table = Table(title="Validation Explanations (query)", box=box.ROUNDED, style="magenta",)
    explanation_table.add_column("Query Validation", overflow="fold", ratio=2)
    explanation_table.add_column("Distribution Validation", overflow="fold", ratio=2)
    for doc in docs_payload[:20]:
        explanation_table.add_row(
            str(doc.get("query_validation_explanation") or "-"),
            str(doc.get("distribution_validation_explanation") or "-"),
        
        )
        break
    console.print(explanation_table)

    table = Table(title="Relevant Corpus Judgments", box=box.ROUNDED, style="magenta", show_lines=True)
    table.add_column("Corpus", style="cyan", overflow="ellipsis", max_width=20)
    table.add_column("Score", justify="right")
    table.add_column("BM25", justify="right")
    table.add_column("Rerank", justify="right")
    table.add_column("RRF", justify="right")
    for row in debug_rows:
        table.add_row(
            str(row.get("corpus_id", "")),
            str(row.get("score", "-")),
            f"{float(row['bm25_score']):.3f}" if row.get("bm25_score") is not None else "-",
            f"{float(row['rerank_score']):.3f}" if row.get("rerank_score") is not None else "-",
            f"{float(row['rrf_score']):.3f}" if row.get("rrf_score") is not None else "-",
        )
    seen = {str(row.get("corpus_id", "")) for row in debug_rows}
    for corpus_id, score in sorted(qrels.items(), key=lambda x: x[1], reverse=True):
        if corpus_id not in seen:
            table.add_row(corpus_id, str(score), "-", "-", "-", "qrels only")
    console.print(table)

    docs_table = Table(title="Documents + Explanations", box=box.ROUNDED, style="magenta", show_lines=True)
    docs_table.add_column("Corpus", style="cyan", overflow="ellipsis", max_width=20)
    docs_table.add_column("Label", justify="right")
    docs_table.add_column("Dbg", justify="right")
    docs_table.add_column("Judge Explanation", overflow="fold", ratio=2)
    docs_table.add_column("Doc Snippet", overflow="fold", ratio=2)
    for doc in docs_payload[:40]:
        docs_table.add_row(
            str(doc.get("corpus_id", "")),
            str(doc.get("qrels_score", "-")),
            str(doc.get("debug_score", "-")),
            str(doc.get("judge_reason") or "-"),
            first_sentence(str(doc.get("text") or ""), 110),
        )
    console.print(docs_table)

    full_doc_table = Table(title="Full Documents", box=box.ROUNDED, style="magenta", show_lines=True)
    full_doc_table.add_column("Corpus", style="cyan", overflow="ellipsis", max_width=20)
    full_doc_table.add_column("Text")

    for doc in docs_payload[:40]:
        full_doc_table.add_row(
            str(doc.get("corpus_id", "")),
            str(doc.get("text") or ""),
        )
    console.print(full_doc_table)


def build_documents_payload(ds: dict[str, Any], qid: str) -> dict[str, Any]:
    debug_rows = sorted(ds["debug_by_query"].get(qid, []), key=lambda r: int(r.get("score", 0)), reverse=True)
    qrels = ds["qrels"].get(qid, {})
    debug_by_corpus = {str(row.get("corpus_id", "")): row for row in debug_rows}

    related_ids: list[str] = []
    for row in debug_rows:
        cid = str(row.get("corpus_id", ""))
        if cid and cid not in related_ids:
            related_ids.append(cid)
    for cid, _ in sorted(qrels.items(), key=lambda x: x[1], reverse=True):
        if cid not in related_ids:
            related_ids.append(cid)

    docs = []
    for cid in related_ids:
        corpus = ds["corpus"].get(cid)
        debug = debug_by_corpus.get(cid)
        docs.append(
            {
                "corpus_id": cid,
                "qrels_score": qrels.get(cid),
                "debug_score": debug.get("score") if debug else None,
                "judge_reason": debug.get("judge_reason") if debug else None,
                "query_validation_explanation": ((debug or {}).get("query_validation") or {}).get("explanation"),
                "distribution_validation_explanation": ((debug or {}).get("distribution_validation") or {}).get("explanation"),
                "title": (corpus or {}).get("title"),
                "text": (corpus or {}).get("text"),
                "present_in_corpus": corpus is not None,
            }
        )
    return {"query_id": qid, "query": ds["queries"].get(qid, ""), "documents": docs}


def build_labels_payload(ds: dict[str, Any], qid: str) -> dict[str, Any]:
    labels = [{"query_id": qid, "corpus_id": cid, "score": score} for cid, score in sorted(ds["qrels"].get(qid, {}).items())]
    return {"query_id": qid, "labels": labels}


def build_query_bundle_payload(ds: dict[str, Any], qid: str) -> dict[str, Any]:
    first = (ds["debug_by_query"].get(qid) or [{}])[0]
    docs = build_documents_payload(ds, qid)["documents"]
    labels = build_labels_payload(ds, qid)["labels"]
    return {
        "query_id": qid,
        "query": ds["queries"].get(qid, ""),
        "query_source": first.get("query_source"),
        "query_parent_id": first.get("query_parent_id"),
        "score_histogram": first.get("score_histogram"),
        "query_validation": first.get("query_validation"),
        "distribution_validation": first.get("distribution_validation"),
        "labels": labels,
        "documents": docs,
    }


def run_repl(console: Console, ds: dict[str, Any]) -> None:
    console.print(
        Panel.fit(
            "[bold]BEIR Analyzer[/bold]\n"
            "Commands: help, summary, queries [text], query <qid>, copy-query [qid], quit",
            border_style="cyan",
        )
    )
    selected_qid: str | None = None

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]compact[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not raw:
            continue

        cmd, _, arg = raw.partition(" ")
        cmd = cmd.lower().strip()
        arg = arg.strip()

        if cmd in {"q", "quit", "exit"}:
            return
        if cmd in {"h", "help", "?"}:
            console.print("summary | queries [text] | query <qid> | copy-query [qid] | quit")
        elif cmd == "summary":
            show_summary(console, ds)
        elif cmd == "queries":
            show_queries(console, ds, arg)
        elif cmd == "query":
            if not arg:
                console.print("[red]Usage:[/red] query <qid>")
            elif arg not in ds["queries"]:
                console.print(f"[red]Unknown query id:[/red] {arg}")
            else:
                selected_qid = arg
                show_query(console, ds, arg)
        elif cmd in {"copy-query", "cq"}:
            qid = arg or selected_qid
            if not qid:
                console.print("[red]Usage:[/red] copy-query <qid> (or run query <qid> first)")
            elif qid not in ds["queries"]:
                console.print(f"[red]Unknown query id:[/red] {qid}")
            else:
                copy_payload(console, build_query_bundle_payload(ds, qid), "query_bundle")
        else:
            console.print(f"[red]Unknown command:[/red] {cmd}. Type [cyan]help[/cyan].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEIR analyzer")
    parser.add_argument("path", nargs="?", type=Path, help="Dataset dir for one-shot mode, or base dir for interactive picker.")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit.")
    parser.add_argument("--query", help="Show a single query and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    interactive_mode = not (args.summary or args.query)
    path_arg = args.path

    try:
        if interactive_mode:
            base_dir = path_arg if path_arg is not None else Path.cwd()
            dataset_dir = choose_dataset(console, base_dir)
        else:
            if path_arg is None:
                raise FileNotFoundError("Provide dataset directory for one-shot mode.")
            dataset_dir = path_arg

        ds = load_dataset(dataset_dir)
    except Exception as exc:
        console.print(f"[red]Could not load dataset:[/red] {exc}")
        raise SystemExit(1) from exc

    if args.summary:
        show_summary(console, ds)
    elif args.query:
        show_query(console, ds, args.query)
    else:
        run_repl(console, ds)


if __name__ == "__main__":
    main()
