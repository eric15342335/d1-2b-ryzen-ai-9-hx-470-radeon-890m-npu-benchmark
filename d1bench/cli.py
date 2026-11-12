"""d1bench: list, check, run and report the d1-2B CPU / iGPU / NPU benchmarks.

Data goes to stdout (Rich tables, or JSON with --json); progress logs go to stderr.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from d1bench.common import RESULTS, ROOT, Benchmark, git_head, load_catalog, redact, write_json
from d1bench.evaluate import evaluate
from d1bench.report import render
from d1bench.summarize import summarize

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
out = Console()
JsonFlag = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")]
RUNNERS = {
    "onnx": "d1bench.run_onnx",
    "npu": "d1bench.run_npu",
    "llama": "d1bench.run_llama",
    "reference": "d1bench.run_reference",
}


def command(bench: Benchmark, output: Path, extra: list[str]) -> tuple[list[str], dict[str, str]]:
    """Exact argv and environment additions for one benchmark."""
    # Relative to ROOT (the child's cwd), so logs and result files hold no machine-specific paths.
    argv = [sys.executable, "-m", RUNNERS[bench.kind], str(output.relative_to(ROOT)), *bench.args, *extra]
    return argv, {"PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in bench.pythonpath)}


def stale_pins(bench: Benchmark) -> dict[str, str]:
    """Pinned checkouts not at their reviewed commit, with the commit they are at."""
    heads = {path: git_head(ROOT / path) if (ROOT / path / ".git").exists() else "missing" for path in bench.pins}
    return {path: head for path, head in heads.items() if head != bench.pins[path]}


def mtime(path: Path) -> str | None:
    """Last-modified time in ISO form, or None if missing."""
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds") if path.exists() else None


def pick(ids: list[str]) -> list[Benchmark]:
    """Resolve ids; `all` means every benchmark expected to succeed."""
    catalog = load_catalog()
    if ids == ["all"]:
        return [b for b in catalog.values() if b.expect == "ok"]
    if unknown := [i for i in ids if i not in catalog]:
        msg = f"unknown id {unknown}; see `d1bench list`"
        raise typer.BadParameter(msg)
    return [catalog[i] for i in ids]


@app.command("list")
def list_(as_json: JsonFlag = False) -> None:
    """Show every benchmark in benchmarks.toml and whether its result exists."""
    catalog = load_catalog()
    if as_json:
        out.print_json(data=[{**vars(b), "result_updated": mtime(b.result_path)} for b in catalog.values()])
        return
    table = Table("id", "kind", "label", "expect", "result updated (UTC)")
    for b in catalog.values():
        table.add_row(b.id, b.kind, b.label, b.expect, mtime(b.result_path) or "[red]missing")
    out.print(table)


@app.command()
def show(bench_id: str, as_json: JsonFlag = False) -> None:
    """Print the exact command `d1bench run BENCH_ID` executes."""
    bench = pick([bench_id])[0]
    argv, env = command(bench, RESULTS / bench.result, [])
    if as_json:
        out.print_json(data={**vars(bench), "argv": argv, "env": env})
        return
    out.print(f"[bold]{bench.label}[/] ({bench.kind}, expect {bench.expect}): {bench.note}")
    out.print(f"model {bench.model}; pins {bench.pins or 'none'}")
    out.print(f"$env:PYTHONPATH = '{env['PYTHONPATH']}'", markup=False, soft_wrap=True)
    out.print(subprocess.list2cmdline(argv), markup=False, soft_wrap=True)


def verify_tomoali() -> list[str]:
    """Files of the Tomoali NPU model whose size or SHA-256 differs from the publisher manifest."""
    model = ROOT / "models/d1-2b-ryzen-ai-npu"
    bad = []
    for item in json.loads((model / "model-manifest.json").read_text(encoding="utf-8"))["files"]:
        path = model / item["path"]
        with path.open("rb") as stream:
            if (
                path.stat().st_size != item["bytes"]
                or hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]
            ):
                bad.append(item["path"])
    return bad


@app.command()
def status(
    as_json: JsonFlag = False,
    verify: Annotated[bool, typer.Option(help="Also hash the Tomoali model files (slow)")] = False,
) -> None:
    """Check prerequisites and upstream pins; exit 1 if anything is missing or moved."""
    paths = [
        ".venv",
        ".cache/ort-modern",
        ".cache/webgpu-plugin",
        ".cache/liquidonnx",
        ".cache/tomoali-runtime",
        ".cache/tomoali-sdk/deployment",
        ".cache/llama-cpp/llama-server.exe",
        ".cache/oha.exe",
        "models/d1-2B-ONNX",
        "models/d1-2b-ryzen-ai-npu",
        "models/d1-2B-GGUF",
        "models/d1-2B-reference",
    ]
    problems: dict[str, Any] = {
        "missing": [p for p in paths if not (ROOT / p).exists()],
        "stale_pins": {b.id: s for b in load_catalog().values() if (s := stale_pins(b))},
    }
    if verify:
        problems["tomoali_checksum_mismatches"] = verify_tomoali()
    ok = not any(problems.values())
    if as_json:
        out.print_json(data={"ok": ok, **problems})
    else:
        out.print("[green]ready[/]" if ok else problems)
    raise typer.Exit(0 if ok else 1)


def profilable(benches: list[Benchmark]) -> list[Benchmark]:
    """Keep the benchmarks that support --profile (ONNX and NPU), warning about the rest."""
    if skipped := [b.id for b in benches if b.kind not in {"onnx", "npu"}]:
        logger.warning("--profile applies only to ONNX and NPU benchmarks; skipping {}", skipped)
    return [b for b in benches if b.kind in {"onnx", "npu"}]


def tee(argv: list[str], env: dict[str, str], log: Path) -> int:
    """Run a child process, copying its combined output to stderr and LOG; return its exit code."""
    with (
        subprocess.Popen(  # noqa: S603
            argv, cwd=ROOT, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as child,
        log.open("wb") as sink,
    ):
        assert child.stdout is not None
        for line in child.stdout:
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
            sink.write(redact(line.decode("utf-8", "surrogateescape")).encode("utf-8", "surrogateescape"))
    return child.returncode


@app.command()
def run(
    bench_ids: Annotated[list[str], typer.Argument(help="Benchmark ids, or `all`")],
    profile: Annotated[bool, typer.Option(help="ONNX/NPU: one profiled call per input instead of timing")] = False,
    as_json: JsonFlag = False,
) -> None:
    """Run benchmarks one at a time, each in its own process; output is teed to results/<id>.log."""
    summary = []
    benches, suffix, extra = (
        (profilable(pick(bench_ids)), "-profile", ["--profile"]) if profile else (pick(bench_ids), "", [])
    )
    for bench in benches:
        if stale := stale_pins(bench):
            logger.error("{}: pinned checkouts moved: {}", bench.id, stale)
            raise typer.Exit(1)
        output = RESULTS / bench.result
        output = output.with_stem(output.stem + suffix)
        argv, env = command(bench, output, extra)
        log = RESULTS / f"{bench.id}{suffix}.log"
        logger.info("{}: {}", bench.id, subprocess.list2cmdline(argv))
        code = tee(argv, env, log)
        summary.append(
            {
                "id": bench.id,
                "returncode": code,
                "expected_failure": bench.expect == "fails",
                "log": str(log.relative_to(ROOT)),
                "result": str(output.relative_to(ROOT)),
            }
        )
        if code and bench.expect == "ok":
            logger.error("{} failed with exit code {}; see {}", bench.id, code, log)
    if as_json:
        out.print_json(data=summary)
    if any(s["returncode"] and not s["expected_failure"] for s in summary):
        raise typer.Exit(1)


@app.command()
def report(as_json: JsonFlag = False) -> None:
    """Recompute stats and reference agreement from raw results; write RESULTS.md and two JSON files."""
    catalog = load_catalog()
    measured = [b for b in catalog.values() if b.expect == "ok" and b.result_path.exists()]
    rows = summarize(measured)
    records = evaluate(measured, catalog["reference"])
    write_json(RESULTS / "comparison.json", {"percentile": "nearest rank", "rows": rows})
    write_json(RESULTS / "correctness.json", {"reference": catalog["reference"].model, "records": records})
    (ROOT / "RESULTS.md").write_text(render(catalog, rows, records), encoding="utf-8")
    logger.success("wrote RESULTS.md, results/comparison.json, results/correctness.json")
    if as_json:
        out.print_json(data=[{k: v for k, v in r.items() if k != "metrics_deltas"} for r in rows])
        return
    table = Table("configuration", "input", "mean ms", "median ms", "p95 ms", "req/s")
    for r in rows:
        table.add_row(
            r["backend"],
            r["workload"],
            *(f"{r[k]:,.2f}" for k in ("mean_ms", "median_ms", "p95_ms", "states_per_second")),
        )
    out.print(table)


def main() -> None:
    """Console entry point."""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</> <level>{level: <7}</> {message}")
    app()
