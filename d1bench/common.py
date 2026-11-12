"""Paths, the benchmark catalog, fixtures and the timing loop shared by every runner."""

import json
import re
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WORKLOADS = ("ticket", "long-ticket")


def _path_pattern(path: Path) -> re.Pattern[str]:
    """Match `path` written with backslashes, JSON-escaped backslashes or forward slashes, any case."""
    separator = r"[\\/]{1,2}"
    return re.compile(re.escape(path.drive) + separator + separator.join(map(re.escape, path.parts[1:])), re.IGNORECASE)


# Runtimes print these into logs and profiles; results/ must not record machine-specific paths.
_PRIVATE_PATHS = ((_path_pattern(ROOT), "."), (_path_pattern(Path.home()), "%USERPROFILE%"))


def redact(text: str) -> str:
    """Replace this checkout's absolute path with "." and the home directory with %USERPROFILE%."""
    for pattern, replacement in _PRIVATE_PATHS:
        text = pattern.sub(replacement, text)
    return text


@dataclass(frozen=True)
class Benchmark:
    """One entry of benchmarks.toml."""

    id: str
    kind: str
    label: str
    result: str
    model: str
    expect: str
    note: str
    args: list[str] = field(default_factory=list)
    pythonpath: list[str] = field(default_factory=list)
    pins: dict[str, str] = field(default_factory=dict)

    @property
    def result_path(self) -> Path:
        """The file whose presence means the benchmark has run."""
        name = f"{self.result}-long-ticket.sqlite" if self.kind == "llama" else self.result
        return RESULTS / name


def load_catalog() -> dict[str, Benchmark]:
    """Read benchmarks.toml in file order."""
    with (ROOT / "benchmarks.toml").open("rb") as stream:
        return {key: Benchmark(id=key, **value) for key, value in tomllib.load(stream).items()}


def load_request(workload: str) -> dict[str, Any]:
    """Load a workload's decision request: a ticket state and three questions."""
    request: dict[str, Any] = json.loads((ROOT / "fixtures" / f"{workload}-common.json").read_text(encoding="utf-8"))
    return request


def git_head(checkout: Path) -> str:
    """Commit the checkout is at."""
    return subprocess.check_output(  # noqa: S603
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip()


def measure(call: Callable[[], Any], sink: list[Any], warmup: int, repeats: int) -> dict[str, Any]:
    """Time one first call, `warmup` discarded calls, then `repeats` timed calls.

    `sink` collects per-forward records while `call` runs; each request gets its slice. Every timed
    answer must equal the first, since the decision readout is deterministic.
    """
    start = len(sink)
    started = time.perf_counter()
    first = call()
    row: dict[str, Any] = {"first_ms": (time.perf_counter() - started) * 1000, "first_forwards": sink[start:]}
    for _ in range(warmup):
        call()
    timings, forwards = [], []
    for _ in range(repeats):
        start = len(sink)
        started = time.perf_counter()
        answer = call()
        timings.append((time.perf_counter() - started) * 1000)
        forwards.append(sink[start:])
        if answer != first:
            msg = "repeated deterministic request changed its answer"
            raise RuntimeError(msg)
    return {**row, "timings_ms": timings, "forwards_per_request": forwards, "output": first}


def read_json(name: str) -> Any:  # noqa: ANN401
    """Load a JSON file from results/; PowerShell wrote some with a BOM."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: object) -> None:
    """Write indented UTF-8 JSON with local paths redacted."""
    path.write_text(redact(json.dumps(data, indent=2)) + "\n", encoding="utf-8")
