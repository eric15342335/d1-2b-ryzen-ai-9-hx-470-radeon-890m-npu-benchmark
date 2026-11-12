"""Time llama.cpp's d1 decision endpoint (/v1/systemone) on the Radeon 890M through Vulkan.

Starts llama-server on loopback, sends three warmup requests per fixture, then lets oha send 20
sequential requests while /metrics snapshots bracket the run. Raw request durations land in oha's
SQLite file; the counter deltas give prompt-processing time.
"""

import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from d1bench.common import ROOT, WORKLOADS, redact

PORT = 8093
BASE = f"http://127.0.0.1:{PORT}"
SERVER = ROOT / ".cache/llama-cpp/llama-server.exe"
OHA = ROOT / ".cache/oha.exe"
# No proxy, even if one is configured: every request goes to loopback.
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path: str, timeout: float = 30) -> bytes:
    """GET a server endpoint."""
    with HTTP.open(BASE + path, timeout=timeout) as response:
        body: bytes = response.read()
    return body


def post(path: str, body: bytes) -> bytes:
    """POST JSON to a server endpoint."""
    request = urllib.request.Request(BASE + path, body, {"Content-Type": "application/json"})  # noqa: S310  (fixed http://127.0.0.1)
    with HTTP.open(request, timeout=120) as response:
        answer: bytes = response.read()
    return answer


def wait_ready(server: subprocess.Popen[bytes], seconds: float = 120) -> None:
    """Poll /health until the model is loaded."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            msg = f"llama-server exited with {server.returncode}; see its log"
            raise RuntimeError(msg)
        try:
            get("/health", timeout=2)
        except OSError:
            time.sleep(0.5)
        else:
            return
    msg = f"llama-server not ready after {seconds:.0f} s"
    raise TimeoutError(msg)


def bench_fixture(prefix: str, body: Path, warmup: int, repeats: int) -> None:
    """Warm up, then time `repeats` sequential requests with oha between two /metrics snapshots."""
    for _ in range(warmup):
        Path(f"{prefix}-output.json").write_bytes(post("/v1/systemone", body.read_bytes()))
    Path(f"{prefix}-metrics-before.txt").write_bytes(get("/metrics"))
    Path(f"{prefix}.sqlite").unlink(missing_ok=True)
    oha = [str(OHA), "-n", str(repeats), "-c", "1", "-m", "POST", "-T", "application/json", "-D", str(body), "--no-tui"]
    oha += ["--output-format", "json", "-o", f"{prefix}-oha.json", "--db-url", f"{prefix}.sqlite"]
    no_proxy = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()} | {"NO_PROXY": "127.0.0.1,localhost"}
    subprocess.run([*oha, f"{BASE}/v1/systemone"], check=True, env=no_proxy)  # noqa: S603
    Path(f"{prefix}-metrics-after.txt").write_bytes(get("/metrics"))


def main(output: Path, model: Annotated[Path, typer.Option()], repeats: int = 20, warmup: int = 3) -> None:
    """Benchmark MODEL; OUTPUT is the prefix for every file written (e.g. results/llama-q8)."""
    try:
        get("/health", timeout=2)
    except OSError:
        pass
    else:
        msg = f"something already answers on {BASE}; stop it so the wrong server is not benchmarked"
        raise RuntimeError(msg)
    argv = [str(SERVER), "-m", str(model), "--device", "Vulkan0", "-ngl", "99", "--threads", "4", "--parallel", "1"]
    argv += ["-c", "2048", "-b", "2048", "-ub", "2048", "--metrics", "--host", "127.0.0.1", "--port", str(PORT)]
    with (
        Path(f"{output}-server.log").open("wb") as log,
        subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as server,  # noqa: S603
    ):
        try:
            wait_ready(server)
            Path(f"{output}-props.json").write_text(redact(get("/props").decode("utf-8")), encoding="utf-8")
            for name in WORKLOADS:
                logger.info("running {} on {}", name, model.name)
                bench_fixture(f"{output}-{name}", ROOT / "fixtures" / f"{name}-common.json", warmup, repeats)
        finally:
            server.terminate()
    server_log = Path(f"{output}-server.log")
    server_log.write_text(redact(server_log.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    logger.success("saved {}-*", output)


if __name__ == "__main__":
    typer.run(main)
