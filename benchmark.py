#!/usr/bin/env python3
"""Evo benchmark harness for math-iir-fir filter throughput.

Builds and runs the `filter_bench` Rust binary in this worktree, parses the
JSON output, and writes the evo result file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Inline evo instrumentation (from references/inline_instrumentation.py)
# ---------------------------------------------------------------------------
_TRACES_DIR = Path(os.environ["EVO_TRACES_DIR"]) if os.environ.get("EVO_TRACES_DIR") else None
_EXPERIMENT_ID = os.environ.get("EVO_EXPERIMENT_ID", "unknown")
_RESULT_PATH = os.environ.get("EVO_RESULT_PATH")
_SCORES: dict[str, float] = {}
_TASK_META: dict[str, dict[str, Any]] = {}
_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")

if _TRACES_DIR:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)


def log_task(
    task_id: str,
    score: float,
    *,
    summary: str | None = None,
    failure_reason: str | None = None,
    log: list[Any] | None = None,
    direction: str | None = None,
    **extra: Any,
) -> None:
    task_id = str(task_id)
    if direction is not None and direction not in ("max", "min"):
        raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
    _SCORES[task_id] = score
    if direction is not None:
        _TASK_META[task_id] = {"direction": direction}
    if _TRACES_DIR is None:
        return
    trace: dict[str, Any] = {
        "experiment_id": _EXPERIMENT_ID,
        "task_id": task_id,
        "status": "passed" if score >= 0.5 else "failed",
        "score": score,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if direction is not None:
        trace["direction"] = direction
    if summary is not None:
        trace["summary"] = summary
    if failure_reason is not None:
        trace["failure_reason"] = failure_reason
    if log is not None:
        trace["log"] = log
    trace.update(extra)
    (_TRACES_DIR / f"task_{task_id}.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )


def write_result(score: float | None = None) -> float:
    if score is None:
        score = sum(_SCORES.values()) / len(_SCORES) if _SCORES else 0.0
    score = round(score, 4)
    result = {
        "score": score,
        "tasks": dict(_SCORES),
        "started_at": _STARTED_AT,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if _TASK_META:
        result["tasks_meta"] = {k: dict(v) for k, v in _TASK_META.items()}
    payload = json.dumps(result, indent=2)
    if _RESULT_PATH:
        target = Path(_RESULT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise RuntimeError(
                f"{target} already exists; only one write_result() per attempt"
            ) from None
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    else:
        print(payload)
    return score


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Filter throughput benchmark")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="If provided, exit non-zero when the aggregate score is below this threshold",
    )
    args = parser.parse_args()

    worktree = Path(__file__).resolve().parent
    repo_root = Path(
        subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(repo_root / "target")

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "filter_bench",
        "-p",
        "math-iir-fir",
        "--quiet",
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    # Parse the last non-empty JSON line from stdout.
    raw = proc.stdout.strip().splitlines()
    if not raw:
        print("filter_bench produced no output", file=sys.stderr)
        return 1
    try:
        data = json.loads(raw[-1])
    except json.JSONDecodeError as exc:
        print(f"Failed to parse filter_bench output: {exc}\n{proc.stdout}", file=sys.stderr)
        return 1

    iir_throughput = float(data["iir_biquad_block_msamples_per_s"])
    fir_throughput = float(data["fir_block_msamples_per_s"])
    aggregate = (iir_throughput + fir_throughput) / 2.0

    log_task(
        "iir_biquad_block",
        iir_throughput,
        direction="max",
        summary=f"{iir_throughput:.2f} Msamples/s (checksum {data['iir_checksum']:.6g})",
    )
    log_task(
        "fir_block",
        fir_throughput,
        direction="max",
        summary=f"{fir_throughput:.2f} Msamples/s (checksum {data['fir_checksum']:.6g})",
    )

    score = write_result(aggregate)
    print(f"Aggregate throughput: {score:.2f} Msamples/s", file=sys.stderr)

    if args.min_score is not None and score < args.min_score:
        print(
            f"GATE FAIL: score {score:.4f} below minimum {args.min_score:.4f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
