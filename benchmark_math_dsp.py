#!/usr/bin/env python3
"""Evo benchmark harness for math-dsp throughput.

Builds and runs the `math_dsp_bench` Rust binary in this worktree, parses the
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
        "status": "passed" if score >= 0.0 else "failed",
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
    parser = argparse.ArgumentParser(description="math-dsp throughput benchmark")
    parser.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="If provided, exit non-zero when the aggregate score is above this threshold (min metric)",
    )
    args = parser.parse_args()

    worktree = Path(__file__).resolve().parent
    repo_root = next(
        p
        for p in worktree.parents
        if (p / ".evo").exists() and (p / "Cargo.toml").exists()
    )
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(worktree / "target")

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "math_dsp_bench",
        "-p",
        "math-dsp",
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

    raw = proc.stdout.strip().splitlines()
    if not raw:
        print("math_dsp_bench produced no output", file=sys.stderr)
        return 1
    try:
        data = json.loads(raw[-1])
    except json.JSONDecodeError as exc:
        print(f"Failed to parse math_dsp_bench output: {exc}\n{proc.stdout}", file=sys.stderr)
        return 1

    for task_id, task_ms in data["tasks"].items():
        log_task(
            task_id,
            float(task_ms),
            direction="min",
            summary=f"{task_ms:.3f} ms (checksum {data['checksums'][task_id]:.6g})",
        )

    score = write_result(float(data["score"]))
    print(f"Aggregate time: {score:.4f} ms", file=sys.stderr)

    if args.max_score is not None and score > args.max_score:
        print(
            f"GATE FAIL: score {score:.4f} above maximum {args.max_score:.4f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
