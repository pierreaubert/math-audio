#!/usr/bin/env python3
"""Prove the prepared FIR contract catches two historical AVX2 defects."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("registry.json")
FIR = Path("crates/math-iir-fir/src/fir.rs")
TEST = "prepared_fir_and_filtfilt_contracts"
BEHAVIOR_ASSERTION = "assertion failed: (actual.to_f64().unwrap() - expected).abs() < tolerance"


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="x86_64 Rust target (e.g. for macOS Rosetta)")
    parser.add_argument("--target-dir", help="optional isolated Cargo build cache")
    args = parser.parse_args()
    if not ((args.target and args.target.startswith("x86_64-")) or
            (not args.target and platform.machine() in ("x86_64", "AMD64"))):
        parser.error("requires an x86_64 execution target with AVX2 and FMA")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(FIR)], cwd=ROOT, text=True)
    if dirty:
        parser.error("commit or restore FIR source before running against HEAD")
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-feature=+avx2,+fma"
    command = ["cargo", "test", "-p", "math-iir-fir", "--test", "prepared_processing", TEST]
    if args.target:
        command.extend(["--target", args.target])
    with tempfile.TemporaryDirectory(prefix="fir-negative-") as directory:
        temp = Path(directory)
        archive = temp / "source.tar"
        run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=ROOT)
        run(["tar", "-xf", str(archive), "-C", str(temp)])
        contract = Path("crates/math-iir-fir/tests/prepared_processing.rs")
        shutil.copyfile(ROOT / contract, temp / contract)
        source = temp / FIR
        original = source.read_bytes()
        env["CARGO_TARGET_DIR"] = str(Path(args.target_dir).resolve() if args.target_dir else temp / "target")

        def test():
            clean = ["cargo", "clean", "-p", "math-iir-fir"]
            if args.target:
                clean.extend(["--target", args.target])
            run(clean, cwd=temp, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return subprocess.run(command, cwd=temp, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)

        print("running baseline in isolated checkout", flush=True)
        baseline = test()
        if baseline.returncode != 0 or f"test {TEST} ... ok" not in baseline.stdout:
            raise RuntimeError(f"baseline failed or did not execute:\n{baseline.stdout}")
        print("baseline: passed", flush=True)
        signature = b"unsafe fn process_block_symmetric_f64_avx2(&mut self, samples: &mut [T]) {"
        if original.count(signature) != 1:
            raise RuntimeError("cannot locate AVX2 kernel for execution probe")
        source.write_bytes(original.replace(signature, signature + b'\n        panic!("AVX2_KERNEL_REACHED");'))
        probe = test()
        if (probe.returncode == 0 or f"test {TEST} ... FAILED" not in probe.stdout
                or "AVX2_KERNEL_REACHED" not in probe.stdout):
            raise RuntimeError(f"owning test did not execute AVX2 kernel:\n{probe.stdout}")
        print("AVX2 kernel execution: confirmed", flush=True)
        for mutant in json.loads(REGISTRY.read_text()):
            source.write_bytes(original)
            run(["git", "apply", "--reverse", str(REGISTRY.parent / mutant["patch"])], cwd=temp)
            result = test()
            if (result.returncode == 0 or f"test {TEST} ... FAILED" not in result.stdout
                    or BEHAVIOR_ASSERTION not in result.stdout):
                raise RuntimeError(f"{mutant['fix']}: expected behavior failure, got:\n{result.stdout}")
            print(f"{mutant['fix']}: detected by {TEST}", flush=True)
        source.write_bytes(original)


if __name__ == "__main__":
    main()
