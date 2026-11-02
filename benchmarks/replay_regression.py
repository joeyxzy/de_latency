"""Replay a historical log and compare attribution to a saved baseline."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from de_latency.analysis.pipeline import process_logs
from de_latency.export.trace_io import artifact_stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("baseline", type=Path, help="baseline trace.compute_breakdown.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stem = artifact_stem(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(stem) + ".analysis.txt").open("w") as stream, contextlib.redirect_stdout(stream):
        result = process_logs(str(args.log), str(args.output))
    old = json.loads(args.baseline.read_text())
    new = json.loads(Path(str(stem) + ".compute_breakdown.json").read_text())
    # Input filename provenance may differ; attribution values may not.
    for key in ("requests", "summary", "skipped_requests", "phase_reentry_after_decode_requests"):
        if old.get(key) != new.get(key):
            raise AssertionError(f"regression in {key}; compare {args.baseline} with output")
    print(json.dumps({"passed": True, **result}, indent=2))


if __name__ == "__main__":
    main()
