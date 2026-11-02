"""CPU-affinity experiment matrix using the same validated session runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from .gpu_monitor import monitor_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("DE_LATENCY_MODEL"))
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--counts", default="1,2,3,4,8")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--ebpf", choices=["sudo", "external", "off"], default="sudo")
    parser.add_argument("--results", type=Path, default=Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if not args.model:
        parser.error("set DE_LATENCY_MODEL or pass --model")
    available = sorted(os.sched_getaffinity(0))
    root = Path(__file__).resolve().parents[1]
    args.results.mkdir(parents=True, exist_ok=False)
    results = {}
    for count in map(int, args.counts.split(",")):
        if count < 1 or count > len(available):
            parser.error(f"invalid CPU count: {count}")
        cores = ",".join(map(str, available[:count]))
        session = args.results / f"{count}cores"
        cmd = [sys.executable, str(root / "benchmarks/validate.py"), "--model", args.model,
               "--gpu", args.gpu, "--tp", str(args.tp), "--pp", str(args.pp),
               "--prompts", str(args.prompts), "--session", str(session), "--cpu-cores", cores,
               "--ebpf", args.ebpf]
        # The runner creates its session directory; keep the sampler beside it.
        gpu_csv = args.results / f"{count}cores.gpu.csv"
        with monitor_gpu(gpu_csv):
            completed = subprocess.run(cmd)
        result = {"n_cores": count, "cores": cores, "status": "failed",
                  "benchtime_s": None, "tpot_ms": None, "mean_os_delay_ms": None,
                  "gpu_csv": str(gpu_csv), "bench_output": str(session / "benchmark.log")}
        if completed.returncode == 0:
            bench = json.loads((session / "benchmark.json").read_text())
            summary = json.loads((session / "trace.summary.json").read_text())
            delays = list(summary["os_delay_ms_by_request"].values())
            result.update(status="ok", benchtime_s=bench.get("duration"), tpot_ms=bench.get("mean_tpot_ms"),
                          mean_os_delay_ms=sum(delays) / len(delays) if delays else None)
        results[str(count)] = result
        (args.results / "all_results.json").write_text(json.dumps(results, indent=2))
    return int(any(result["status"] != "ok" for result in results.values()))
