#!/usr/bin/env python3
"""
Automated experiment orchestration for vLLM latency benchmarking.
Runs experiments with different CPU core counts (1,2,3,4,8) and
collects benchtime, TPOT, mean OS delay, and GPU utilization.

Usage: sudo -E python3 run_experiments.py
"""

import subprocess
import signal
import time
import os
import sys
import re
import json
import csv
import shutil
import threading
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build"
CONDA_PREFIX = "/home/joeyxzy/miniconda3/envs/vllm-0200"
CONDA_PYTHON = f"{CONDA_PREFIX}/bin/python"
CONDA_VLLM = f"{CONDA_PREFIX}/bin/vllm"
MODEL = "/home/joeyxzy/models/Qwen2.5-32B-Instruct"
VLLM_SRC = "/home/joeyxzy/vllm_0.20.1/vllm"
VLLM_TRACE = str(REPO_ROOT / "vllm_trace")
LD_LIBRARY_PATH = (
    f"{CONDA_PREFIX}/lib:"
    "/home/joeyxzy/jsonc_install/lib:"
    "/home/joeyxzy/.local/lib"
)
PORT = 8001
HOST = "127.0.0.1"
GPU_IDS = "0,1,2,3,4,5,6,7"
PP_SIZE = 4
TP_SIZE = 2
MAX_MODEL_LEN = 20000
NUM_PROMPTS = 1024
INPUT_LEN = 8
OUTPUT_LEN = 4
MODEL_LOAD_TIMEOUT = 600
GPU_MONITOR_INTERVAL = 0.1
CONTROLLER_KILL_TIMEOUT = 15

RESULTS_DIR = REPO_ROOT / "results"
LOG_FILE = REPO_ROOT / "de_latency.log"
LOG_TO_TRACE_PY = REPO_ROOT / "perfetto" / "log_to_trace.py"
LOG_TO_TRACE_PY_ORIG = REPO_ROOT / "log_to_trace.py"

CORE_CONFIGS = {
    1: "40",
    2: "40,41",
    3: "40,41,42",
    4: "40,41,42,43",
    8: "40,41,42,43,44,45,46,47",
}

# ── Helpers ────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_cmd(cmd, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess. Raises on non-zero exit."""
    log(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, check=True, **kwargs)


def kill_sudo_process(pid: int) -> None:
    """Kill a process. Tries direct kill first, then sudo."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            subprocess.run(
                ["sudo", "-n", "kill", "-TERM", str(pid)],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass


def cleanup_stale() -> None:
    """Kill leftover processes and remove stale files from previous runs."""
    log("Cleaning up stale processes and files...")
    patterns = [
        "controller",
        "start_server.py",
        "vllm.entrypoints",
        "collector.py",
        "ebpf_monitor",
        "VLLM::Worker",
    ]
    for pat in patterns:
        # Try with sudo first (if running as non-root), then without
        try:
            subprocess.run(
                ["pkill", "-f", pat],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        try:
            subprocess.run(
                ["sudo", "-n", "pkill", "-f", pat],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        time.sleep(0.2)

    # Remove IPC socket and state files
    stale_files = [
        "/tmp/tracer.sock",
        "/tmp/tracer_worker_pids",
        "/tmp/de_latency_control_state",
    ]
    for f in stale_files:
        try:
            Path(f).unlink(missing_ok=True)
        except PermissionError:
            subprocess.run(
                ["sudo", "-n", "rm", "-f", f],
                capture_output=True, timeout=2,
            )

    # Give OS time to release ports / sockets
    time.sleep(2)
    log("Cleanup done.")


def reset_gpu_memory() -> None:
    """Find and kill all processes using GPU memory, wait until all GPUs freed."""
    log("Resetting GPU memory...")
    subprocess.run(
        ["pkill", "-9", "-f", "VLLM::Worker"],
        capture_output=True, timeout=5,
    )
    time.sleep(5)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        used = [int(v.strip()) for v in result.stdout.strip().split("\n") if v.strip()]
        max_used = max(used) if used else 0
        log(f"  GPU memory max_used={max_used} MiB (after reset)")
    except Exception:
        pass


# ── GPU Monitor ────────────────────────────────────────────────────────────


def gpu_monitor_worker(csv_path: Path, stop_event: threading.Event,
                       interval: float = 0.5) -> None:
    """Background thread: polls nvidia-smi and writes GPU utilization to CSV."""
    header = ["timestamp"] + [f"gpu{i}_util" for i in range(8)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        while not stop_event.is_set():
            t0 = time.time()
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                values = [
                    v.strip().replace(" %", "").replace("%", "")
                    for v in result.stdout.strip().split("\n")
                    if v.strip()
                ]
                if len(values) >= 8:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow([ts] + values[:8])
                    f.flush()
            except Exception as e:
                log(f"[GPU Monitor] warning: {e}")
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)


def start_gpu_monitor(csv_path: Path) -> tuple[threading.Thread, threading.Event]:
    """Start GPU monitor in a background thread."""
    stop_event = threading.Event()
    t = threading.Thread(
        target=gpu_monitor_worker,
        args=(csv_path, stop_event, GPU_MONITOR_INTERVAL),
        daemon=True,
    )
    t.start()
    log(f"GPU monitor started (interval={GPU_MONITOR_INTERVAL}s) -> {csv_path}")
    return t, stop_event


# ── Server Launch ──────────────────────────────────────────────────────────


def start_server(cores: str, server_log: Path) -> subprocess.Popen:
    """Launch vLLM via controller on specified CPU cores. Returns Popen."""
    cmd = [
        "sudo",
        "taskset", "-c", cores,
        "/usr/bin/env",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"CUDA_VISIBLE_DEVICES={GPU_IDS}",
        f"PYTHONPATH={VLLM_TRACE}:{VLLM_SRC}",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        f"DE_LATENCY_LOG_PATH={LOG_FILE}",
        "TRACER_CUPTI_START_ENABLED=0",
        str(BUILD_DIR / "controller"),
        CONDA_PYTHON,
        str(REPO_ROOT / "vllm_trace" / "start_server.py"),
        "--model", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--pipeline-parallel-size", str(PP_SIZE),
        "--tensor-parallel-size", str(TP_SIZE),
    ]

    log(f"Starting server (cores={cores})...")
    with open(server_log, "w") as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
    log(f"Controller PID={proc.pid}")
    return proc


def wait_for_server(timeout: int = MODEL_LOAD_TIMEOUT) -> bool:
    """Poll /v1/models until the server responds, or timeout."""
    log(f"Waiting for model to load (timeout={timeout}s)...")
    for i in range(timeout):
        try:
            result = subprocess.run(
                ["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                log(f"Model loaded after {i}s")
                return True
        except Exception:
            pass
        time.sleep(1)
    log("TIMEOUT — server did not respond")
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM to controller, wait, then force-kill if needed."""
    if proc is None or proc.poll() is not None:
        return

    log(f"Stopping controller (PID={proc.pid})...")
    kill_sudo_process(proc.pid)

    try:
        proc.wait(timeout=CONTROLLER_KILL_TIMEOUT)
        log("Controller exited cleanly.")
    except subprocess.TimeoutExpired:
        log("Timeout — force-killing...")
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            subprocess.run(
                ["sudo", "-n", "kill", "-9", str(proc.pid)],
                capture_output=True, timeout=5,
            )
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    cleanup_stale()


# ── Benchmark ──────────────────────────────────────────────────────────────


def run_benchmark(bench_log: Path) -> tuple[float, float, str]:
    """Run vllm bench serve. Returns (benchtime, tpot_ms, raw_output)."""
    log("Running benchmark...")
    cmd = [
        CONDA_VLLM, "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--num-prompts", str(NUM_PROMPTS),
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--request-rate", "inf",
        "--max-concurrency", "512",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": f"{VLLM_TRACE}:{VLLM_SRC}"},
    )

    raw_output = proc.stdout + proc.stderr

    with open(bench_log, "w") as f:
        f.write(raw_output)

    if proc.returncode != 0:
        log(f"Benchmark exited with code {proc.returncode}")

    benchtime = None
    tpot_ms = None

    for line in raw_output.split("\n"):
        m = re.search(r"Benchmark duration \(s\):\s+([\d.]+)", line)
        if m:
            benchtime = float(m.group(1))
        m = re.search(r"Mean TPOT \(ms\):\s+([\d.]+)", line)
        if m:
            tpot_ms = float(m.group(1))

    log(f"  benchtime={benchtime}, tpot={tpot_ms}")
    return benchtime or 0.0, tpot_ms or 0.0, raw_output


# ── OS Delay Analysis ──────────────────────────────────────────────────────


def run_log_to_trace(trace_out: Path) -> tuple[float, str]:
    """Run log_to_trace.py and extract mean OS delay per request."""
    log("Running log_to_trace.py...")
    log_to_trace = LOG_TO_TRACE_PY_ORIG if LOG_TO_TRACE_PY_ORIG.exists() else LOG_TO_TRACE_PY

    if not LOG_FILE.exists():
        log("  WARNING: Log file not found, os_delay=0")
        return 0.0, ""

    proc = subprocess.run(
        [sys.executable, str(log_to_trace), str(LOG_FILE), str(trace_out)],
        capture_output=True, text=True, timeout=300,
    )
    stdout = proc.stdout

    with open(trace_out.with_suffix(".log_to_trace_stdout.txt"), "w") as f:
        f.write(stdout)

    if proc.returncode != 0:
        log(f"  log_to_trace exited with code {proc.returncode}")
        log(f"  stderr: {proc.stderr[:500]}")

    # Parse per-request OS delays
    total_os_delay_ms = 0.0
    count = 0
    for line in stdout.split("\n"):
        m = re.search(
            r"Request\s+\S+:\s+([\d.]+)\s+ms\s+\(OS Scheduling",
            line,
        )
        if m:
            total_os_delay_ms += float(m.group(1))
            count += 1

    mean_os_delay = total_os_delay_ms / count if count > 0 else 0.0
    log(f"  OS delays: {count} requests affected, mean={mean_os_delay:.3f} ms")
    return mean_os_delay, stdout


# ── Orchestration ──────────────────────────────────────────────────────────


def run_experiment(n_cores: int, cores: str) -> dict:
    """Run one experiment configuration. Returns results dict."""
    exp_name = f"{n_cores}cores"
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*60}")
    log(f"=== Experiment: {exp_name} (cores={cores}) ===")
    log(f"{'='*60}")

    server_log = exp_dir / "server.log"
    bench_log = exp_dir / "bench_output.txt"
    gpu_csv = exp_dir / "gpu_util.csv"
    trace_json = exp_dir / "trace.json"

    # 1. Cleanup (do NOT delete log file yet — it will be cleared by collector on start)
    cleanup_stale()

    # 1b. Delete old log file and results from previous run
    for f in [LOG_FILE, trace_json, LOG_FILE.with_suffix(".log_to_trace_stdout.txt")]:
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass

    # 2. Start GPU monitor
    gpu_thread, gpu_stop = start_gpu_monitor(gpu_csv)

    # 3. Start server
    server_proc = start_server(cores, server_log)

    # 4. Wait for model
    if not wait_for_server():
        log("ERROR: Server failed to start, skipping this experiment.")
        stop_server(server_proc)
        gpu_stop.set()
        gpu_thread.join(timeout=5)
        return {
            "n_cores": n_cores,
            "cores": cores,
            "benchtime_s": None,
            "tpot_ms": None,
            "mean_os_delay_ms": None,
            "status": "server_start_failed",
        }

    # 5. Run benchmark (restart GPU monitor right before benchmark for cleaner data)
    # Actually, we keep the GPU monitor running throughout
    try:
        benchtime, tpot, bench_output = run_benchmark(bench_log)
    except Exception as e:
        log(f"ERROR: Benchmark failed: {e}")
        benchtime, tpot = 0.0, 0.0

    # 6. Stop server
    stop_server(server_proc)
    reset_gpu_memory()
    time.sleep(2)

    # 7. Run log_to_trace
    try:
        mean_os_delay, lt_out = run_log_to_trace(trace_json)
    except Exception as e:
        log(f"ERROR: log_to_trace failed: {e}")
        mean_os_delay = 0.0

    # 8. Stop GPU monitor
    gpu_stop.set()
    gpu_thread.join(timeout=5)

    result = {
        "n_cores": n_cores,
        "cores": cores,
        "benchtime_s": benchtime,
        "tpot_ms": tpot,
        "mean_os_delay_ms": mean_os_delay,
        "status": "ok",
        "gpu_csv": str(gpu_csv),
        "bench_output": str(bench_log),
    }
    log(f"Experiment {exp_name} done: {result}")
    return result


def main() -> None:
    log("=" * 60)
    log("vLLM Latency Benchmark — Automated Experiment Runner")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n_cores in [1, 2, 3, 4, 8]:
        cores = CORE_CONFIGS[n_cores]
        try:
            result = run_experiment(n_cores, cores)
        except Exception as e:
            log(f"FATAL: Experiment {n_cores}cores failed: {e}")
            result = {
                "n_cores": n_cores,
                "cores": cores,
                "status": "crashed",
                "error": str(e),
            }
        all_results[str(n_cores)] = result

        # Save intermediate results
        summary_path = RESULTS_DIR / "all_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"Intermediate results saved to {summary_path}")

        # Brief pause between experiments
        time.sleep(5)

    log("\n" + "=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    log(f"Results: {RESULTS_DIR / 'all_results.json'}")
    log("Next: run 'python3 plot_results.py' to generate charts")
    log("=" * 60)


if __name__ == "__main__":
    main()
