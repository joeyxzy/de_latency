#!/usr/bin/env python3
"""Isolated real vLLM benchmark and three-source acceptance checks.

Run with the vLLM 0.20 interpreter. --ebpf external permits a separately
authorized monitor without running the server or collector as root.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.acceptance import validate_capture
from de_latency.runtime.socket_client import send_command
from de_latency.runtime.session import STARTUP


def wait_for(predicate, timeout, label, processes=()):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for proc in processes:
            if proc.poll() is not None:
                raise RuntimeError(f"{label}: process {proc.pid} exited {proc.returncode}")
        if predicate():
            return
        time.sleep(0.2)
    raise TimeoutError(label)


def stop_process(proc, timeout=30):
    if proc is None:
        return
    # The process group was created by this harness, never selected by name.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def gpu_uuids(selection):
    """Record physical devices; vLLM 0.20 expects numeric visible-device IDs."""
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    by_index = dict(tuple(part.strip() for part in line.split(",", 1))
                    for line in devices.splitlines())
    return ",".join(by_index[item] if item.isdigit() else item
                    for item in (part.strip() for part in selection.split(",")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--ebpf", choices=["sudo", "external", "off"], default="sudo")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", default="0.45")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cpu-cores", help="taskset CPU list applied only to the server")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=False)
    if len(str(session / "de_latency_monkeypatch_9999999.sock").encode()) >= 108:
        raise ValueError("session path too long for Unix sockets")
    env = os.environ.copy()
    env.update({
        "DE_LATENCY_SESSION_ID": session.name,
        "DE_LATENCY_LOG_PATH": str(session / "events.log"),
        "TRACER_ZMQ_ADDR": f"ipc://{session}/events.sock",
        "TRACER_WORKER_PID_FILE": str(session / "targets"),
        "TRACER_MONKEYPATCH_CONTROL_DIR": str(session),
        "TRACER_CUPTI_CONTROL_DIR": str(session),
        "DE_LATENCY_EBPF_READY_FILE": str(session / "ebpf.ready"),
        "DE_LATENCY_STOP_FILE": str(session / "stop"),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": os.pathsep.join([str(STARTUP), str(ROOT), env.get("PYTHONPATH", "")]),
        "TRACER_CUPTI_START_ENABLED": "0",
        "TRACER_CUPTI_START_DELAY_SEC": "0",
        "TRACER_ZMQ_SNDHWM": "200000",
        "TRACER_DEBUG_LOG_PATH": str(session / "cupti.log"),
        "TRACER_DRAIN_TIMEOUT_SEC": "120",
    })
    plain_env = {**env, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1"}
    for key in ("NO_PROXY", "no_proxy"):
        plain_env[key] = ",".join(filter(None, [plain_env.get(key), "127.0.0.1", "localhost"]))
    plain_env.pop("LD_PRELOAD", None)
    collector = server = monitor = None
    files = []

    def spawn(name, cmd, process_env):
        out = (session / f"{name}.log").open("w")
        files.append(out)
        return subprocess.Popen(cmd, env=process_env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    # Bind a free local port before launch; server health is checked separately.
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    manifest = {"python": sys.executable, "model": args.model, "gpu": args.gpu,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "gpu_uuids": gpu_uuids(args.gpu),
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "graph": args.graph, "tp": args.tp, "pp": args.pp, "port": port}
    (session / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        collector = spawn("collector", [sys.executable, "-m", "de_latency.collector"], plain_env)
        wait_for(lambda: (session / "events.sock").exists(), 15, "collector ready", [collector])
        server_env = {**env, "LD_PRELOAD": str(ROOT / "build/libtracer.so"), "DE_LATENCY_DISABLE_SITECUSTOMIZE": "0"}
        cmd = [sys.executable, "-m", "de_latency.instrumentation.launch", "--model", args.model,
               "--host", "127.0.0.1", "--port", str(port), "--max-model-len", "1024",
               "--max-num-seqs", "16", "--gpu-memory-utilization", args.gpu_memory_utilization,
               "--tensor-parallel-size", str(args.tp), "--pipeline-parallel-size", str(args.pp),
               "--no-enable-log-requests"]
        if not args.graph:
            cmd += ["--enforce-eager"]
        if args.cpu_cores:
            cmd = ["taskset", "-c", args.cpu_cores, *cmd]
        server = spawn("server", cmd, server_env)
        (session / "target.pid").write_text(str(server.pid) + "\n")
        if args.ebpf == "sudo":
            monitor = spawn("ebpf", ["sudo", "-n", "bash", str(ROOT / "benchmarks/ebpf_session.sh"), str(session)], plain_env)
        if args.ebpf != "off":
            print(f"Waiting for eBPF: sudo bash {ROOT}/benchmarks/ebpf_session.sh {session}", flush=True)
            wait_for(lambda: (session / "ebpf.ready").exists(), args.timeout, "eBPF ready", [server, collector] + ([monitor] if monitor else []))

        def healthy():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False

        wait_for(healthy, args.timeout, "vLLM ready", [server, collector])
        worker_pids = {int(line) for line in (session / "targets").read_text().splitlines() if line.isdigit()}
        controls = [session / f"de_latency_cupti_{pid}.sock" for pid in sorted(worker_pids)]
        if not controls:
            raise RuntimeError("no discovered worker CUPTI control sockets")
        responses = {}
        for path in controls:
            response = send_command(str(path), "on", timeout=30)
            responses[path.name] = response
            if not response.startswith("ok ") or "state=on" not in response:
                raise RuntimeError(f"CUPTI enable failed: {response}")
        (session / "cupti_on.json").write_text(json.dumps(responses, indent=2))
        started_ns = time.time_ns()
        bench_cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                     "--backend", "vllm", "--model", args.model, "--host", "127.0.0.1",
                     "--port", str(port), "--endpoint", "/v1/completions", "--dataset-name", "random",
                     "--random-input-len", str(args.input_len), "--random-output-len", str(args.output_len),
                     "--num-prompts", str(args.prompts), "--request-rate", "4", "--ignore-eos",
                     "--save-result", "--save-detailed", "--result-dir", str(session), "--result-filename", "benchmark.json"]
        bench = spawn("benchmark", bench_cmd, plain_env)
        try:
            rc = bench.wait(timeout=args.timeout)
            if rc:
                raise RuntimeError(f"benchmark failed: {rc}")
        finally:
            stop_process(bench)
        ended_ns = time.time_ns()
        benchmark_result = json.loads((session / "benchmark.json").read_text())
        if benchmark_result.get("completed") != args.prompts or benchmark_result.get("failed", 0):
            raise RuntimeError("benchmark did not complete every request")
        responses = {}
        for path in controls:
            responses[path.name] = send_command(str(path), "off", timeout=60)
        (session / "cupti_off.json").write_text(json.dumps(responses, indent=2))
        manifest.update(measurement_start_ns=started_ns, measurement_end_ns=ended_ns)
        (session / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        (session / "stop").touch()
        stop_process(server)
        if monitor:
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_process(monitor)
        try:
            if args.ebpf == "external" and (session / "ebpf.ready").exists():
                wait_for(lambda: (session / "ebpf.stats.json").exists(), 15, "eBPF final statistics")
        finally:
            stop_process(collector, timeout=135)
            for out in files:
                out.close()
        if collector is not None and collector.returncode != 0:
            raise RuntimeError(f"collector failed to drain: exit={collector.returncode}; see collector.log")

    required = {"monkey_patch", "cupti"} | ({"ebpf"} if args.ebpf != "off" else set())
    with (session / "analysis.log").open("w") as out:
        subprocess.run([sys.executable, "-m", "de_latency.analysis.pipeline",
                         str(session / "events.log"), str(session / "trace.json.gz")],
                       env=plain_env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
    report = validate_capture(session, args.prompts, required)
    (session / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
