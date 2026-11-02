"""Serve with capture, attach eBPF separately, then analyze. No benchmark policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime.session import (
    ROOT, check_port, control_many, control_paths, health_url, healthy,
    option_value, process_identity, read_state, session_active, session_environment,
    vllm_arguments, wait_monitor, write_json,
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start your vLLM server and collector, not a benchmark")
    serve.add_argument("--session", type=Path, required=True, help="new capture directory (never reused)")
    serve.add_argument("--gpu", "--gpus", help="CUDA_VISIBLE_DEVICES; omitted = inherit your environment")
    serve.add_argument("--python", choices=("on", "off"), default="on", dest="python_capture")
    serve.add_argument("--cupti", choices=("on", "off"), default="on")
    serve.add_argument("--startup-timeout", type=float, default=1800)
    serve.add_argument("server_args", nargs=argparse.REMAINDER, help="-- MODEL [original vLLM serve options]")
    ebpf = commands.add_parser("ebpf", help="attach only the eBPF monitor; sudo is requested if necessary")
    ebpf.add_argument("--session", type=Path, required=True)
    analysis = commands.add_parser("analyze", help="convert a finished capture; never starts a workload")
    analysis.add_argument("session", type=Path)
    analysis.add_argument("--output", type=Path, help="default: SESSION/trace.json.gz; explicit .json stays plain")
    analysis.add_argument("--overview", action="store_true", help="also write a compressed overview; full output remains unchanged")
    overview = commands.add_parser("overview", help="stream an overview from an existing full trace; no raw-log reanalysis")
    overview.add_argument("source", type=Path, help="session directory or full .json/.json.gz trace")
    overview.add_argument("--output", type=Path, help="default: SOURCE_STEM.overview.json.gz")
    return cli


def serve(args):
    from .instrumentation.bootstrap import check_version
    check_version()
    arguments = vllm_arguments(args.server_args)
    expected_workers = int(option_value(arguments, ("--tensor-parallel-size", "-tp"), "1")) * int(
        option_value(arguments, ("--pipeline-parallel-size", "-pp"), "1"))
    check_port(arguments)
    session = args.session.resolve()
    if len(str(session / "de_latency_monkeypatch_9999999.sock").encode()) >= 108:
        raise ValueError("session path too long for Unix sockets; choose a shorter directory")
    controller = ROOT / "build/controller"
    if not controller.is_file() or (args.cupti == "on" and not (ROOT / "build/libtracer.so").is_file()):
        raise ValueError(f"native artifacts missing: run make in {ROOT}")
    session.mkdir(parents=True, exist_ok=False)
    env = session_environment(session, python_enabled=args.python_capture == "on",
                              cupti_enabled=args.cupti == "on", gpu=args.gpu)
    # The service command is unchanged except for the instrumentation bootstrap.
    command = [str(controller), sys.executable, "-m", "de_latency.instrumentation.launch", *arguments]
    manifest = {"mode": "serve", "session_id": session.name, "status": "starting",
                "server_args": arguments, "server_command": command,
                "python_capture": args.python_capture == "on", "cupti_capture": args.cupti == "on",
                "gpu": env.get("CUDA_VISIBLE_DEVICES"), "supervisor_pid": os.getpid(),
                "supervisor_start": process_identity(os.getpid()), "started_ns": time.time_ns()}
    write_json(session / "manifest.json", manifest)
    requested_stop = False

    def stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = None
    active_controls = set()
    errors = []
    ready = False
    returncode = 0
    on_responses = {}
    print(f"[de-latency] session: {session}", flush=True)
    print(f"[de-latency] Python={args.python_capture}, CUPTI={args.cupti}; no benchmark is started", flush=True)
    try:
        proc = subprocess.Popen(command, env=env, start_new_session=True)
        manifest.update(controller_pid=proc.pid, controller_start=process_identity(proc.pid))
        write_json(session / "manifest.json", manifest)
        deadline = time.monotonic() + args.startup_timeout
        while not requested_stop and proc.poll() is None:
            state = read_state(session)
            if state.get("target_pid") and not (session / "target.pid").exists():
                (session / "target.pid").write_text(state["target_pid"] + "\n")
            if not ready and state.get("target_pid") and healthy(health_url(arguments)):
                if args.cupti == "on":
                    paths = control_paths(session, "cupti", worker_only=args.python_capture == "on")
                    if not paths or (args.python_capture == "on" and len(paths) < expected_workers):
                        if time.monotonic() > deadline:
                            raise TimeoutError("server is healthy but no CUPTI control sockets were discovered")
                        time.sleep(0.2)
                        continue
                    # Record each successful enable so cleanup also covers partial failures.
                    for path in paths:
                        on_responses.update(control_many([path], "on"))
                        active_controls.add(path)
                    write_json(session / "cupti_on.json", on_responses)
                ready = True
                manifest.update(status="ready", capture_ready_ns=time.time_ns(), target_pid=int(state["target_pid"]),
                                target_start=process_identity(int(state["target_pid"])))
                write_json(session / "manifest.json", manifest)
                print("\n[de-latency] READY: run your original benchmark command in another terminal.", flush=True)
                print(f"[de-latency] Optional eBPF: de-latency ebpf --session {session}", flush=True)
                print("[de-latency] After requests finish, Ctrl+C here to drain and close the capture.\n", flush=True)
            if not ready and time.monotonic() > deadline:
                raise TimeoutError("vLLM/capture startup timed out")
            time.sleep(0.2)
        if proc.poll() is not None:
            returncode = proc.returncode
            if not ready:
                errors.append("server exited before capture became ready")
    except Exception as exc:
        errors.append(str(exc))
    finally:
        manifest["status"] = "draining"
        write_json(session / "manifest.json", manifest)
        print("[de-latency] Stopping capture and draining collector...", flush=True)
        responses = {}
        for path in sorted(active_controls):
            try:
                responses.update(control_many([path], "off"))
            except (OSError, RuntimeError) as exc:
                errors.append(f"CUPTI final status unavailable: {exc}")
        if args.cupti == "on":
            write_json(session / "cupti_off.json", responses)
            for name, response in responses.items():
                fields = dict(word.split("=", 1) for word in response.split() if "=" in word)
                if any(fields.get(key) != "0" for key in ("ring_dropped", "zmq_dropped", "cupti_dropped")):
                    errors.append(f"{name}: native activity loss reported")
        for path in control_paths(session, "monkeypatch"):
            try:
                control_many([path], "off")
            except (OSError, RuntimeError) as exc:
                if proc is not None and proc.poll() is None:
                    errors.append(f"Python capture stop failed: {exc}")
        (session / "stop").touch()
        try:
            wait_monitor(session)
        except TimeoutError as exc:
            errors.append(str(exc))
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()  # Controller owns and stops only its own server group.
            try:
                proc.wait(timeout=180)
                if requested_stop and proc.returncode not in (0, 128 + signal.SIGTERM, 128 + signal.SIGINT):
                    errors.append(f"controller shutdown failed: exit={proc.returncode}")
            except subprocess.TimeoutExpired:
                # Never kill an unknown process group or silently claim a drain.
                errors.append("controller shutdown timed out; capture may be incomplete")
        stats = session / "events.log.stats.json"
        if not stats.exists():
            errors.append("missing collector final statistics")
        else:
            try:
                counts = json.loads(stats.read_text())
                if counts.get("invalid_events") or any(counts.get("producer_dropped", {}).values()):
                    errors.append("collector reported invalid events or producer loss")
            except (OSError, ValueError) as exc:
                errors.append(f"collector final statistics unreadable: {exc}")
        ebpf_stats = session / "ebpf.stats.json"
        if ebpf_stats.exists():
            try:
                counters = json.loads(ebpf_stats.read_text())
                if any(counters.get(key) != 0 for key in ("ring_dropped", "map_dropped", "send_dropped", "exit_code")):
                    errors.append("eBPF reported failure or loss")
            except (OSError, ValueError) as exc:
                errors.append(f"eBPF final statistics unreadable: {exc}")
        manifest.update(status="failed" if errors or returncode else "finished",
                        ended_ns=time.time_ns(), errors=errors, exit_code=returncode,
                        controller_exit_code=proc.returncode if proc is not None else None)
        write_json(session / "manifest.json", manifest)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    for error in errors:
        print(f"[de-latency] ERROR: {error}", file=sys.stderr)
    print(f"[de-latency] Analyze: de-latency analyze {session}", flush=True)
    return 1 if errors else returncode


def ebpf(args):
    session = args.session.resolve()
    manifest = json.loads((session / "manifest.json").read_text())
    if not session_active(manifest) or (session / "stop").exists():
        raise ValueError("session is not running; start de-latency serve first")
    if (session / "ebpf.ready").exists():
        raise ValueError("an eBPF monitor was already started for this session")
    monitor = ROOT / "build/ebpf_monitor"
    if not monitor.is_file():
        raise ValueError(f"eBPF artifact missing: run make in {ROOT}")
    deadline = time.monotonic() + 30
    while not (session / "target.pid").exists():
        if not session_active(manifest) or time.monotonic() >= deadline:
            raise TimeoutError("server PID was not published")
        time.sleep(0.1)
    target = int((session / "target.pid").read_text())
    if target <= 0 or process_identity(target) is None:
        raise ValueError("session server PID is not running")
    if manifest.get("target_start") is not None and process_identity(target) != manifest["target_start"]:
        raise ValueError("session server PID was recycled; refusing to attach")
    env = session_environment(session)
    keys = ("TRACER_ZMQ_ADDR", "DE_LATENCY_SESSION_ID", "DE_LATENCY_EBPF_READY_FILE",
            "DE_LATENCY_EBPF_STATS_FILE", "DE_LATENCY_STOP_FILE")
    command = ["env", *(f"{key}={env[key]}" for key in keys), str(monitor),
               "--auto", "--root-pid", str(target)]
    if manifest["python_capture"]:
        command += ["--worker-pid-file", str(session / "targets")]
    else:
        # Without hooks there is no role/PID publication. Track main threads
        # within this server's process tree, never arbitrary name matches.
        command += ["--worker-pattern", ""]
    if os.geteuid() != 0:
        command.insert(0, "sudo")
    print("[de-latency] Starting eBPF only; wait for 'eBPF Tracer Running' before your benchmark.", flush=True)
    os.execvp(command[0], command)


def analyze(args):
    session = args.session.resolve()
    manifest_path = session / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        live_controller = (manifest.get("controller_start") is not None and
                           process_identity(manifest.get("controller_pid")) == manifest["controller_start"])
        if session_active(manifest) or live_controller:
            raise ValueError("capture is still running: finish your benchmark, then Ctrl+C the serve terminal and wait for drain")
        if manifest.get("status") in ("starting", "ready", "draining"):
            raise ValueError("capture did not shut down cleanly; inspect its manifest and collector before analyzing")
        if manifest.get("status") == "failed":
            print("[de-latency] WARNING: analyzing a failed/partial capture; see manifest.json", file=sys.stderr)
    if not (session / "events.log.stats.json").exists():
        raise ValueError("missing collector final statistics; do not analyze an active or undrained log")
    from .analysis.pipeline import process_logs
    from .export.trace_io import artifact_stem, overview_path
    output = args.output.resolve() if args.output else session / "trace.json.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    process_logs(str(session / "events.log"), str(output), overview=args.overview)
    print(f"[de-latency] Perfetto: {output}")
    if args.overview:
        print(f"[de-latency] Overview (omits operator/API/flow detail): {overview_path(output)}")
    print(f"[de-latency] Request charts: {Path(str(artifact_stem(output)) + '.compute_breakdown_requests') / 'index.html'}")
    return 0


def overview(args):
    from .export.overview import overview_from_trace
    source = args.source.resolve()
    if source.is_dir():
        source = next((source / name for name in ("trace.json.gz", "trace.json")
                       if (source / name).is_file()), None)
        if source is None:
            raise ValueError("no full trace.json.gz or trace.json found; run analyze first")
    output, count = overview_from_trace(source, args.output)
    print(f"[de-latency] Overview: {output} ({count:,} retained events)")
    print("[de-latency] Only the view is filtered; raw logs, the full trace and TTFT/TPOT files are unchanged.")
    return 0


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        return {"serve": serve, "ebpf": ebpf, "analyze": analyze, "overview": overview}[args.command](args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"[de-latency] {exc}", file=sys.stderr)
        return 1
