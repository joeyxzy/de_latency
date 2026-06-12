#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import subprocess
import sys
import time


DEFAULT_STATE_FILE = os.getenv("DE_LATENCY_CONTROL_STATE_FILE", "/tmp/de_latency_control_state")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified runtime control for de_latency tracing tools.")
    parser.add_argument("command", choices=["status", "on", "off"], help="Control command.")
    parser.add_argument(
        "--tool",
        choices=["all", "monkeypatch", "cupti", "ebpf"],
        default="all",
        help="Which tracing tool to control.",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help="Path to the controller runtime state file.",
    )
    parser.add_argument(
        "--mode",
        choices=["graceful", "fast"],
        default="graceful",
        help="Only applies to CUPTI off.",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout in seconds.")
    parser.add_argument(
        "--markers-only",
        choices=["on", "off"],
        default=None,
        help="Enable/disable CUPTI markers-only mode (only collect de_marker events to reduce log size).",
    )
    return parser.parse_args()


def load_state(path):
    state = {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                state[key.strip()] = value.strip()
    except FileNotFoundError:
        raise SystemExit(f"state file not found: {path}")
    return state


def save_state(path, state):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        for key in sorted(state):
            fp.write(f"{key}={state[key]}\n")
    os.replace(tmp_path, path)


def int_or_zero(value):
    try:
        return int(value)
    except Exception:
        return 0


def pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_ppid(pid):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def descendant_pids(root_pid):
    if root_pid <= 0:
        return []

    ppid_map = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        ppid = read_ppid(pid)
        if ppid is not None:
            ppid_map.setdefault(ppid, []).append(pid)

    results = []
    stack = list(ppid_map.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in results:
            continue
        results.append(pid)
        stack.extend(ppid_map.get(pid, []))
    return sorted(results)


def read_worker_pids(path):
    pids = []
    if not path:
        return pids
    try:
        with open(path, "r", encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line or line.startswith("tid:"):
                    continue
                try:
                    pid = int(line)
                except ValueError:
                    continue
                if pid > 0:
                    pids.append(pid)
    except FileNotFoundError:
        return pids
    return sorted(set(pids))


def unique_pids(*groups):
    seen = set()
    ordered = []
    for group in groups:
        for pid in group:
            if pid > 0 and pid not in seen:
                seen.add(pid)
                ordered.append(pid)
    return ordered


def monkeypatch_socket_path(pid):
    control_dir = os.getenv("TRACER_MONKEYPATCH_CONTROL_DIR", "/tmp")
    return os.path.join(control_dir, f"de_latency_monkeypatch_{pid}.sock")


def cupti_socket_path(pid):
    control_dir = os.getenv("TRACER_CUPTI_CONTROL_DIR", "/tmp")
    return os.path.join(control_dir, f"de_latency_cupti_{pid}.sock")


def send_socket_command(path, command, timeout):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall((command + "\n").encode("utf-8"))
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def control_socket_group(tool_name, pids, command, timeout, path_builder):
    results = []
    failures = []
    for pid in pids:
        path = path_builder(pid)
        if not os.path.exists(path):
            continue
        try:
            response = send_socket_command(path, command, timeout)
            results.append((pid, response))
        except OSError as exc:
            failures.append((pid, str(exc)))
    if not results and not failures:
        failures.append((0, "no control sockets found"))
    return results, failures


def ebpf_status(state):
    pid = int_or_zero(state.get("ebpf_pid"))
    alive = pid_alive(pid)
    return f"ebpf state={'on' if alive else 'off'} pid={pid if alive else 0}"


def stop_ebpf(state, state_file):
    pid = int_or_zero(state.get("ebpf_pid"))
    if pid > 0 and pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not pid_alive(pid):
                break
            time.sleep(0.1)
        if pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    state["ebpf_pid"] = "0"
    save_state(state_file, state)
    return "ebpf state=off pid=0"


def start_ebpf(state, state_file):
    pid = int_or_zero(state.get("ebpf_pid"))
    if pid > 0 and pid_alive(pid):
        return f"ebpf state=on pid={pid}"

    target_pid = int_or_zero(state.get("target_pid"))
    if target_pid <= 0 or not pid_alive(target_pid):
        raise RuntimeError("target process is not running")

    ebpf_monitor_bin = state.get("ebpf_monitor_bin", "")
    if not ebpf_monitor_bin:
        raise RuntimeError("ebpf_monitor_bin missing from state file")

    worker_pid_file = state.get("worker_pid_file", "")
    cmd = [
        ebpf_monitor_bin,
        "--auto",
        "--root-pid",
        str(target_pid),
    ]
    if worker_pid_file:
        cmd.extend(["--worker-pid-file", worker_pid_file])

    env = os.environ.copy()
    if state.get("zmq_addr"):
        env["TRACER_ZMQ_ADDR"] = state["zmq_addr"]

    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    time.sleep(0.3)
    if proc.poll() is not None:
        raise RuntimeError(f"ebpf_monitor exited immediately with code {proc.returncode}")

    state["ebpf_pid"] = str(proc.pid)
    save_state(state_file, state)
    return f"ebpf state=on pid={proc.pid}"


def print_results(title, results, failures):
    print(title)
    for pid, response in results:
        print(f"  pid={pid} {response}")
    for pid, error in failures:
        prefix = f"pid={pid} " if pid else ""
        print(f"  {prefix}error={error}")


def monkeypatch_targets(state):
    root_pid = int_or_zero(state.get("target_pid"))
    return unique_pids([root_pid], descendant_pids(root_pid), read_worker_pids(state.get("worker_pid_file")))


def cupti_targets(state):
    root_pid = int_or_zero(state.get("target_pid"))
    return unique_pids([root_pid], descendant_pids(root_pid), read_worker_pids(state.get("worker_pid_file")))


def run_monkeypatch(state, command, timeout):
    return control_socket_group(
        "monkeypatch",
        monkeypatch_targets(state),
        command,
        timeout,
        monkeypatch_socket_path,
    )


def run_cupti(state, command, timeout):
    return control_socket_group(
        "cupti",
        cupti_targets(state),
        command,
        timeout,
        cupti_socket_path,
    )


def main():
    args = parse_args()
    state = load_state(args.state_file)
    overall_failures = 0

    if args.tool in {"all", "monkeypatch"}:
        if args.command == "status":
            results, failures = run_monkeypatch(state, "status", args.timeout)
        elif args.command == "on":
            results, failures = run_monkeypatch(state, "on", args.timeout)
        else:
            results, failures = run_monkeypatch(state, "off", args.timeout)
        print_results("monkeypatch:", results, failures)
        overall_failures += len(failures)

    if args.tool in {"all", "cupti"}:
        if args.markers_only is not None:
            markers_cmd = f"markers_only {args.markers_only}"
            results, failures = run_cupti(state, markers_cmd, args.timeout)
            print_results(f"cupti markers_only={args.markers_only}:", results, failures)
            overall_failures += len(failures)
        cupti_command = args.command
        if args.command == "off" and args.mode == "fast":
            cupti_command = "off fast"
        results, failures = run_cupti(state, cupti_command, args.timeout)
        print_results("cupti:", results, failures)
        overall_failures += len(failures)

    if args.tool in {"all", "ebpf"}:
        try:
            if args.command == "status":
                print(ebpf_status(state))
            elif args.command == "on":
                print(start_ebpf(state, args.state_file))
            else:
                print(stop_ebpf(state, args.state_file))
        except Exception as exc:
            print(f"ebpf error={exc}")
            overall_failures += 1

    return 1 if overall_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
