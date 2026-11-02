"""Configuration and process boundaries shared by the user-facing session CLI."""
import json
import os
from pathlib import Path
import socket
import time
import urllib.request

from .socket_client import send_command

ROOT = Path(__file__).resolve().parents[2]
STARTUP = Path(__file__).resolve().parents[1] / "instrumentation/startup"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_state(session):
    path = Path(session) / "control.state"
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def vllm_arguments(arguments):
    """Only adapt the positional model; never inject benchmark/server tuning."""
    arguments = list(arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    if not arguments:
        raise ValueError("missing model: serve [capture options] -- MODEL [vLLM options]")
    if arguments[0].startswith("-"):
        raise ValueError("put the model first after --, as with vllm serve")
    return ["--model", os.path.expanduser(arguments[0]), *arguments[1:]]


def option_value(arguments, names, default):
    value = default
    for index, argument in enumerate(arguments):
        for name in names:
            if argument == name and index + 1 < len(arguments):
                value = arguments[index + 1]
            elif argument.startswith(name + "="):
                value = argument.split("=", 1)[1]
    return value


def health_url(arguments):
    host = option_value(arguments, ("--host",), "127.0.0.1")
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    port = int(option_value(arguments, ("--port", "-p"), "8000"))
    scheme = "https" if option_value(arguments, ("--ssl-certfile",), None) else "http"
    return f"{scheme}://{'[' + host + ']' if ':' in host else host}:{port}/health"


def check_port(arguments):
    """Do not confuse another user's existing service with our new server."""
    host = option_value(arguments, ("--host",), "127.0.0.1") or "127.0.0.1"
    port = int(option_value(arguments, ("--port", "-p"), "8000"))
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, port))


def healthy(url):
    # Client-side proxies must not redirect the local health probe.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def session_environment(session, *, python_enabled=True, cupti_enabled=True, gpu=None, base=None):
    session = Path(session).resolve()
    env = dict(os.environ if base is None else base)
    # Do not inherit an earlier manual injection of our tracer into collector
    # or disabled-CUPTI sessions. Preserve unrelated user preload libraries.
    preload = env.get("LD_PRELOAD", "")
    if preload:
        tracer = (ROOT / "build/libtracer.so").resolve()
        libraries = [item for item in preload.replace(":", " ").split() if Path(item).resolve() != tracer]
        if libraries:
            env["LD_PRELOAD"] = ":".join(libraries)
        else:
            env.pop("LD_PRELOAD", None)
    env.update({
        "DE_LATENCY_SESSION_ID": session.name,
        "DE_LATENCY_LOG_PATH": str(session / "events.log"),
        "DE_LATENCY_CONTROL_STATE_FILE": str(session / "control.state"),
        "TRACER_ZMQ_ADDR": f"ipc://{session}/events.sock",
        "TRACER_WORKER_PID_FILE": str(session / "targets"),
        "TRACER_TARGET_FILE": str(session / "targets"),
        "TRACER_MONKEYPATCH_CONTROL_DIR": str(session),
        "TRACER_CUPTI_CONTROL_DIR": str(session),
        "TRACER_DEBUG_LOG_PATH": str(session / "cupti.log"),
        "TRACER_CUPTI_PRELOAD": "1" if cupti_enabled else "0",
        "TRACER_CUPTI_START_ENABLED": "0",
        "TRACER_CUPTI_START_DELAY_SEC": "0",
        "TRACER_MONKEYPATCH_START_ENABLED": "1",
        "TRACER_EBPF_START_ENABLED": "0",
        "TRACER_DRAIN_TIMEOUT_SEC": "120",
        "DE_LATENCY_DISABLE_SITECUSTOMIZE": "0" if python_enabled else "1",
        "DE_LATENCY_PYTHON_ENABLED": "1" if python_enabled else "0",
        "DE_LATENCY_EBPF_READY_FILE": str(session / "ebpf.ready"),
        "DE_LATENCY_EBPF_STATS_FILE": str(session / "ebpf.stats.json"),
        "DE_LATENCY_STOP_FILE": str(session / "stop"),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(STARTUP), str(ROOT), env.get("PYTHONPATH")])),
    })
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def control_paths(session, tool, *, worker_only=False):
    session = Path(session)
    root_pid = read_state(session).get("target_pid")
    if root_pid is None:
        return []
    if worker_only:
        target_file = session / "targets"
        pids = [line for line in target_file.read_text().splitlines() if line.isdigit()] if target_file.exists() else []
        paths = [session / f"de_latency_{tool}_{pid}.sock" for pid in sorted(set(pids), key=int)]
    else:
        paths = sorted(session.glob(f"de_latency_{tool}_*.sock"))
    # Fork/exec and os._exit can leave filesystem sockets after their listener
    # has gone away. Never control a stale/recycled PID outside this server.
    return [path for path in paths if path.is_socket() and
            is_descendant_pid(path.stem.rsplit("_", 1)[1], root_pid)]


def is_descendant_pid(pid, root_pid):
    try:
        pid, root_pid = int(pid), int(root_pid)
        while pid > 1:
            status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if status[0] == "Z":
                return False
            if pid == root_pid:
                return True
            pid = int(status[1])
    except (OSError, ValueError, TypeError, IndexError):
        pass
    return False


def control_many(paths, command):
    responses = {}
    for path in paths:
        response = send_command(str(path), command, timeout=60)
        responses[path.name] = response
        if not response.startswith("ok "):
            raise RuntimeError(f"{path.name}: {response}")
    return responses


def process_identity(pid):
    """PID start time prevents treating a recycled PID as the session owner."""
    try:
        status = Path(f"/proc/{int(pid)}/stat").read_text()
        return status.rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, TypeError, IndexError):
        return None


def session_active(manifest):
    pid, identity = manifest.get("supervisor_pid"), manifest.get("supervisor_start")
    return pid is not None and identity is not None and process_identity(pid) == identity


def wait_monitor(session, timeout=15):
    session = Path(session)
    if not (session / "ebpf.ready").exists():
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            counters = json.loads((session / "ebpf.stats.json").read_text())
            if "exit_code" in counters:
                return
        except (OSError, ValueError):
            pass  # The native process may have opened but not yet flushed it.
        if time.monotonic() >= deadline:
            raise TimeoutError("eBPF did not finish; inspect the separate eBPF terminal")
        time.sleep(0.1)
