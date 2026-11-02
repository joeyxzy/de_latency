"""Exercise native control transitions with real CUDA work, no vLLM model."""
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.validate import wait_for, stop_process
from de_latency.protocol import iter_log
from de_latency.runtime.socket_client import send_command

WORKER = r'''
import ctypes
import sys
import torch
torch.cuda.init()
mark = ctypes.CDLL(None).de_marker_push
mark.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
mark.restype = ctypes.c_int
print("ready", flush=True)
for command in sys.stdin:
    tensor = torch.ones(1024, device="cuda")
    tensor.add_(1)
    assert mark(int(torch.cuda.current_stream().cuda_stream), command.strip().encode()) == 0
    torch.cuda.synchronize()
    print("done", flush=True)
'''


def main():
    with tempfile.TemporaryDirectory(prefix="de-control-") as tmp:
        folder = Path(tmp)
        env = {**os.environ, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1",
               "TRACER_ZMQ_ADDR": f"ipc://{folder}/events.sock",
               "DE_LATENCY_LOG_PATH": str(folder / "events.log"),
               "TRACER_WORKER_PID_FILE": str(folder / "targets"),
               "TRACER_CUPTI_CONTROL_DIR": str(folder), "TRACER_CUPTI_START_ENABLED": "0",
               "TRACER_DEBUG_LOG_PATH": str(folder / "cupti.log")}
        plain = dict(env)
        plain.pop("LD_PRELOAD", None)
        worker = collector = None
        with (folder / "collector.log").open("w") as output:
            try:
                collector = subprocess.Popen([sys.executable, "-m", "de_latency.collector"],
                    env=plain, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                wait_for(lambda: (folder / "events.sock").exists(), 15, "collector", [collector])
                env["LD_PRELOAD"] = str(ROOT / "build/libtracer.so")
                worker = subprocess.Popen([sys.executable, "-u", "-c", WORKER], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
                assert worker.stdout.readline().strip() == "ready"
                path = str(folder / f"de_latency_cupti_{worker.pid}.sock")

                def control(command, state, markers):
                    response = send_command(path, command, 60)
                    assert response.startswith("ok "), response
                    fields = dict(word.split("=", 1) for word in response.split() if "=" in word)
                    assert fields["state"] == state and fields["markers_only"] == markers, response
                    for key in ("ring_dropped", "zmq_dropped", "cupti_dropped"):
                        assert fields[key] == "0", response

                def work(label):
                    worker.stdin.write(label + "\n")
                    worker.stdin.flush()
                    assert worker.stdout.readline().strip() == "done"

                control("status", "off", "no")
                # Clients are allowed to disconnect without consuming a reply.
                for _ in range(3):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(path)
                        client.sendall(b"status\n")
                control("status", "off", "no")
                control("markers_only on", "off", "yes")
                control("markers_only off", "off", "no")
                work("disabled")
                control("on", "on", "no")
                work("normal")
                control("flush", "on", "no")
                control("off", "off", "no")
                control("markers_only on", "off", "yes")
                control("on", "on", "yes")
                work("markers")
                control("markers_only off", "on", "no")
                work("restored")
                control("off", "off", "no")
                # An accepted client that never sends must not pin shutdown.
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as idle:
                    idle.connect(path)
                    time.sleep(0.1)
                    worker.stdin.close()
                    assert worker.wait(timeout=5) == 0
            finally:
                stop_process(worker)
                stop_process(collector)
        events = list(iter_log(folder / "events.log", strict=True))
        names = {event["payload"].get("name") for event in events if event["event_type"] == "marker"}
        assert {"normal", "markers", "restored"} <= names and "disabled" not in names, names
        assert any(event["event_type"] == "kernel" for event in events)
        assert any(event["event_type"] == "runtime" for event in events)
        stats = json.loads((folder / "events.log.stats.json").read_text())
        assert stats["invalid_events"] == 0
        print(json.dumps({"passed": True, "events": len(events), "marker_labels": sorted(names)}))


if __name__ == "__main__":
    main()
