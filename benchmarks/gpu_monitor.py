"""GPU utilization CSV consumed by tools/plot_results.py."""
import csv
from contextlib import contextmanager
import subprocess
import threading
import time


def sample():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    row = {"timestamp": time.time()}
    for line in output.splitlines():
        index, utilization, memory = (part.strip() for part in line.split(","))
        row[f"gpu{index}_util"] = utilization
        row[f"gpu{index}_mem"] = memory
    return row


@contextmanager
def monitor_gpu(path, interval=1):
    stop = threading.Event()
    failures = []
    with path.open("w", newline="") as stream:
        first = sample()
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)

        def collect():
            while not stop.wait(interval):
                try:
                    writer.writerow(sample())
                    stream.flush()
                except Exception as exc:
                    failures.append(exc)
                    return

        thread = threading.Thread(target=collect, name="gpu-monitor", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
    if failures:
        raise RuntimeError("GPU utilization sampling failed") from failures[0]
