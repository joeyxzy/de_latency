"""Line-oriented Unix control protocol shared by both public control CLIs."""
import os
import socket


def control_path(tool, pid):
    directory = os.getenv(f"TRACER_{tool.upper()}_CONTROL_DIR", "/tmp")
    return os.path.join(directory, f"de_latency_{tool}_{pid}.sock")


def send_command(path, command, timeout=2.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall((command + "\n").encode())
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()
