#!/usr/bin/env python3
import argparse
import os
import socket
import sys


def default_socket_path(pid: int) -> str:
    control_dir = os.getenv("TRACER_CUPTI_CONTROL_DIR", "/tmp")
    return os.path.join(control_dir, f"de_latency_cupti_{pid}.sock")


def build_command(args: argparse.Namespace) -> str:
    if args.command == "off" and args.mode == "fast":
        return "off fast"
    return args.command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control libtracer CUPTI runtime state for a target PID.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="Target process PID.")
    target.add_argument("--socket-path", help="Explicit control socket path.")
    parser.add_argument(
        "command",
        choices=["status", "on", "off", "flush"],
        help="Control command to send.",
    )
    parser.add_argument(
        "--mode",
        choices=["graceful", "fast"],
        default="graceful",
        help="Only applies to 'off'.",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    socket_path = args.socket_path or default_socket_path(args.pid)
    command = build_command(args)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(args.timeout)
            sock.connect(socket_path)
            sock.sendall((command + "\n").encode("utf-8"))
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
    except FileNotFoundError:
        print(f"control socket not found: {socket_path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"failed to talk to {socket_path}: {exc}", file=sys.stderr)
        return 1

    response = b"".join(chunks).decode("utf-8", errors="replace").strip()
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
