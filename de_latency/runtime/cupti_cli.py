"""Low-level CUPTI control: python -m de_latency.runtime.cupti_cli."""
import argparse
import sys
from .socket_client import control_path, send_command


def default_socket_path(pid: int) -> str:
    return control_path("cupti", pid)


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
    parser.add_argument("--timeout", type=float, default=60.0, help="Socket timeout in seconds (includes graceful drain).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    socket_path = args.socket_path or default_socket_path(args.pid)
    command = build_command(args)

    try:
        response = send_command(socket_path, command, args.timeout)
    except FileNotFoundError:
        print(f"control socket not found: {socket_path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"failed to talk to {socket_path}: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0 if response.startswith("ok ") else 1


if __name__ == "__main__":
    raise SystemExit(main())
