import atexit
import logging
import os
import socket
import threading


_logger = logging.getLogger("de_latency_monkeypatch_control")
_lock = threading.Lock()
_server_thread = None
_server_socket = None
_server_started = False
_server_start_failed_pid = 0
_stop_event = threading.Event()
_control_dir = os.getenv("TRACER_MONKEYPATCH_CONTROL_DIR", "/tmp")
_process_pid = os.getpid()
_socket_path = os.path.join(_control_dir, f"de_latency_monkeypatch_{_process_pid}.sock")
_poll_timeout_s = max(0.05, float(os.getenv("TRACER_MONKEYPATCH_CONTROL_POLL_TIMEOUT_S", "0.5")))
_backlog = max(1, int(os.getenv("TRACER_MONKEYPATCH_CONTROL_BACKLOG", "8")))
_enabled = os.getenv("TRACER_MONKEYPATCH_START_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_generation = 1 if _enabled else 0


def _socket_path_for_pid(pid):
    return os.path.join(_control_dir, f"de_latency_monkeypatch_{pid}.sock")


def _reset_after_fork_locked(pid):
    global _server_socket, _server_thread, _server_started, _server_start_failed_pid
    global _process_pid, _socket_path

    inherited_socket = _server_socket
    _server_socket = None
    _server_thread = None
    _server_started = False
    _server_start_failed_pid = 0
    _process_pid = pid
    _socket_path = _socket_path_for_pid(pid)
    _stop_event.clear()

    if inherited_socket is not None:
        try:
            inherited_socket.close()
        except OSError:
            pass


def _refresh_process_identity():
    pid = os.getpid()
    if pid == _process_pid:
        return
    with _lock:
        if pid != _process_pid:
            _reset_after_fork_locked(pid)


def _after_fork_child():
    _reset_after_fork_locked(os.getpid())


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def is_enabled():
    _refresh_process_identity()
    if not _server_started and _server_start_failed_pid != os.getpid():
        try:
            start_control_server()
        except OSError as exc:
            _logger.warning("monkeypatch control start failed in pid=%s: %s", os.getpid(), exc)
    with _lock:
        return _enabled


def get_socket_path():
    _refresh_process_identity()
    return _socket_path


def get_status():
    _refresh_process_identity()
    with _lock:
        return {
            "enabled": _enabled,
            "generation": _generation,
            "pid": os.getpid(),
            "socket": _socket_path,
        }


def status_line(prefix="ok"):
    state = get_status()
    return (
        f"{prefix} state={'on' if state['enabled'] else 'off'} "
        f"generation={state['generation']} pid={state['pid']} socket={state['socket']}\n"
    )


def set_enabled(enabled, reason="control"):
    global _enabled, _generation

    with _lock:
        changed = (_enabled != enabled)
        _enabled = enabled
        if enabled and changed:
            _generation += 1
        generation = _generation

    _logger.info(
        "[%s] monkeypatch tracing %s (generation=%s)",
        reason,
        "enabled" if enabled else "disabled",
        generation,
    )
    return changed


def _handle_command(command):
    text = (command or "").strip().lower()
    if not text or text == "status":
        return status_line("ok")
    if text in {"on", "enable"}:
        set_enabled(True, reason="control:on")
        return status_line("ok")
    if text in {"off", "disable"}:
        set_enabled(False, reason="control:off")
        return status_line("ok")
    return "error=unknown_command\n"


def _server_loop():
    while not _stop_event.is_set():
        try:
            assert _server_socket is not None
            _server_socket.settimeout(_poll_timeout_s)
            conn, _ = _server_socket.accept()
        except socket.timeout:
            continue
        except OSError as exc:
            if _stop_event.is_set():
                break
            _logger.warning("monkeypatch control accept failed: %s", exc)
            break

        with conn:
            try:
                data = conn.recv(256)
                response = _handle_command(data.decode("utf-8", errors="replace"))
                conn.sendall(response.encode("utf-8"))
            except OSError as exc:
                _logger.warning("monkeypatch control client handling failed: %s", exc)


def start_control_server():
    global _server_socket, _server_thread, _server_started, _server_start_failed_pid

    _refresh_process_identity()
    with _lock:
        if _server_started:
            return _socket_path

        try:
            os.makedirs(_control_dir, exist_ok=True)
        except Exception:
            pass

        try:
            os.unlink(_socket_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(_socket_path)
            server.listen(_backlog)
            os.chmod(_socket_path, 0o600)
        except OSError:
            _server_start_failed_pid = os.getpid()
            server.close()
            raise

        _stop_event.clear()
        _server_socket = server
        _server_thread = threading.Thread(
            target=_server_loop,
            name="MonkeyPatchControl",
            daemon=True,
        )
        _server_thread.start()
        _server_started = True
        _server_start_failed_pid = 0
        _logger.info("monkeypatch control listening on %s", _socket_path)
        return _socket_path


def stop_control_server():
    global _server_socket, _server_thread, _server_started

    with _lock:
        if not _server_started:
            return
        _stop_event.set()
        server = _server_socket
        thread = _server_thread
        _server_socket = None
        _server_thread = None
        _server_started = False

    if server is not None:
        try:
            server.close()
        except OSError:
            pass

    if thread is not None:
        thread.join(timeout=max(1.0, _poll_timeout_s * 4))

    try:
        os.unlink(_socket_path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


atexit.register(stop_control_server)
