"""The user CLI must preserve workloads and own only capture/service lifecycle."""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from de_latency import cli
from de_latency.runtime.session import (
    ROOT, STARTUP, control_paths, health_url, process_identity, session_environment, vllm_arguments,
)


class SessionCliTests(unittest.TestCase):
    def test_server_options_are_forwarded_not_replaced_with_smoke_defaults(self):
        options = ["--", "~/models/Qwen3-32B", "--served-model-name", "qwen3-32b", "--port", "8000",
                   "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--max-model-len", "8192",
                   "--max-num-seqs", "32", "--max-num-batched-tokens", "2048", "--enable-chunked-prefill",
                   "--no-enable-prefix-caching"]
        args = cli.parser().parse_args(["serve", "--session", "runs/test", "--gpu", "0,1,2,3", *options])
        self.assertEqual(args.python_capture, "on")
        self.assertEqual(args.cupti, "on")
        self.assertEqual(vllm_arguments(args.server_args), ["--model", str(Path.home() / "models/Qwen3-32B"), *options[2:]])
        self.assertEqual(health_url(vllm_arguments(args.server_args)), "http://127.0.0.1:8000/health")

    def test_startup_component_switches_are_independent(self):
        for python_enabled in (True, False):
            for cupti_enabled in (True, False):
                with self.subTest(python=python_enabled, cupti=cupti_enabled):
                    env = session_environment('/tmp/opencode/session', python_enabled=python_enabled,
                        cupti_enabled=cupti_enabled, base={"CUDA_VISIBLE_DEVICES": "4,5"})
                    self.assertEqual(env["DE_LATENCY_PYTHON_ENABLED"], "1" if python_enabled else "0")
                    self.assertEqual(env["DE_LATENCY_DISABLE_SITECUSTOMIZE"], "0" if python_enabled else "1")
                    self.assertEqual(env["TRACER_CUPTI_PRELOAD"], "1" if cupti_enabled else "0")
                    self.assertEqual(env["TRACER_CUPTI_START_ENABLED"], "0")
                    self.assertEqual(env["TRACER_EBPF_START_ENABLED"], "0")
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5")
                    self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[:2], [str(STARTUP), str(ROOT)])

    def test_own_preload_is_not_inherited_by_collector_or_disabled_session(self):
        base = {"LD_PRELOAD": str(ROOT / 'build/libtracer.so') + ':/other/user-library.so'}
        env = session_environment('/tmp/opencode/session', cupti_enabled=False, base=base)
        self.assertEqual(env['LD_PRELOAD'], '/other/user-library.so')
        self.assertIn('libtracer.so', base['LD_PRELOAD'])

    def test_analyze_does_not_read_a_live_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'manifest.json').write_text(json.dumps({"supervisor_pid": os.getpid(),
                "supervisor_start": process_identity(os.getpid()), "status": "ready"}))
            with patch('de_latency.analysis.pipeline.process_logs') as process, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['analyze', tmp]), 1)
            process.assert_not_called()

    def test_analyze_defaults_to_full_gzip_and_overview_is_additional(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'events.log.stats.json').write_text('{}')
            for option in ([], ['--overview']):
                with patch('de_latency.analysis.pipeline.process_logs') as process, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(['analyze', tmp, *option]), 0)
                process.assert_called_once_with(str(folder / 'events.log'), str(folder / 'trace.json.gz'), overview=bool(option))

    def test_existing_trace_overview_does_not_reanalyze_or_modify_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'trace.json'
            source.write_text(json.dumps({'traceEvents': [
                {'ph': 'X', 'cat': 'request_lifecycle', 'name': 'request', 'ts': 1, 'dur': 2},
                {'ph': 'X', 'cat': 'gpu_kernel', 'name': 'gemm', 'ts': 1, 'dur': 1},
            ]}))
            before = source.read_bytes()
            with patch('de_latency.analysis.pipeline.process_logs') as process, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['overview', tmp]), 0)
            process.assert_not_called()
            self.assertEqual(source.read_bytes(), before)
            with gzip.open(folder / 'trace.overview.json.gz', 'rt') as stream:
                self.assertEqual(len(json.load(stream)['traceEvents']), 1)

    def test_stale_cupti_sockets_are_not_discovered_as_live_workers(self):
        with tempfile.TemporaryDirectory(prefix='de-sockets-') as tmp:
            folder = Path(tmp)
            (folder / 'control.state').write_text(f'target_pid={os.getpid()}\n')
            (folder / 'targets').write_text(f'{os.getpid()}\n999999999\ntid:123\n')
            with socket.socket(socket.AF_UNIX) as live, socket.socket(socket.AF_UNIX) as stale:
                live_path = folder / f'de_latency_cupti_{os.getpid()}.sock'
                live.bind(str(live_path))
                stale.bind(str(folder / 'de_latency_cupti_999999999.sock'))
                self.assertEqual(control_paths(folder, 'cupti'), [live_path])
                self.assertEqual(control_paths(folder, 'cupti', worker_only=True), [live_path])

    def test_ebpf_only_escalates_native_monitor_and_sets_session_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'manifest.json').write_text(json.dumps({"python_capture": True, "supervisor_pid": os.getpid(),
                "supervisor_start": process_identity(os.getpid()), "status": "ready"}))
            (folder / 'target.pid').write_text(str(os.getpid()))
            with patch('de_latency.cli.os.geteuid', return_value=1000), patch('de_latency.cli.os.execvp') as execute:
                cli.ebpf(cli.parser().parse_args(['ebpf', '--session', tmp]))
            executable, command = execute.call_args.args
            self.assertEqual(executable, 'sudo')
            self.assertEqual(command[:2], ['sudo', 'env'])
            self.assertIn(str(ROOT / 'build/ebpf_monitor'), command)
            self.assertIn('TRACER_ZMQ_ADDR=ipc://' + str(folder / 'events.sock'), command)
            self.assertNotIn('serve', command)
            self.assertNotIn('LD_PRELOAD', ' '.join(command))

    def test_ebpf_without_python_stays_within_the_server_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'manifest.json').write_text(json.dumps({"python_capture": False, "supervisor_pid": os.getpid(),
                "supervisor_start": process_identity(os.getpid()), "status": "ready"}))
            (folder / 'target.pid').write_text(str(os.getpid()))
            with patch('de_latency.cli.os.geteuid', return_value=1000), patch('de_latency.cli.os.execvp') as execute:
                cli.ebpf(cli.parser().parse_args(['ebpf', '--session', tmp]))
            command = execute.call_args.args[1]
            self.assertNotIn('--worker-pid-file', command)
            self.assertEqual(command[-2:], ['--worker-pattern', ''])
            self.assertEqual(command[command.index('--root-pid')+1], str(os.getpid()))


@unittest.skipUnless((ROOT / 'build/controller').exists(), 'run make first')
class SessionLifecycleTests(unittest.TestCase):
    def test_service_client_and_analysis_lifecycles_remain_separate(self):
        with tempfile.TemporaryDirectory(prefix='de-cli-') as tmp:
            folder = Path(tmp)
            (folder / 'build').mkdir()
            (folder / 'build/controller').symlink_to(ROOT / 'build/controller')
            for package in ('vllm', 'vllm/entrypoints', 'vllm/entrypoints/openai'):
                (folder / package).mkdir(exist_ok=True)
                (folder / package / '__init__.py').write_text('')
            # This fake vLLM serves HTTP without CUDA or changing any user workload.
            (folder / 'vllm/entrypoints/openai/api_server.py').write_text('''
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--port', type=int)
p.add_argument('--model')
args, extra=p.parse_known_args()
assert os.environ['DE_LATENCY_PYTHON_ENABLED']=='0'
assert 'libtracer.so' not in os.environ.get('LD_PRELOAD','')
Path(os.environ['DE_LATENCY_LOG_PATH']).with_name('server.argv.json').write_text(json.dumps(sys.argv[1:]))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
''')
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            session = folder / 'capture'
            entry = f'''
from pathlib import Path
from de_latency import cli
from de_latency.instrumentation import bootstrap
bootstrap.check_version=lambda: '0.20.0'
cli.ROOT=Path({str(folder)!r})
raise SystemExit(cli.main(['serve','--session',{str(session)!r},'--python','off','--cupti','off',
    '--','fake-model','--port',{str(port)!r},'--max-model-len','8192']))
'''
            proc = subprocess.Popen([sys.executable, '-u', '-c', entry], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, 'DE_LATENCY_DISABLE_SITECUSTOMIZE':'1', 'PYTHONDONTWRITEBYTECODE':'1',
                    'PYTHONPATH': os.pathsep.join(filter(None, [str(folder), os.environ.get('PYTHONPATH')]))})
            try:
                deadline=time.monotonic()+10
                while True:
                    path=session/'manifest.json'
                    if path.exists() and json.loads(path.read_text()).get('status')=='ready':
                        break
                    if proc.poll() is not None or time.monotonic()>deadline:
                        self.fail('service CLI failed to become ready')
                    time.sleep(.05)
                # An external client uses the requested port, with no runner involvement.
                import urllib.request
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                    self.assertEqual(response.read(),b'ok')
                proc.send_signal(signal.SIGINT)
                output,_=proc.communicate(timeout=10)
                self.assertEqual(proc.returncode,0,output)
                manifest=json.loads((session/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'finished',manifest)
                self.assertEqual(manifest['server_command'][2:4], ['-m', 'de_latency.instrumentation.launch'])
                self.assertEqual(json.loads((session/'server.argv.json').read_text()),
                                 ['--model','fake-model','--port',str(port),'--max-model-len','8192'])
                self.assertTrue((session/'events.log.stats.json').exists())
                self.assertTrue((session/'stop').exists())
                self.assertFalse((session/'benchmark.json').exists())
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(['analyze',str(session)]),0)
                self.assertTrue((session/'trace.json.gz').exists())
                with gzip.open(session/'trace.json.gz', 'rt') as stream:
                    self.assertIn('traceEvents', json.load(stream))
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    proc.communicate(timeout=10)
