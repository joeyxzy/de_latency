"""Package-local launch and opt-in startup must work without legacy paths."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from de_latency.instrumentation import launch
from de_latency.runtime.session import ROOT, STARTUP, session_environment


class LayoutTests(unittest.TestCase):
    def test_launch_preserves_server_arguments_and_installs_request_hooks(self):
        arguments = ['--model', '/models/example', '--tensor-parallel-size', '2', '--max-model-len', '8192']
        with patch.dict(os.environ, {'DE_LATENCY_PYTHON_ENABLED': '1'}), \
             patch.object(sys, 'argv', ['launch', *arguments]), \
             patch('de_latency.instrumentation.bootstrap.install_request_hooks') as install, \
             patch.object(launch.runpy, 'run_module') as run:
            launch.main()
            install.assert_called_once_with()
            run.assert_called_once_with('vllm.entrypoints.openai.api_server', run_name='__main__')
            self.assertEqual(sys.argv, ['vllm.entrypoints.openai.api_server', *arguments])

    def test_disabled_python_launch_does_not_install_hooks(self):
        with patch.dict(os.environ, {'DE_LATENCY_PYTHON_ENABLED': '0'}), \
             patch.object(sys, 'argv', ['launch', '--model', 'example']), \
             patch('de_latency.instrumentation.bootstrap.install_request_hooks') as install, \
             patch.object(launch.runpy, 'run_module'):
            launch.main()
            install.assert_not_called()

    def test_startup_hook_honors_capture_switch(self):
        for disabled in ('0', '1'):
            with self.subTest(disabled=disabled), \
                 patch.dict(os.environ, {'DE_LATENCY_DISABLE_SITECUSTOMIZE': disabled}), \
                 patch('de_latency.instrumentation.bootstrap.install_worker_hooks') as install:
                runpy.run_path(str(STARTUP / 'sitecustomize.py'))
                self.assertEqual(install.call_count, int(disabled == '0'))

    def test_startup_directory_is_discovered_by_parent_and_spawn_child(self):
        with tempfile.TemporaryDirectory(prefix='de-layout-') as tmp:
            folder = Path(tmp)
            script = folder / 'spawn_check.py'
            script.write_text('''
import json, multiprocessing, sys
def child(queue):
    queue.put(sys.modules['sitecustomize'].__file__)
if __name__ == '__main__':
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    print(json.dumps([sys.modules['sitecustomize'].__file__, queue.get(timeout=10)]))
    process.join(timeout=10)
    assert process.exitcode == 0
    queue.close()
''')
            env = session_environment(folder, python_enabled=False, cupti_enabled=False)
            completed = subprocess.run([sys.executable, str(script)], cwd=folder, env=env,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual([Path(name).resolve() for name in json.loads(completed.stdout)],
                             [(STARTUP / 'sitecustomize.py').resolve()] * 2)

    def test_control_modules_offer_help_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix='de-layout-') as tmp:
            env = {**os.environ, 'DE_LATENCY_DISABLE_SITECUSTOMIZE': '1',
                   'PYTHONPATH': str(ROOT)}
            for module in ('de_latency.runtime.control_cli', 'de_latency.runtime.cupti_cli'):
                completed = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp,
                                           env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn('usage:', completed.stdout)
