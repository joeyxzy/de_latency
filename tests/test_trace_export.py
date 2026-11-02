import contextlib
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from de_latency.analysis.model import AnalysisResult
from de_latency.export.bundle import export_result
from de_latency.export.overview import keep_overview_event, overview_from_trace
from de_latency.export.trace_io import artifact_stem, iter_trace_events, read_json, write_trace


def sample_result():
    events = [
        {'name': 'process_name', 'ph': 'M', 'pid': 1, 'args': {'name': 'worker'}},
        {'name': '中文 request', 'cat': 'request_lifecycle', 'ph': 'X', 'pid': 11000, 'tid': 1,
         'ts': 1.125, 'dur': 2.25, 'args': {'timestamp_ns': 1793585878000000001}},
        {'name': 'de_marker:exec_start|step', 'cat': 'gpu_kernel', 'ph': 'X', 'pid': 9000, 'tid': 1, 'ts': 3, 'dur': .5},
        {'name': 'gemm_kernel', 'cat': 'gpu_kernel', 'ph': 'X', 'pid': 9000, 'tid': 1, 'ts': 4, 'dur': 2.5},
        {'name': 'exec_start|step', 'cat': 'cuda_runtime', 'ph': 'X', 'pid': 1, 'tid': 1, 'ts': 3, 'dur': .5},
        {'name': 'cudaGraphLaunch', 'cat': 'cuda_runtime', 'ph': 'X', 'pid': 1, 'tid': 1, 'ts': 2, 'dur': 1},
        {'name': 'link', 'cat': 'flow', 'ph': 's', 'pid': 1, 'tid': 1, 'ts': 2, 'id': 1000000000},
        {'name': 'link', 'cat': 'flow', 'ph': 'f', 'pid': 9000, 'tid': 1, 'ts': 4, 'id': 1000000000},
        {'name': 'PP receive', 'cat': 'pp_comm', 'ph': 'X', 'pid': 1, 'tid': 2, 'ts': 1, 'dur': .125},
        {'name': 'zero duration', 'cat': 'python', 'ph': 'X', 'pid': 1, 'tid': 2, 'ts': 9, 'dur': 0},
    ]
    return AnalysisResult(events, {'requests': {}, 'summary': {'ttft_request_count': 0}},
                          {'request_count': 0}, [], [{'name': 'alignment', 'ph': 'X', 'ts': 1, 'dur': 1}], 11000)


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def export(self, output, **kwargs):
        result = sample_result()
        before = copy.deepcopy(result)
        with contextlib.redirect_stdout(io.StringIO()):
            export_result(result, output, **kwargs)
        self.assertEqual(result, before)
        return result

    def test_full_gzip_preserves_all_events_fields_and_plain_sidecars(self):
        path = self.folder / 'trace.json.gz'
        result = self.export(path)
        self.assertEqual(path.read_bytes()[:2], b'\x1f\x8b')
        events = read_json(path)['traceEvents']
        self.assertEqual(events, sorted(result.trace_events, key=lambda event: event.get('ts', 0)))
        self.assertEqual(list(iter_trace_events(path)), events)
        self.assertTrue(any(event['name'] == 'gemm_kernel' for event in events))
        self.assertTrue(any(event.get('cat') == 'flow' for event in events))
        self.assertEqual(next(event for event in events if event['name'] == 'zero duration')['dur'], 0)
        self.assertEqual(read_json(self.folder/'trace.compute_breakdown.json'), result.compute_breakdown)
        self.assertEqual(read_json(self.folder/'trace.summary.json'), result.summary)
        self.assertEqual(read_json(self.folder/'trace.alignment.json')['traceEvents'], result.alignment_events)
        self.assertTrue((self.folder/'trace.compute_breakdown_requests/index.html').exists())
        self.assertFalse((self.folder/'trace.overview.json.gz').exists())
        self.assertFalse((self.folder/'trace.json.compute_breakdown.json').exists())

    def test_explicit_plain_json_is_still_supported(self):
        path = self.folder/'trace.json'
        self.export(path)
        self.assertEqual(path.read_bytes()[:1], b'{')
        self.assertEqual(list(iter_trace_events(path)), read_json(path)['traceEvents'])

    def test_optional_overview_keeps_markers_and_request_context_without_changing_full(self):
        path = self.folder/'trace.json.gz'
        self.export(path, overview=True)
        full = read_json(path)['traceEvents']
        view = read_json(self.folder/'trace.overview.json.gz')
        self.assertEqual(view['traceEvents'], [event for event in full if keep_overview_event(event)])
        names = {event['name'] for event in view['traceEvents']}
        self.assertIn('de_marker:exec_start|step', names)
        self.assertIn('exec_start|step', names)
        self.assertIn('PP receive', names)
        self.assertNotIn('gemm_kernel', names)
        self.assertNotIn('cudaGraphLaunch', names)
        self.assertFalse(any(event.get('ph') in ('s', 'f', 't') for event in view['traceEvents']))
        self.assertEqual(view['otherData']['view'], 'overview')

    def test_existing_plain_and_gzip_inputs_stream_identical_overviews(self):
        for suffix in ('.json', '.json.gz'):
            with self.subTest(suffix=suffix):
                path = self.folder/('input'+suffix)
                write_trace(path, sample_result().trace_events)
                before = path.read_bytes()
                # Neither the full trace nor raw log is materialized/reanalyzed.
                with patch('json.load', side_effect=AssertionError('must stream')):
                    output, count = overview_from_trace(path)
                self.assertEqual(count, sum(map(keep_overview_event, sample_result().trace_events)))
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(read_json(output)['traceEvents'],
                                 [event for event in sample_result().trace_events if keep_overview_event(event)])

    def test_failed_stream_does_not_truncate_existing_output(self):
        path = self.folder/'trace.json.gz'
        write_trace(path, sample_result().trace_events)
        before = path.read_bytes()
        def failing():
            yield {'name': 'first'}
            raise RuntimeError('input failed')
        with self.assertRaisesRegex(RuntimeError, 'input failed'):
            write_trace(path, failing())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.folder.glob('.*.tmp')), [])
        broken = self.folder/'broken.json'
        broken.write_text('{"traceEvents":[{"name":"valid"},')
        with self.assertRaises(ValueError):
            overview_from_trace(broken, path)
        self.assertEqual(path.read_bytes(), before)

    def test_overview_cannot_overwrite_source_even_through_symlink(self):
        source = self.folder/'trace.json'
        write_trace(source, [])
        alias = self.folder/'alias.json'
        alias.symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            overview_from_trace(source, alias)

    def test_stream_reader_validates_root_and_allows_empty_traces(self):
        path = self.folder/'trace.json'
        for value in ([], {'other': []}, {'traceEvents': {}}, {'traceEvents': [1]}):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(iter_trace_events(path))
        path.write_text(json.dumps({'otherData': {'traceEvents': 'not the array'}, 'traceEvents': []}))
        self.assertEqual(list(iter_trace_events(path)), [])

    def test_artifact_stems_do_not_include_json_or_gzip_extensions(self):
        for name in ('trace.json', 'trace.json.gz'):
            self.assertEqual(artifact_stem(self.folder/name), self.folder/'trace')
