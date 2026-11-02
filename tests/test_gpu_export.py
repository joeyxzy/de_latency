from collections import Counter
import unittest

from de_latency.analysis.correlation import cupti_corr_key, kernels_for_launch
from de_latency.export.gpu import append_gpu_trace
from de_latency.export.perfetto import PerfettoWriter


class GpuExportTests(unittest.TestCase):
    def test_process_scoped_ids_and_graph_flows(self):
        base = 1_800_000_000_000_000_000
        kernels = [dict(pid=pid, deviceId=0, streamId=7, start_ns=base + n,
                        end_ns=base + n + 1, correlationId=42, name="kernel")
                   for pid, n in ((1, 10), (1, 20), (2, 30))]
        index = {(1, 42): kernels[:2], (2, 42): kernels[2:]}
        apis = [dict(pid=pid, tid=pid, start_ns=base, end_ns=base + 5,
                     correlationId=42, name="cudaGraphLaunch") for pid in (1, 2)]
        events = []
        append_gpu_trace(events, PerfettoWriter(base), apis, index, kernels)
        flows = Counter(event["id"] for event in events if event.get("ph") in ("s", "f"))
        self.assertEqual(len(flows), 3)
        self.assertEqual(set(flows.values()), {2})
        tracks = {event["pid"] for event in events if event.get("cat") == "gpu_kernel"}
        self.assertEqual(len(tracks), 2)
        names = [event for event in events if event.get("name") == "thread_name"]
        self.assertEqual(len(names), 2)  # once per stream, not once per kernel
        self.assertEqual(len(kernels_for_launch(index, apis[0], 1)), 2)
        self.assertEqual(cupti_corr_key(apis[0]), (1, 42))

    def test_reused_correlation_epoch_and_unknown_host_tid(self):
        api = dict(pid=1, tid=0, start_ns=50_000_000_000, end_ns=50_000_000_010,
                   correlationId=2, name="launch")
        old = dict(pid=1, start_ns=1, end_ns=2, correlationId=2)
        self.assertEqual(kernels_for_launch({(1, 2): [old]}, api, 1), [])
        events = []
        append_gpu_trace(events, PerfettoWriter(0), [api, dict(api)], {}, [])
        apis = [event for event in events if event.get("cat") == "cuda_runtime"]
        self.assertNotEqual(apis[0]["tid"], apis[1]["tid"])
