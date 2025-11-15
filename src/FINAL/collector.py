import zmq
import json

SOCK_ADDR = "ipc:///tmp/tracer.sock"

def collector():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(SOCK_ADDR)

    print("[collector] started, listening on", SOCK_ADDR)

    while True:
        frames = sock.recv_multipart()   # 支持 1 到 N 帧

        if len(frames) == 1:
            # 单帧 JSON → CUPTI / eBPF
            meta_json = frames[0].decode()
            meta = json.loads(meta_json)
            payload = None
            print("[CUPTI/eBPF]", meta)

        elif len(frames) == 2:
            # Python multipart → meta + payload
            meta_json = frames[0].decode()
            payload = frames[1]      # 二进制
            meta = json.loads(meta_json)
            print("[PYTHON]", meta, "payload_size=", len(payload))

        else:
            # 不期望出现多帧
            print("[ERROR] Unexpected multipart size:", len(frames))
            continue

        # 你可以在这里进一步处理 meta + payload
        # process(meta, payload)

if __name__ == "__main__":
    collector()
