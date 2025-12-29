# vLLM框架追踪

v0.2

```shell
de_latency/
├── src/
│   └── CPU_demo/
│       ├── cpu_time.bpf.c
│       ├── cpu_time_user.c
│       └── Makefile       # CPU_demo 独立 Makefile
├── include/
│   └── libbpf/bpf
|   └── vmlinux.h
├── lib/
│   ├── libbpf.so -> libbpf.so.1.3.0
│   └── libbpf.so.1.3.0
├── build/                 # 所有模块的 build 输出
└── Makefile               # 顶层 Makefile（可选，调用子模块 Makefile）

```

## 使用

```shell
# 启动vLLM服务，同时patch我们的追踪程序
python start_server.py     --model Qwen/Qwen1.5-4B-Chat     --tensor-parallel-size 2     --dtype auto     --max-model-len 4096     --port 8000


#向启动的服务发送请求测试
python request_test.py

#使用v0.1的trace追踪并启动vllm
sudo -E ./controller /home/joeyxzy/miniconda3/envs/vllm/bin/python /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py     --model Qwen/Qwen1.5-4B-Chat     --tensor-parallel-size 2     --dtype auto     --max-model-len 4096     --port 8000

# 在sudo下抹去用户环境变量时
sudo LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib ./controller /home/joeyxzy/de_latency/cuda-samples/Samples/0_Introduction/simpleMultiGPU/simpleMultiGPU

sudo -E LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib ./controller /home/joeyxzy/miniconda3/envs/vllm/bin/python /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py     --model Qwen/Qwen1.5-4B-Chat     --tensor-parallel-size 2     --dtype auto     --max-model-len 4096     --port 8000

# 启动sitecustomize
sudo -E   PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace  LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib   ./controller   /home/joeyxzy/miniconda3/envs/vllm/bin/python   /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py   --model Qwen/Qwen1.5-4B-Chat   --tensor-parallel-size 2   --dtype auto   --max-model-len 4096   --port 8000

#将直接输出的日志转换为trace.json，供perffeto查看
python log_to_trace.py /home/joeyxzy/de_latency/de_latency/perffeto/de_latency.log trace.json

#启动bench的指令
vllm bench latency     --model Qwen/Qwen1.5-4B-Chat     --input-len 32     --output-len 1     --max-model-len 32300     --enforce-eager     --load-format dummy

#启动bench同时trace的指令
#应该是需要打开注释的
sudo -E \
PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace \
LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib \
./controller \
/home/joeyxzy/miniconda3/envs/vllm/bin/python \
/home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py \
--model Qwen/Qwen1.5-4B-Chat \
--input-len 32 \
--output-len 1 \
--max-model-len 32300 \
--enforce-eager \
--load-format dummy \
--tensor-parallel-size 2
```

## 新增环境

- ZeroMQ

```shell
#c环境依赖
sudo apt-get install -y libzmq3-dev

#python的环境依赖
pip install pyzmq


```
