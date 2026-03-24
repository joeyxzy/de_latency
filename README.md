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

#避开端口和GPU冲突，指定使用本地下载好的模型，不同担心HF访问问题
CUDA_VISIBLE_DEVICES=1,2 sudo -E   PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace  LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib   ./controller   /home/joeyxzy/miniconda3/envs/vllm/bin/python   /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py   --model /home/joeyxzy/models/Qwen1.5-4B-Chat   --tensor-parallel-size 2   --dtype auto   --max-model-len 4096   --port 8001

#开单卡启动
sudo -E   PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace  LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib   ./controller   /home/joeyxzy/miniconda3/envs/vllm/bin/python   /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py   --model /home/joeyxzy/models/Qwen1.5-4B-Chat   --tensor-parallel-size 1   --dtype auto   --max-model-len 4096   --port 8001

#将直接输出的日志转换为trace.json，供perfetto查看
python log_to_trace.py de_latency.log trace.json

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

# 使用控制参数切换 vLLM 启动模式（start_server.py）
# 说明：
# 1) 默认模式是 serve（不传 --mode 也会走 serve）
# 2) --mode bench 时，可用 --bench-subcommand 指定 bench 子命令
# 3) 也支持环境变量：DE_LATENCY_VLLM_MODE / DE_LATENCY_BENCH_SUBCOMMAND

# A. serve 模式（默认）
sudo -E \
PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace \
LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib \
./controller \
/home/joeyxzy/miniconda3/envs/vllm/bin/python \
/home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py \
--model /home/joeyxzy/models/Qwen1.5-4B-Chat \
--tensor-parallel-size 1 \
--dtype auto \
--max-model-len 4096 \
--port 8001

# B. bench latency 模式
sudo -E \
PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace \
LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib \
./controller \
/home/joeyxzy/miniconda3/envs/vllm/bin/python \
/home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py \
--mode bench \
--bench-subcommand latency \
--model /home/joeyxzy/models/Qwen1.5-4B-Chat \
--tensor-parallel-size 1 \
--input-len 32 \
--output-len 1 \
--max-model-len 4096 \
--enforce-eager \
--load-format dummy

# C. bench serve 模式
sudo -E \
PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace \
LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib \
./controller \
/home/joeyxzy/miniconda3/envs/vllm/bin/python \
/home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py \
--mode bench \
--bench-subcommand serve \
--backend vllm \
--model /home/joeyxzy/models/Qwen1.5-4B-Chat \
--host 127.0.0.1 \
--port 8001 \
--endpoint /v1/completions \
--dataset-name random \
--num-prompts 200

# D. 环境变量方式（等价于 --mode bench --bench-subcommand serve）
DE_LATENCY_VLLM_MODE=bench \
DE_LATENCY_BENCH_SUBCOMMAND=serve \
sudo -E \
PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace \
LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib \
./controller \
/home/joeyxzy/miniconda3/envs/vllm/bin/python \
/home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py \
--backend vllm \
--model /home/joeyxzy/models/Qwen1.5-4B-Chat \
--host 127.0.0.1 \
--port 8001 \
--endpoint /v1/completions \
--dataset-name random \
--num-prompts 200

#全流程使用总结：
#编译
make clean && make

# 终端A：跑服务程序并写日志（collector 会写 perfetto/de_latency.log）
# controller 会自动拉起 ebpf_monitor，并自动发现 worker 进程开始调度追踪
# worker PID 由 monkey_patch 的 worker_process_ready 事件自动发现，不依赖进程名匹配
cd /de_latency/build
sudo -E   PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace  LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib   ./controller   /home/joeyxzy/miniconda3/envs/vllm/bin/python   /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py   --model /home/joeyxzy/models/Qwen1.5-4B-Chat   --tensor-parallel-size 1   --dtype auto   --max-model-len 4096   --port 8001

# 手动单独运行（可选）：
# 自动模式：sudo ./ebpf_monitor
# 手动模式：sudo ./ebpf_monitor <worker_tid>

# 终端C：将日志转换为 trace.json
cd /de_latency/perfetto
python log_to_trace.py de_latency.log trace.json

#benchmarks
#客户端压测：
#1.sharegpt
/home/joeyxzy/miniconda3/envs/vllm/bin/vllm bench serve   --backend vllm   --model /home/joeyxzy/models/Qwen1.5-4B-Chat   --host 127.0.0.1   --port 8001   --endpoint /v1/completions   --dataset-name sharegpt  --dataset-path /home/joeyxzy/vllm_bench/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200   --ready-check-timeout-sec 30
#2.random
/home/joeyxzy/miniconda3/envs/vllm/bin/vllm bench serve   --backend vllm   --model /home/joeyxzy/models/Qwen1.5-4B-Chat   --host 127.0.0.1   --port 8001   --endpoint /v1/completions   --dataset-name random   --num-prompts 200   --ready-check-timeout-sec 30

#服务端启动：
#1.使用trace工具
sudo -E PYTHONPATH=/home/joeyxzy/de_latency/de_latency/vllm_trace LD_LIBRARY_PATH=/home/joeyxzy/zeromq_install/lib:/home/joeyxzy/jsonc_install/lib ./controller /home/joeyxzy/miniconda3/envs/vllm/bin/python /home/joeyxzy/de_latency/de_latency/vllm_trace/start_server.py --model /home/joeyxzy/models/Qwen1.5-4B-Chat --tensor-parallel-size 1 --dtype auto --max-model-len 4096 --port 8001
#2.纯净启动：
/home/joeyxzy/miniconda3/envs/vllm/bin/vllm serve /home/joeyxzy/models/Qwen1.5-4B-Chat   --host 127.0.0.1   --port 8001   --tensor-parallel-size 1   --dtype auto   --max-model-len 4096

```

## 新增环境

- ZeroMQ

```shell
#c环境依赖
sudo apt-get install -y libzmq3-dev

#python的环境依赖
pip install pyzmq


```
