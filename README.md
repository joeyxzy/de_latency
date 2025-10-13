# CPU&GPU时间分解

v0.1

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

## eBPF

实现了追踪装载程序所在线程组（TGID）所有线程 （TID）被调度到CPU上运行的时间

```shell
#src/CPU_demo 构建探测程序
make
#build/ 运行程序样例，可以更换样例
sudo ./cpu_time_user /home/joeyxzy/de_latency/cuda-samples/Samples/0_Introduction/simpleMultiCopy/simpleMultiCopy

```

TODO:

ebpf骨架是什么意思，之前的rodata什么问题，attach顺序又是什么问题，为什么
