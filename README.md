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

GPU2机器上的编译流程

```shell
#编译内核态程序
clang -O2 -g -target bpf -c cpu_time.bpf.c -o cpu_time.bpf.o \
  -I . \
  -I /usr/src/linux-headers-6.8.0-78-generic/tools/bpf/resolve_btfids/libbpf/include

#编译用户态程序
gcc cpu_time_user.c -o cpu_time_user \
  -I/usr/src/linux-headers-6.8.0-78-generic/tools/bpf/resolve_btfids/libbpf/include \
  /lib/x86_64-linux-gnu/libbpf.so.1


```
