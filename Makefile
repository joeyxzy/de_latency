# === FIXED Makefile for STANDALONE eBPF Monitor ===
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI
CUDA_TARGET_DIR ?= $(firstword $(wildcard $(CUDA_INSTALL_PATH)/targets/x86_64-linux))
CUDA_INCLUDE_DIR ?= $(firstword $(wildcard $(CUDA_TARGET_DIR)/include) $(wildcard $(CUDA_INSTALL_PATH)/include))
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_TARGET_DIR)/lib) $(wildcard $(CUDA_INSTALL_PATH)/lib64))
CUPTI_INCLUDE_DIR ?= $(firstword $(wildcard $(CUPTI_INSTALL_PATH)/include) $(wildcard $(CUDA_INSTALL_PATH)/extras/CUPTI/include) $(wildcard $(CUDA_TARGET_DIR)/include))
CUPTI_LIB_DIR ?= $(firstword $(wildcard $(CUPTI_INSTALL_PATH)/lib64) $(wildcard $(CUDA_INSTALL_PATH)/extras/CUPTI/lib64) $(wildcard $(CUDA_TARGET_DIR)/lib) $(wildcard $(CUDA_INSTALL_PATH)/lib64))

CC = gcc
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
BPF_CLANG = clang

SRC_DIR := ./src/FINAL
INCLUDE_DIR := ./include
LIBBPF_DIR := ./include/libbpf  # MUST exist with libbpf headers
LIB_DIR := ./lib
BUILD_DIR := ./build
comma := ,

ZMQ_INCLUDE_DIR ?= $(firstword $(wildcard /home/joeyxzy/.local/include) \
                               $(wildcard /usr/include))

# --- Include paths ---
CPPFLAGS := -I$(INCLUDE_DIR) -I$(LIBBPF_DIR) $(if $(ZMQ_INCLUDE_DIR),-I$(ZMQ_INCLUDE_DIR),)

# 自定义库目录可选；若安装在系统默认路径则无需设置。
ZMQ_LIB_PATH ?= $(firstword $(wildcard /home/joeyxzy/.local/lib) \
                            $(wildcard /usr/lib/x86_64-linux-gnu) \
                            $(wildcard /lib/x86_64-linux-gnu))
JSONC_LIB_PATH ?=
EXTRA_LIB_DIRS := $(if $(ZMQ_LIB_PATH),-L$(ZMQ_LIB_PATH),) $(if $(JSONC_LIB_PATH),-L$(JSONC_LIB_PATH),)
LIBBPF_LIB ?= $(firstword $(wildcard ./third_party/libbpf/src/libbpf.so) \
                         $(wildcard ./tmp/libbpf/src/libbpf.so) \
                         $(wildcard /usr/lib/x86_64-linux-gnu/libbpf.so) \
                         $(wildcard /usr/lib/x86_64-linux-gnu/libbpf.so.0) \
                         $(wildcard /lib/x86_64-linux-gnu/libbpf.so) \
                         $(wildcard /lib/x86_64-linux-gnu/libbpf.so.0) \
                         $(wildcard $(LIB_DIR)/libbpf.so))
ELF_LIB ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libelf.so) \
                      $(wildcard /usr/lib/x86_64-linux-gnu/libelf.so.1) \
                      $(wildcard /lib/x86_64-linux-gnu/libelf.so) \
                      $(wildcard /lib/x86_64-linux-gnu/libelf.so.1))

# --- Final Targets ---
TARGET_CONTROLLER := $(BUILD_DIR)/controller
TARGET_EBPF_MONITOR := $(BUILD_DIR)/ebpf_monitor
TARGET_TRACER := $(BUILD_DIR)/libtracer.so

# ==============================================================================
# BPF Configuration
# ==============================================================================
BPF_SRC := $(SRC_DIR)/sched_latency.bpf.c
BPF_OBJ := $(BUILD_DIR)/sched_latency.bpf.o
BPF_SKELETON := $(BUILD_DIR)/sched_latency.skel.h

# ==============================================================================
# Controller Configuration (NO eBPF)
# ==============================================================================
CONTROLLER_SRC := $(SRC_DIR)/main.c
CONTROLLER_OBJ := $(BUILD_DIR)/main.o
ENCODE_SRC := $(SRC_DIR)/encode.c
ENCODE_OBJ := $(BUILD_DIR)/encode.o

CFLAGS := -g -Wall
CONTROLLER_LDFLAGS := -L$(LIB_DIR) $(EXTRA_LIB_DIRS) -lzmq -ljson-c
CONTROLLER_LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
CONTROLLER_LDFLAGS += $(if $(ZMQ_LIB_PATH),-Wl$(comma)-rpath$(comma)$(ZMQ_LIB_PATH),)

# ==============================================================================
# eBPF Monitor Configuration (STANDALONE)
# ==============================================================================
EBPF_MONITOR_SRC := $(SRC_DIR)/sched_latency.c
EBPF_MONITOR_OBJ := $(BUILD_DIR)/sched_latency.o
# CRITICAL FIX 1: Add libbpf include path for eBPF monitor
EBPF_CPPFLAGS := $(CPPFLAGS) -I$(LIBBPF_DIR) -I$(BUILD_DIR)
EBPF_LDFLAGS := $(EXTRA_LIB_DIRS) $(if $(LIBBPF_LIB),$(LIBBPF_LIB),-L$(LIB_DIR) -lbpf) \
                $(if $(ELF_LIB),$(ELF_LIB),-lelf) -lz -lzmq
EBPF_LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
EBPF_LDFLAGS += $(if $(ZMQ_LIB_PATH),-Wl$(comma)-rpath$(comma)$(ZMQ_LIB_PATH),)

# ==============================================================================
# CUPTI Tracer
# ==============================================================================
TRACER_SRC := $(SRC_DIR)/tracer.cu
NVCC_CPPFLAGS := -I"$(CUDA_INCLUDE_DIR)" -I"$(CUPTI_INCLUDE_DIR)"
NVCC_CFLAGS := -g -shared -Xcompiler -fPIC
NVCC_LDFLAGS := $(EXTRA_LIB_DIRS) -L"$(CUPTI_LIB_DIR)" -L"$(CUDA_LIB_DIR)" \
                -Xlinker -rpath -Xlinker "$(CUPTI_LIB_DIR)" \
                -Xlinker -rpath -Xlinker "$(CUDA_LIB_DIR)" \
                -lcupti -lcuda -lzmq -ljson-c

# ==============================================================================
# Build Rules
# ==============================================================================

all: $(TARGET_CONTROLLER) $(TARGET_EBPF_MONITOR) $(TARGET_TRACER)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# --- Shared encode module for controller ---
$(ENCODE_OBJ): $(ENCODE_SRC) | $(BUILD_DIR)
	@echo "  CC       $@"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# --- CUPTI Tracer ---
$(TARGET_TRACER): $(TRACER_SRC) $(INCLUDE_DIR)/tracer_comm.h $(ENCODE_OBJ) | $(BUILD_DIR)
	@echo "  NVCC     $@"
	$(NVCC) $(CPPFLAGS) $(NVCC_CPPFLAGS) $(NVCC_CFLAGS) -o $@ $< $(ENCODE_OBJ) $(NVCC_LDFLAGS)

# ==============================================================================
# Controller Build (NO eBPF)
# ==============================================================================
$(TARGET_CONTROLLER): $(CONTROLLER_OBJ) $(ENCODE_OBJ) | $(BUILD_DIR)
	@echo "  LINK     $@"
	$(CC) $(CFLAGS) -o $@ $^ $(CONTROLLER_LDFLAGS)

$(CONTROLLER_OBJ): $(CONTROLLER_SRC) | $(BUILD_DIR)
	@echo "  CC       $@"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# ==============================================================================
# eBPF Monitor Build (CRITICAL FIXES)
# ==============================================================================
# CRITICAL FIX 2: Remove $(BPF_SKELETON) from link dependencies!
$(TARGET_EBPF_MONITOR): $(EBPF_MONITOR_OBJ) | $(BUILD_DIR)
	@echo "  LINK (eBPF) $@"
	$(CC) $(CFLAGS) -o $@ $^ $(EBPF_LDFLAGS)

$(EBPF_MONITOR_OBJ): $(EBPF_MONITOR_SRC) $(BPF_SKELETON) | $(BUILD_DIR)
	@echo "  CC (eBPF) $@"
	# CRITICAL FIX 3: Use correct CPPFLAGS with libbpf path
	$(CC) $(EBPF_CPPFLAGS) $(CFLAGS) -c $< -o $@

# ==============================================================================
# BPF Rules
# ==============================================================================
$(BPF_SKELETON): $(BPF_OBJ) | $(BUILD_DIR)
	@echo "  GEN_SKEL $@"
	@bpftool gen skeleton $< > $@

$(BPF_OBJ): $(BPF_SRC) | $(BUILD_DIR)
	@echo "  CLANG    $@"
	$(BPF_CLANG) $(EBPF_CPPFLAGS) -g -O2 -target bpf -c $< -o $@

clean:
	@echo "  CLEAN"
	@rm -rf $(BUILD_DIR)

run_controller:
	@echo "--- Running Controller (NO eBPF) ---"
	sudo ./$(TARGET_CONTROLLER) python -c "import time; time.sleep(5)"

run_ebpf_monitor:
	@echo "--- Running eBPF Monitor (STANDALONE) ---"
	@echo "Auto mode: sudo ./$(TARGET_EBPF_MONITOR)"
	@echo "With root pid: sudo ./$(TARGET_EBPF_MONITOR) --auto --root-pid <pid>"
	@echo "With worker pid file: sudo ./$(TARGET_EBPF_MONITOR) --auto --root-pid <pid> --worker-pid-file /tmp/tracer_worker_pids"
	@echo "Manual mode: sudo ./$(TARGET_EBPF_MONITOR) <TID1> [TID2] ..."

run: all
	@echo "Build complete. Two independent executables created:"
	@echo "1. Controller: $(TARGET_CONTROLLER)"
	@echo "2. eBPF Monitor: $(TARGET_EBPF_MONITOR)"
