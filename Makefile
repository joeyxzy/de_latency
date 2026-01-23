# === FIXED Makefile for STANDALONE eBPF Monitor ===
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI

CC = gcc
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
BPF_CLANG = clang

SRC_DIR := ./src/FINAL
INCLUDE_DIR := ./include
LIBBPF_DIR := ./include/libbpf  # MUST exist with libbpf headers
LIB_DIR := ./lib
BUILD_DIR := ./build

# --- Include paths ---
CPPFLAGS := -I$(INCLUDE_DIR) -I$(LIBBPF_DIR)

#自定义的库路径（给sched_latency服务）
ZMQ_LIB_PATH := /home/joeyxzy/zeromq_install/lib
JSONC_LIB_PATH := /home/joeyxzy/jsonc_install/lib

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
CONTROLLER_LDFLAGS := -L$(LIB_DIR) -lzmq -ljson-c -lcupti
CONTROLLER_LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'

# ==============================================================================
# eBPF Monitor Configuration (STANDALONE)
# ==============================================================================
EBPF_MONITOR_SRC := $(SRC_DIR)/sched_latency.c
EBPF_MONITOR_OBJ := $(BUILD_DIR)/sched_latency.o
EBPF_ENCODE_OBJ := $(BUILD_DIR)/encode_ebpf.o

# CRITICAL FIX 1: Add libbpf include path for eBPF monitor
EBPF_CPPFLAGS := $(CPPFLAGS) -I$(LIBBPF_DIR) -I$(BUILD_DIR)
EBPF_LDFLAGS := -L$(LIB_DIR) -L$(ZMQ_LIB_PATH) -L$(JSONC_LIB_PATH) -lbpf -lelf -lz -lzmq -ljson-c
EBPF_LDFLAGS += -Wl,-rpath,$(ZMQ_LIB_PATH):$(JSONC_LIB_PATH):'$$ORIGIN/../lib'

# ==============================================================================
# CUPTI Tracer
# ==============================================================================
TRACER_SRC := $(SRC_DIR)/tracer.cu
NVCC_CPPFLAGS := -I"$(CUDA_INSTALL_PATH)/include" -I"$(CUPTI_INSTALL_PATH)/include"
NVCC_CFLAGS := -g -shared -Xcompiler -fPIC
NVCC_LDFLAGS := -L"$(CUPTI_INSTALL_PATH)/lib64" -L"$(CUDA_INSTALL_PATH)/lib64" \
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

# --- encode module for ebpf_monitor ---
$(EBPF_ENCODE_OBJ): $(ENCODE_SRC) | $(BUILD_DIR)
	@echo "  CC (eBPF) $@"
	$(CC) $(EBPF_CPPFLAGS) $(CFLAGS) -c $< -o $@

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
$(TARGET_EBPF_MONITOR): $(EBPF_MONITOR_OBJ) $(EBPF_ENCODE_OBJ) | $(BUILD_DIR)
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
	@echo "Usage: sudo ./$(TARGET_EBPF_MONITOR) -p <PID>"
	@echo "Example: sudo ./$(TARGET_EBPF_MONITOR) -p $$(pgrep -f 'python -c')"

run: all
	@echo "Build complete. Two independent executables created:"
	@echo "1. Controller: $(TARGET_CONTROLLER)"
	@echo "2. eBPF Monitor: $(TARGET_EBPF_MONITOR)"