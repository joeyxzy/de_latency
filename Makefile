# === Unified Makefile for eBPF Controller and CUPTI Tracer ===
# CUDA paths
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI

# --- Compilers ---
CC = gcc
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
BPF_CLANG = clang

# --- Project Paths ---
# Assumes this Makefile is in the project root
SRC_DIR := ./src/FINAL
INCLUDE_DIR := ./include
LIBBPF_DIR := ./include/libbpf
LIB_DIR := ./lib
BUILD_DIR := ./build

# --- Common Preprocessor Flags (for Include Paths) ---
# This is where we tell ALL compilers where to find headers
# -I$(BUILD_DIR) is for the generated BPF skeleton
CPPFLAGS := -I$(INCLUDE_DIR) -I$(BUILD_DIR) -I$(LIBBPF_DIR)

# --- Final Targets ---
TARGET_CONTROLLER := $(BUILD_DIR)/controller
TARGET_TRACER := $(BUILD_DIR)/libtracer.so

# ==============================================================================
# Part 1: BPF Configuration
# ==============================================================================
BPF_SRC := $(SRC_DIR)/cpu_time.bpf.c
BPF_OBJ := $(BUILD_DIR)/cpu_time.bpf.o
BPF_SKELETON := $(BUILD_DIR)/cpu_time.skel.h

# --- Controller (User-space App) Configuration ---
USER_SRC := $(SRC_DIR)/cpu_time_user.c
CFLAGS := -g -Wall
LDFLAGS := -L$(LIB_DIR) -lbpf -lelf -lz -lcupti
# Set rpath so the controller can find libbpf.so at runtime
LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'


# ==============================================================================
# Part 2: CUDA CUPTI Tracer Configuration
# ==============================================================================
TRACER_SRC := $(SRC_DIR)/tracer.cu


# Add CUDA/CUPTI include paths ONLY for the nvcc command
NVCC_CPPFLAGS := -I"$(CUDA_INSTALL_PATH)/include" -I"$(CUPTI_INSTALL_PATH)/include"

# NVCC compiler and linker flags
NVCC_CFLAGS := -g -shared -Xcompiler -fPIC
NVCC_LDFLAGS := -L"$(CUPTI_INSTALL_PATH)/lib64" -L"$(CUDA_INSTALL_PATH)/lib64" -lcupti -lcuda


# ==============================================================================
# Build Rules
# ==============================================================================

# Default target: build both the controller and the tracer library
all: $(TARGET_CONTROLLER) $(TARGET_TRACER)

# Rule to create the build directory
$(BUILD_DIR):
	@echo "  MKDIR    $@"
	@mkdir -p $(BUILD_DIR)

# --- Rule to build the CUPTI Tracer Library ---
$(TARGET_TRACER): $(TRACER_SRC) $(INCLUDE_DIR)/tracer_comm.h | $(BUILD_DIR)
	@echo "  NVCC     $@"
	$(NVCC) $(CPPFLAGS) $(NVCC_CPPFLAGS) $(NVCC_CFLAGS) -o $@ $< $(NVCC_LDFLAGS)

# --- Rule to build the Controller Application ---
$(TARGET_CONTROLLER): $(USER_SRC) $(BPF_SKELETON) | $(BUILD_DIR)
	@echo "  CC       $@"
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS)

# --- Rule to generate the BPF Skeleton Header ---
# This rule is triggered by the controller's dependency on it
$(BPF_SKELETON): $(BPF_OBJ) | $(BUILD_DIR)
	@echo "  GEN_SKEL $@"
	@bpftool gen skeleton $< > $@

# --- Rule to compile the BPF Kernel Program ---
# This rule is triggered by the skeleton's dependency on it
$(BPF_OBJ): $(BPF_SRC) | $(BUILD_DIR)
	@echo "  CLANG    $@"
	$(BPF_CLANG) $(CPPFLAGS) -g -O2 -target bpf -c $< -o $@

# --- Cleanup Rule ---
clean:
	@echo "  CLEAN"
	@rm -rf $(BUILD_DIR)

# --- Run Example ---
# Assumes you have a CUDA app to test against
run: all
	@echo "--- Running Controller ---"
	sudo ./$(TARGET_CONTROLLER) /path/to/your/cuda/app