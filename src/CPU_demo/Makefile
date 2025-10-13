# === src/CPU_demo/Makefile ===

BPF_CLANG = clang
USER_GCC  = gcc

# Paths
CUR_DIR      := $(CURDIR)
ROOT_DIR     := $(CUR_DIR)/../..
INCLUDE_DIRS := -I$(ROOT_DIR)/include -I$(ROOT_DIR)/include/libbpf
LIB_DIR      := $(ROOT_DIR)/lib

# Sources
BPF_SRC  := cpu_time.bpf.c
USER_SRC := cpu_time_user.c

# Build dirs
BUILD_DIR     := $(ROOT_DIR)/build/CPU_demo
BPF_OBJ       := $(BUILD_DIR)/cpu_time.bpf.o
BPF_SKEL_DIR  := $(BUILD_DIR)/include
BPF_SKELETON  := $(BPF_SKEL_DIR)/cpu_time.skel.h
USER_BIN      := $(BUILD_DIR)/cpu_time_user

# Default
all: $(BUILD_DIR) $(BPF_OBJ) $(USER_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BPF_SKEL_DIR)

# Compile BPF
$(BPF_OBJ): $(BPF_SRC)
	$(BPF_CLANG) -O2 -g -target bpf -c $< -o $@ $(INCLUDE_DIRS)

# Generate Skeleton + build user
$(USER_BIN): $(USER_SRC) $(BPF_OBJ)
	@echo "  GEN_SKEL   $(BPF_SKELETON)"
	bpftool gen skeleton $(BPF_OBJ) > $(BPF_SKELETON)

	@echo "  GCC        $@"
	$(USER_GCC) $< -o $@ \
		-I$(BPF_SKEL_DIR) $(INCLUDE_DIRS) \
		-L$(LIB_DIR) -lbpf -lelf -lz \
		-Wl,-rpath,'$$ORIGIN/../../lib'

clean:
	rm -rf $(BUILD_DIR)
