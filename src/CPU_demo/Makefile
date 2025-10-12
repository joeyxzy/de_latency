# src/CPU_demo/Makefile

# 编译器
BPF_CLANG=clang
USER_GCC=gcc

# include 路径
INCLUDE_DIRS=-I$(PWD)/../../include \
             -I$(PWD)/../../include/libbpf

# lib 路径
LIB_DIR=$(PWD)/../../lib

# 源文件
BPF_SRC=cpu_time.bpf.c
USER_SRC=cpu_time_user.c

# 输出目录
BUILD_DIR=$(PWD)/../../build/CPU_demo
BPF_OBJ=$(BUILD_DIR)/cpu_time.bpf.o
USER_BIN=$(BUILD_DIR)/cpu_time_user

# 默认目标
all: $(BUILD_DIR) $(BPF_OBJ) $(USER_BIN)

# 创建 build 目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译内核态 BPF 程序（完全自包含）
$(BPF_OBJ): $(BPF_SRC)
	$(BPF_CLANG) -O2 -g -target bpf -c $< -o $@ \
		$(INCLUDE_DIRS)

# 编译用户态程序，自动查找项目 lib
$(USER_BIN): $(USER_SRC)
	$(USER_GCC) $< -o $@ \
		$(INCLUDE_DIRS) \
		-L$(LIB_DIR) -lbpf \
		-Wl,-rpath,'$$ORIGIN/../../lib'

# 清理
clean:
	rm -rf $(BUILD_DIR)
