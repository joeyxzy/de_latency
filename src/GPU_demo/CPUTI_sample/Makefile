# Makefile for building libtracer.so

# 使用你的 CUDA 安装路径
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"

# CUPTI 相关的路径
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I$(CUPTI_INSTALL_PATH)/include

# 链接库路径和库文件
LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
CUPTI_LIB_PATH ?= $(CUPTI_INSTALL_PATH)/lib64
LIBS := -L$(LIB_PATH) -L$(CUPTI_LIB_PATH) -lcupti -lcudart

# Gencode arguments - 确保这里只包含你的 CUDA 版本支持的架构
SMS ?= 75 80 86 87 89 90
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# 目标库文件名
TARGET_LIB = libtracer.so

# 编译标志:
# -Xcompiler -fPIC: 告诉宿主编译器(g++/gcc)生成位置无关代码，这是构建共享库所必需的。
# -c: 只编译，不链接，生成 .o 文件。
NVCCFLAGS_COMPILE = $(GENCODE_FLAGS) -Xcompiler -fPIC -c

# 链接标志:
# -shared: 告诉 nvcc 生成一个共享库 (.so) 而不是可执行文件。
NVCCFLAGS_LINK = -shared

# 默认目标，构建我们的库
all: $(TARGET_LIB)

# 链接规则：从 tracer.o 创建 libtracer.so
$(TARGET_LIB): tracer.o
	@echo "--- Linking $(TARGET_LIB) ---"
	$(NVCC) $(NVCCFLAGS_LINK) -o $@ $^ $(LIBS)

# 编译规则：从 tracer.cu 创建 tracer.o
tracer.o: tracer.cu
	@echo "--- Compiling $< ---"
	$(NVCC) $(NVCCFLAGS_COMPILE) $(INCLUDES) $< -o $@

# 清理规则
clean:
	@echo "--- Cleaning up ---"
	rm -f $(TARGET_LIB) *.o