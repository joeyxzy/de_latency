# Makefile
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CC = gcc

# Adjust this path based on your CUDA version and system
CUPTI_LIB_DIR = $(CUDA_HOME)/extras/cupti/lib64
CUPTI_INC_DIR = $(CUDA_HOME)/extras/cupti/include

TARGET_LIB = libtracer.so

CFLAGS = -fPIC -I$(CUPTI_INC_DIR) -I$(CUDA_HOME)/include
LDFLAGS = -shared -L$(CUPTI_LIB_DIR) -lcupti -Wl,-rpath,$(CUPTI_LIB_DIR)

all: $(TARGET_LIB)

$(TARGET_LIB): tracer.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET_LIB) *.o