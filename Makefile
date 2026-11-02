# Native producers. Public build/* artifact names are stable.
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.8
CUPTI_INSTALL_PATH ?= $(CUDA_INSTALL_PATH)/extras/CUPTI
CUDA_TARGET_DIR ?= $(CUDA_INSTALL_PATH)/targets/x86_64-linux
CUDA_INCLUDE_DIR ?= $(firstword $(wildcard $(CUDA_TARGET_DIR)/include) $(wildcard $(CUDA_INSTALL_PATH)/include))
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_TARGET_DIR)/lib) $(wildcard $(CUDA_INSTALL_PATH)/lib64))
CUPTI_INCLUDE_DIR ?= $(firstword $(wildcard $(CUPTI_INSTALL_PATH)/include) $(wildcard $(CUDA_TARGET_DIR)/include))
CUPTI_LIB_DIR ?= $(firstword $(wildcard $(CUPTI_INSTALL_PATH)/lib64) $(wildcard $(CUDA_TARGET_DIR)/lib))
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
BPF_CLANG ?= clang
BUILD_DIR := build
ZMQ_INCLUDE_DIR ?=
ZMQ_LIB_PATH ?= $(or $(shell pkg-config --variable=libdir libzmq 2>/dev/null),$(patsubst %/,%,$(dir $(realpath $(shell $(CC) -print-file-name=libzmq.so)))))
JSONC_LIB_PATH ?= $(or $(shell pkg-config --variable=libdir json-c 2>/dev/null),$(patsubst %/,%,$(dir $(realpath $(shell $(CC) -print-file-name=libjson-c.so)))))
NATIVE_INCLUDE := native/include
LIBBPF_DIR := third_party/libbpf
LIBBPF_LIB ?= $(firstword $(wildcard /usr/lib/libbpf.so) $(wildcard /usr/lib/x86_64-linux-gnu/libbpf.so) $(wildcard $(LIBBPF_DIR)/lib/libbpf.so))
CPPFLAGS += -I$(NATIVE_INCLUDE) -I$(LIBBPF_DIR)/include $(shell pkg-config --cflags libzmq json-c 2>/dev/null)
CPPFLAGS += $(if $(ZMQ_INCLUDE_DIR),-I$(ZMQ_INCLUDE_DIR))
CFLAGS ?= -O2 -g -Wall
comma := ,
EXTRA_LIBS := $(if $(ZMQ_LIB_PATH),-L$(ZMQ_LIB_PATH) -Wl$(comma)-rpath$(comma)$(ZMQ_LIB_PATH))
EXTRA_LIBS += $(if $(JSONC_LIB_PATH),-L$(JSONC_LIB_PATH) -Wl$(comma)-rpath$(comma)$(JSONC_LIB_PATH))
NVCC_EXTRA_LIBS := $(foreach directory,$(ZMQ_LIB_PATH) $(JSONC_LIB_PATH),-L"$(directory)" -Xlinker -rpath -Xlinker "$(directory)")
ZMQ_JSON_LIBS := $(shell pkg-config --libs libzmq json-c 2>/dev/null || printf '%s' '-lzmq -ljson-c')

.PHONY: all clean run run_controller run_ebpf_monitor
all: build/controller build/ebpf_monitor build/libtracer.so

build:
	mkdir -p $@

build/controller: native/controller/main.c Makefile | build
	$(CC) $(CFLAGS) $< -o $@

build/sched_latency.bpf.o: native/ebpf/sched_latency.bpf.c $(NATIVE_INCLUDE)/sched_event.h Makefile | build
	$(BPF_CLANG) -I$(NATIVE_INCLUDE) -I$(LIBBPF_DIR)/include -g -O2 -target bpf -c $< -o $@

build/sched_latency.skel.h: build/sched_latency.bpf.o
	bpftool gen skeleton $< > $@.tmp
	mv $@.tmp $@

build/ebpf_monitor: native/ebpf/sched_latency.c $(NATIVE_INCLUDE)/sched_event.h build/sched_latency.skel.h Makefile | build
	$(CC) $(CPPFLAGS) -Ibuild $(CFLAGS) $< -o $@ $(EXTRA_LIBS) $(LIBBPF_LIB) -lelf -lz $(ZMQ_JSON_LIBS) -Wl,-rpath,'$$ORIGIN/../$(LIBBPF_DIR)/lib'

build/serialize.o: native/cupti/serialize.cpp native/cupti/serialize.h $(NATIVE_INCLUDE)/tracer_comm.h Makefile | build
	$(CXX) $(CPPFLAGS) -std=c++17 -O2 -g -fPIC -MMD -MP -c $< -o $@

build/cupti_%.o: native/cupti/%.cpp native/cupti/transport.h native/cupti/support.h $(NATIVE_INCLUDE)/tracer_comm.h Makefile | build
	$(CXX) $(CPPFLAGS) -I"$(CUDA_INCLUDE_DIR)" -I"$(CUPTI_INCLUDE_DIR)" -std=c++17 -O2 -g -fPIC -pthread -MMD -MP -c $< -o $@

build/cupti_markers.o: native/cupti/markers.cu native/cupti/markers.h Makefile | build
	$(NVCC) -std=c++17 -Xcompiler -fPIC -c $< -o $@

build/libtracer.so: native/cupti/tracer.cu native/cupti/serialize.h native/cupti/transport.h native/cupti/support.h native/cupti/control_server.h native/cupti/activity.h native/cupti/markers.h $(NATIVE_INCLUDE)/tracer_comm.h build/serialize.o build/cupti_support.o build/cupti_transport.o build/cupti_control_server.o build/cupti_activity.o build/cupti_markers.o Makefile | build
	$(NVCC) $(CPPFLAGS) -I"$(CUDA_INCLUDE_DIR)" -I"$(CUPTI_INCLUDE_DIR)" -std=c++17 -shared -Xcompiler -fPIC -o $@ $< build/serialize.o build/cupti_support.o build/cupti_transport.o build/cupti_control_server.o build/cupti_activity.o build/cupti_markers.o -L"$(CUPTI_LIB_DIR)" -L"$(CUDA_LIB_DIR)" -Xlinker -rpath -Xlinker "$(CUPTI_LIB_DIR)" -Xlinker -rpath -Xlinker "$(CUDA_LIB_DIR)" $(NVCC_EXTRA_LIBS) -lcupti -lcuda $(ZMQ_JSON_LIBS)

clean:
	rm -rf build

-include $(wildcard build/*.d)

run: all
run_controller: all
	sudo -E ./build/controller python3 -c "import time; time.sleep(5)"
run_ebpf_monitor: build/ebpf_monitor
	./build/ebpf_monitor --help
