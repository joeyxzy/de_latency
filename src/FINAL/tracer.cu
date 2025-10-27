#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <mutex>

#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>

#include "tracer_comm.h"

// --- CUPTI MACROS AND HELPERS ---
#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define BUF_SIZE (64 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

// --- GLOBAL STATE ---
static int comm_fd = -1;
static uint64_t startTimestamp; // For relative timestamps
// std::map<uint32_t, pid_t> g_corr_to_os_tid_map;
// std::mutex g_thread_map_mutex;
static std::map<uint32_t, pid_t> g_corr_to_ostid_map;
static std::mutex g_corr_mutex;
static CUpti_SubscriberHandle g_subscriber; // CUPTI subscriber 句柄

pid_t get_os_tid() { return syscall(SYS_gettid); }

// void register_thread_id() {
//     pthread_t self_pthread_id = pthread_self();
//     pid_t self_os_tid = get_os_tid();
//     uint32_t cupti_thread_id = (uint32_t)self_pthread_id;

//     std::lock_guard<std::mutex> lock(*g_thread_map_mutex);
//     if (g_corr_to_os_tid_map->find(cupti_thread_id) == g_corr_to_os_tid_map->end()) {
//         (*g_corr_to_os_tid_map)[cupti_thread_id] = self_os_tid;

//         if (comm_fd != -1) { // Send mapping to controller
//             UnifiedTraceRecord rec = {};
//             rec.type = RECORD_TYPE_METADATA_TID_MAP;
//             rec.pid = getpid();
//             rec.tid = self_os_tid;
//             rec.correlationId = cupti_thread_id; // Reuse field
//             write(comm_fd, &rec, sizeof(rec));
//         }
//     }
// }
// 这是我们的同步回调函数

void CUPTIAPI api_callback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cbid,
  const void *cbdata_void)
{
  const CUpti_CallbackData *cbdata = (const CUpti_CallbackData *)cbdata_void;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
      uint32_t correlationId = cbdata->correlationId;
      pid_t os_tid = syscall(SYS_gettid);

      {
          std::lock_guard<std::mutex> lock(g_corr_mutex);
          g_corr_to_ostid_map[correlationId] = os_tid;
      }

      // 这里你可以输出调试信息
      //printf("[ENTER] corr=%u tid=%d api=%s\n", correlationId, os_tid, cbdata->functionName);
  }
  //这里需要注意这里的大小会不会爆，即map
  // else if (cbdata->callbackSite == CUPTI_API_EXIT) {
  //     // 可选：清理映射，防止 map 太大
  //     std::lock_guard<std::mutex> lock(g_corr_mutex);
  //     g_corr_to_tid.erase(cbdata->correlationId);
  // }
}

// --- CUPTI CALLBACKS ---
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    fprintf(stderr, "Tracer Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    std::vector<uint32_t> consumed_ids;
    CUpti_Activity *record = NULL;
    if (validSize == 0 || comm_fd == -1) {
        if(buffer) free(buffer);
        return;
    }
    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status != CUPTI_SUCCESS) break;

        UnifiedTraceRecord rec = {}; // Zero-initialize
        rec.pid = getpid();
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4 *)record;
                rec.type = RECORD_TYPE_KERNEL;
                rec.start_ns = k->start;
                rec.end_ns = k->end;
                rec.correlationId = k->correlationId;
                rec.deviceId = k->deviceId;
                rec.contextId = k->contextId;
                rec.streamId = k->streamId;
                strncpy(rec.name, k->name, sizeof(rec.name) - 1);
                rec.gridX = k->gridX; rec.gridY = k->gridY; rec.gridZ = k->gridZ;
                rec.blockX = k->blockX; rec.blockY = k->blockY; rec.blockZ = k->blockZ;
                rec.staticSharedMemory = k->staticSharedMemory;
                rec.dynamicSharedMemory = k->dynamicSharedMemory;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy *m = (CUpti_ActivityMemcpy *)record;
                rec.type = RECORD_TYPE_MEMCPY;
                rec.start_ns = m->start;
                rec.end_ns = m->end;
                rec.correlationId = m->correlationId;
                rec.deviceId = m->deviceId;
                rec.contextId = m->contextId;
                rec.streamId = m->streamId;
                strncpy(rec.name, getMemcpyKindString((CUpti_ActivityMemcpyKind)m->copyKind), sizeof(rec.name) - 1);
                rec.memcpy_runtimeCorrelationId = m->runtimeCorrelationId;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                CUpti_ActivityMemset *m = (CUpti_ActivityMemset *)record;
                rec.type = RECORD_TYPE_MEMSET;
                rec.start_ns = m->start;
                rec.end_ns = m->end;
                rec.correlationId = m->correlationId;
                rec.deviceId = m->deviceId;
                rec.contextId = m->contextId;
                rec.streamId = m->streamId;
                rec.memset_value = m->value;
                break;
            }
            case CUPTI_ACTIVITY_KIND_DRIVER:
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
                rec.type = (record->kind == CUPTI_ACTIVITY_KIND_DRIVER) ? RECORD_TYPE_DRIVER : RECORD_TYPE_RUNTIME;
                rec.start_ns = api->start;
                rec.end_ns = api->end;
                rec.correlationId = api->correlationId;
                rec.pid = api->processId;
                rec.cbid = api->cbid;
                
                std::lock_guard<std::mutex> lock(g_corr_mutex);
                auto it = g_corr_to_ostid_map.find(api->correlationId); // api->threadId 是 CUPTI internal TID
                if (it != g_corr_to_ostid_map.end()) {
                    rec.tid = it->second; // 找到了对应的 OS TID！
                    consumed_ids.push_back(api->correlationId);
                } else {
                    rec.tid = 0; // 理论上不应该发生，但作为保护
                }
                break;
            }
            default: continue; // Skip other record types
        }
        write(comm_fd, &rec, sizeof(rec));

    } while (status == CUPTI_SUCCESS);
    {
      std::lock_guard<std::mutex> lock(g_corr_mutex);
      for (uint32_t cid : consumed_ids) {
          g_corr_to_ostid_map.erase(cid);
      }
    }
    free(buffer);
}

// --- CONSTRUCTOR & DESTRUCTOR ---
__attribute__((constructor))
void initCuptiTracer() {
    const char* fd_str = getenv("CUPTI_COMM_FD");
    if (fd_str) comm_fd = atoi(fd_str);
    //register_thread_id(); // Register main thread
    CUPTI_CALL(cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)api_callback, nullptr));
    //对runtime和driver两个domain都启动
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));

    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

__attribute__((destructor))
void deinitCuptiTracer() {
    CUPTI_CALL(cuptiActivityFlushAll(0));
    if (comm_fd != -1) {
        UnifiedTraceRecord rec = {};
        rec.type = RECORD_TYPE_METADATA_FLUSH_COMPLETE;
        write(comm_fd, &rec, sizeof(rec));
        close(comm_fd);
    }
}