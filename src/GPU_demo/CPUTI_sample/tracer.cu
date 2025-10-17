// tracer.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cupti.h>
#include "tracer_comm.h"

//用于处理错误
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

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// 全局变量，用于保存追踪开始时的时间戳
static int comm_fd=-1;
static uint64_t startTimestamp;

std::map<uint32_t, pid_t> g_cupti_tid_to_os_tid_map;
std::mutex g_thread_map_mutex;

pid_t get_os_tid() {
    return syscall(SYS_gettid);
}

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

const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

// static const char *
// getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
// {
//   switch (kind) {
//   case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
//     return "CUDA";
//   case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
//     return "CUDA_MPS";
//   default:
//     break;
//   }

//   return "<unknown>";
// }

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  // case CUPTI_ACTIVITY_KIND_DEVICE:
  //   {
  //     CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
  //     printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
  //            "multiprocessors %u, clock %u MHz\n",
  //            device->name, device->id,
  //            device->computeCapabilityMajor, device->computeCapabilityMinor,
  //            (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
  //            (unsigned int) (device->globalMemorySize / 1024 / 1024),
  //            device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
  //   {
  //     CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
  //     printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
  //            attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_CONTEXT:
  //   {
  //     CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
  //     printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
  //            context->contextId, context->deviceId,
  //            getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
  //            (int) context->nullStreamId);
  //     break;
  //   }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
             (unsigned long long) (memcpy->start - startTimestamp),
             (unsigned long long) (memcpy->end - startTimestamp),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             memset->value,
             (unsigned long long) (memset->start - startTimestamp),
             (unsigned long long) (memset->end - startTimestamp),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record;
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  // case CUPTI_ACTIVITY_KIND_NAME:
  //   {
  //     CUpti_ActivityName *name = (CUpti_ActivityName *) record;
  //     switch (name->objectKind)
  //     {
  //     case CUPTI_ACTIVITY_OBJECT_CONTEXT:
  //       printf("NAME  %s %u %s id %u, name %s\n",
  //              getActivityObjectKindString(name->objectKind),
  //              getActivityObjectKindId(name->objectKind, &name->objectId),
  //              getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
  //              getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
  //              name->name);
  //       break;
  //     case CUPTI_ACTIVITY_OBJECT_STREAM:
  //       printf("NAME %s %u %s %u %s id %u, name %s\n",
  //              getActivityObjectKindString(name->objectKind),
  //              getActivityObjectKindId(name->objectKind, &name->objectId),
  //              getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
  //              getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
  //              getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
  //              getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
  //              name->name);
  //       break;
  //     default:
  //       printf("NAME %s id %u, name %s\n",
  //              getActivityObjectKindString(name->objectKind),
  //              getActivityObjectKindId(name->objectKind, &name->objectId),
  //              name->name);
  //       break;
  //     }
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_MARKER:
  //   {
  //     CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
  //     printf("MARKER id %u [ %llu ], name %s, domain %s\n",
  //            marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_MARKER_DATA:
  //   {
  //     CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
  //     printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
  //            marker->id, marker->color, marker->category,
  //            (unsigned long long) marker->payload.metricValueUint64,
  //            marker->payload.metricValueDouble);
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_OVERHEAD:
  //   {
  //     CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
  //     printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
  //            getActivityOverheadKindString(overhead->overheadKind),
  //            (unsigned long long) overhead->start - startTimestamp,
  //            (unsigned long long) overhead->end - startTimestamp,
  //            getActivityObjectKindString(overhead->objectKind),
  //            getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
  //     break;
  //   }
  default:
    printf("  <unknown>\n");
    break;
  }
}

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

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        if (comm_fd == -1) continue;
            TraceRecord rec = {}; // 初始化为0
            rec.start_ns = 0;
            rec.end_ns = 0;

            switch (record->kind) {
                case CUPTI_ACTIVITY_KIND_KERNEL:
                case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                    CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
                    rec.type = RECORD_TYPE_KERNEL;
                    rec.start_ns = kernel->start;
                    rec.end_ns = kernel->end;
                    rec.correlationId = kernel->correlationId;
                    
                    rec.gpu_activity.deviceId = kernel->deviceId;
                    rec.gpu_activity.contextId = kernel->contextId;
                    rec.gpu_activity.streamId = kernel->streamId;
                    strncpy(rec.gpu_activity.name, kernel->name, sizeof(rec.gpu_activity.name) - 1);
                    
                    rec.gpu_activity.kernel.gridX = kernel->gridX;
                    rec.gpu_activity.kernel.gridY = kernel->gridY;
                    rec.gpu_activity.kernel.gridZ = kernel->gridZ;
                    rec.gpu_activity.kernel.blockX = kernel->blockX;
                    rec.gpu_activity.kernel.blockY = kernel->blockY;
                    rec.gpu_activity.kernel.blockZ = kernel->blockZ;
                    rec.gpu_activity.kernel.staticSharedMemory = kernel->staticSharedMemory;
                    rec.gpu_activity.kernel.dynamicSharedMemory = kernel->dynamicSharedMemory;
                    
                    write(comm_fd, &rec, sizeof(rec));
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMCPY: {
                    CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
                    rec.type = RECORD_TYPE_MEMCPY;
                    rec.start_ns = memcpy->start;
                    rec.end_ns = memcpy->end;
                    rec.correlationId = memcpy->correlationId;
                    
                    rec.gpu_activity.deviceId = memcpy->deviceId;
                    rec.gpu_activity.contextId = memcpy->contextId;
                    rec.gpu_activity.streamId = memcpy->streamId;
                    strncpy(rec.gpu_activity.name, getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind), sizeof(rec.gpu_activity.name) - 1);

                    rec.gpu_activity.memcpy.runtimeCorrelationId = memcpy->runtimeCorrelationId;
                    
                    write(comm_fd, &rec, sizeof(rec));
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMSET: {
                    CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *)record;
                    rec.type = RECORD_TYPE_MEMSET;
                    rec.start_ns = memset->start;
                    rec.end_ns = memset->end;
                    rec.correlationId = memset->correlationId;
                    
                    rec.gpu_activity.deviceId = memset->deviceId;
                    rec.gpu_activity.contextId = memset->contextId;
                    rec.gpu_activity.streamId = memset->streamId;
                    
                    rec.gpu_activity.memset.value = memset->value;
                    
                    write(comm_fd, &rec, sizeof(rec));
                    break;
                }
                case CUPTI_ACTIVITY_KIND_DRIVER:
                case CUPTI_ACTIVITY_KIND_RUNTIME: {
                    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
                    rec.type = (record->kind == CUPTI_ACTIVITY_KIND_DRIVER) ? RECORD_TYPE_DRIVER : RECORD_TYPE_RUNTIME;
                    rec.start_ns = api->start;
                    rec.end_ns = api->end;
                    rec.correlationId = api->correlationId;
                    rec.processId = api->processId;
                    
                    // 查询 OS TID
                    std::lock_guard<std::mutex> lock(g_thread_map_mutex);
                    auto it = g_cupti_tid_to_os_tid_map.find(api->threadId);
                    if (it != g_cupti_tid_to_os_tid_map.end()) {
                        rec.threadId = it->second;
                    } else {
                        rec.threadId = 0; // 未找到
                    }
                    
                    rec.cpu_api.cbid = api->cbid;
                    rec.cpu_api.cupti_internal_tid = api->threadId;

                    write(comm_fd, &rec, sizeof(rec));
                    break;
                }
                default:
                    break;
            }
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Tracer Warning: Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

// ---- 自动初始化和反初始化 ----

// 这个函数会在目标程序main函数执行前，由动态链接器自动调用
__attribute__((constructor))
void initCuptiTracer()
{
  fprintf(stderr, "--- CUPTI Tracer Library Loaded ---\n");

  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  // 添加其他活动类型

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

// 这个函数会在目标程序退出后，由动态链接器自动调用
__attribute__((destructor))
void deinitCuptiTracer()
{
  // 强制刷新所有CUPTI缓冲区，以确保能收集到所有记录
  CUPTI_CALL(cuptiActivityFlushAll(0));
  fprintf(stderr, "--- CUPTI Tracer Library Unloaded ---\n");
}