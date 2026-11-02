#include "support.h"
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <strings.h>
#include <unistd.h>

void tracer_log(const char *format, ...) {
    const char *path = getenv("TRACER_DEBUG_LOG_PATH");
    FILE *stream = fopen(path ? path : "/tmp/libtracer.log", "a");
    if (!stream) return;
    fprintf(stream, "[TRACER PID %d] ", getpid());
    va_list args;
    va_start(args, format);
    vfprintf(stream, format, args);
    va_end(args);
    fprintf(stream, "\n");
    fclose(stream);
}

int read_env_int(const char *key, int fallback, int minimum, int maximum) {
    const char *text = getenv(key);
    if (!text || !*text) return fallback;
    char *end = nullptr;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end) {
        tracer_log("Invalid integer for %s=%s, fallback to %d", key, text, fallback);
        return fallback;
    }
    return value < minimum ? minimum : value > maximum ? maximum : static_cast<int>(value);
}

int read_env_bool(const char *key, int fallback) {
    const char *text = getenv(key);
    if (!text || !*text) return fallback;
    for (const char *value : {"0", "false", "off", "no"})
        if (!strcasecmp(text, value)) return 0;
    for (const char *value : {"1", "true", "on", "yes"})
        if (!strcasecmp(text, value)) return 1;
    tracer_log("Invalid boolean for %s=%s, fallback to %d", key, text, fallback);
    return fallback;
}
