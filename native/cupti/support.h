#pragma once

void tracer_log(const char *format, ...);
int read_env_int(const char *key, int fallback, int minimum, int maximum);
int read_env_bool(const char *key, int fallback);
