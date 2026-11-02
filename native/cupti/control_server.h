#pragma once
#include <string>

using ControlCommand = std::string (*)(const std::string &);
using ControlTick = void (*)();
void control_start(ControlCommand command, ControlTick tick);
void control_stop();
const char *control_socket_path();
