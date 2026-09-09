// WeaselServer.h
#pragma once
#include <string>

// [GHOST-SERVICE] prediction service / offline recorder hosted by WeaselServer
void StartGhostService();
void StopGhostService();
void RestartGhostService();
std::wstring GhostMode();
void SetGhostMode(const wchar_t* mode);
