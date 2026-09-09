#pragma once
#include <windows.h>
#include <string>
#include <cstdio>

inline void LlmLog(const std::wstring& line) {
  wchar_t temp[MAX_PATH]{};
  ExpandEnvironmentStringsW(L"%TEMP%\\rime.weasel", temp, MAX_PATH);
  CreateDirectoryW(temp, nullptr);
  std::wstring path = std::wstring(temp) + L"\\llm-ime.log";
  FILE* f = nullptr;
  if (_wfopen_s(&f, path.c_str(), L"a,ccs=UTF-8") != 0 || !f)
    return;
  SYSTEMTIME st{};
  GetLocalTime(&st);
  fwprintf(f, L"[%04u-%02u-%02u %02u:%02u:%02u.%03u pid=%lu tid=%lu] %s\n",
           st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
           st.wMilliseconds, GetCurrentProcessId(), GetCurrentThreadId(),
           line.c_str());
  fclose(f);
}