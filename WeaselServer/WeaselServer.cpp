// WeaselServer.cpp : main source file for WeaselServer.exe
//
//	WTL MessageLoop 封装了消息循环. 实现了 getmessage/dispatchmessage....

#include "stdafx.h"
#include "resource.h"
#include "WeaselService.h"
#include <WeaselIPC.h>
#include <WeaselUI.h>
#include <RimeWithWeasel.h>
#include <WeaselUtility.h>
#include <winsparkle.h>
#include <functional>
#include <ShellScalingApi.h>
#include <WinUser.h>
#include <memory>
#include <atlstr.h>
#pragma comment(lib, "Shcore.lib")
CAppModule _Module;


// [GHOST-SERVICE] The LLM prediction service runs as a child of WeaselServer so
// that it starts with the input method and dies with it, instead of being tied
// to whatever editor or agent session happened to launch it.
static HANDLE g_ghost_job = NULL;

static void GhostLog(const std::wstring& message) {
  wchar_t temp[MAX_PATH] = {0};
  ExpandEnvironmentStringsW(L"%TEMP%\\rime.weasel", temp, MAX_PATH);
  CreateDirectoryW(temp, nullptr);
  std::wstring path = std::wstring(temp) + L"\\ghost-service.log";
  FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"a,ccs=UTF-8") == 0 && file) {
    SYSTEMTIME st = {};
    GetLocalTime(&st);
    fwprintf(file, L"[%04u-%02u-%02u %02u:%02u:%02u] %s\n", st.wYear,
             st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
             message.c_str());
    fclose(file);
  }
}

static void StartGhostService() {
  std::filesystem::path dir = WeaselServerApp::install_dir();
  std::filesystem::path script = dir / L"ghost_service.cmd";
  GhostLog(L"start: dir=" + dir.wstring() + L" script=" + script.wstring());
  if (GetFileAttributesW(script.c_str()) == INVALID_FILE_ATTRIBUTES) {
    GhostLog(L"script not found; ghost service not started");
    return;
  }

  g_ghost_job = CreateJobObjectW(NULL, NULL);
  if (g_ghost_job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = {};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(g_ghost_job, JobObjectExtendedLimitInformation,
                            &info, sizeof(info));
  }

  STARTUPINFOW si = {};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  std::wstring command = L"cmd.exe /c \"" + script.wstring() + L"\"";
  std::vector<wchar_t> buffer(command.begin(), command.end());
  buffer.push_back(0);

  PROCESS_INFORMATION pi = {};
  if (CreateProcessW(NULL, buffer.data(), NULL, NULL, FALSE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL,
                     dir.c_str(), &si, &pi)) {
    BOOL assigned = g_ghost_job
                        ? AssignProcessToJobObject(g_ghost_job, pi.hProcess)
                        : FALSE;
    GhostLog(L"spawned pid=" + std::to_wstring(pi.dwProcessId) +
             L" job_assigned=" + std::to_wstring(assigned) +
             L" job_error=" + std::to_wstring(GetLastError()));
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
  } else {
    GhostLog(L"CreateProcess failed, error=" +
             std::to_wstring(GetLastError()));
  }
}

static void StopGhostService() {
  if (g_ghost_job) {
    CloseHandle(g_ghost_job);
    g_ghost_job = NULL;
  }
}

int WINAPI _tWinMain(HINSTANCE hInstance,
                     HINSTANCE /*hPrevInstance*/,
                     LPTSTR lpstrCmdLine,
                     int nCmdShow) {
  LANGID langId = get_language_id();
  SetThreadUILanguage(langId);
  SetThreadLocale(langId);

  if (!IsWindowsBlueOrLaterEx()) {
    CString info, cap;
    info.LoadStringW(IDS_STR_SYSTEM_VERSION_WARNING);
    cap.LoadStringW(IDS_STR_SYSTEM_VERSION_WARNING_CAPTION);
    MessageBoxExW(NULL, info, cap, MB_ICONERROR, langId);
    return 0;
  }
  SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

  // 防止服务进程开启输入法
  ImmDisableIME(-1);

  WCHAR user_name[20] = {0};
  DWORD size = _countof(user_name);
  GetUserName(user_name, &size);
  if (!_wcsicmp(user_name, L"SYSTEM")) {
    return 1;
  }

  HRESULT hRes = ::CoInitialize(NULL);
  // If you are running on NT 4.0 or higher you can use the following call
  // instead to make the EXE free threaded. This means that calls come in on a
  // random RPC thread.
  // HRESULT hRes = ::CoInitializeEx(NULL, COINIT_MULTITHREADED);
  ATLASSERT(SUCCEEDED(hRes));

  // this resolves ATL window thunking problem when Microsoft Layer for Unicode
  // (MSLU) is used
  ::DefWindowProc(NULL, 0, 0, 0L);

  AtlInitCommonControls(
      ICC_BAR_CLASSES);  // add flags to support other controls

  hRes = _Module.Init(NULL, hInstance);
  ATLASSERT(SUCCEEDED(hRes));

  if (!wcscmp(L"/userdir", lpstrCmdLine)) {
    CreateDirectory(WeaselUserDataPath().c_str(), NULL);
    WeaselServerApp::explore(WeaselUserDataPath());
    return 0;
  }
  if (!wcscmp(L"/weaseldir", lpstrCmdLine)) {
    WeaselServerApp::explore(WeaselServerApp::install_dir());
    return 0;
  }
  if (!wcscmp(L"/ascii", lpstrCmdLine) || !wcscmp(L"/nascii", lpstrCmdLine)) {
    weasel::Client client;
    bool ascii = !wcscmp(L"/ascii", lpstrCmdLine);
    if (client.Connect())  // try to connect to running server
    {
      if (ascii)
        client.TrayCommand(ID_WEASELTRAY_ENABLE_ASCII);
      else
        client.TrayCommand(ID_WEASELTRAY_DISABLE_ASCII);
    }
    return 0;
  }

  // command line option /q stops the running server
  bool quit = !wcscmp(L"/q", lpstrCmdLine) || !wcscmp(L"/quit", lpstrCmdLine);
  // restart if already running
  {
    weasel::Client client;
    if (client.Connect())  // try to connect to running server
    {
      client.ShutdownServer();
      if (quit)
        return 0;
      int retry = 0;
      while (client.Connect() && retry < 10) {
        client.ShutdownServer();
        retry++;
        Sleep(50);
      }
      if (retry >= 10)
        return 0;
    } else if (quit)
      return 0;
  }

  bool check_updates = !wcscmp(L"/update", lpstrCmdLine);
  if (check_updates) {
    WeaselServerApp::check_update();
  }

  CreateDirectory(WeaselUserDataPath().c_str(), NULL);

  StartGhostService();

  int nRet = 0;
  try {
    WeaselServerApp app;
    RegisterApplicationRestart(NULL, 0);
    nRet = app.Run();
  } catch (...) {
    // bad luck...
    nRet = -1;
  }

  StopGhostService();

  _Module.Term();
  ::CoUninitialize();

  return nRet;
}
