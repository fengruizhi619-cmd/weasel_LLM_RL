// [CLI-000 SECTION-INDEX]
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shellapi.h>
#include <sddl.h>

#include <string>
#include <vector>

// [CLI-001 HELPERS]
static std::wstring ModuleRoot() {
  wchar_t path[MAX_PATH]{};
  GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring file(path);
  size_t slash = file.find_last_of(L'\\');
  return slash == std::wstring::npos ? L"." : file.substr(0, slash);
}

static bool IsAdmin() {
  BOOL is_admin = FALSE;
  SID_IDENTIFIER_AUTHORITY authority = SECURITY_NT_AUTHORITY;
  PSID group = nullptr;
  if (AllocateAndInitializeSid(&authority, 2, SECURITY_BUILTIN_DOMAIN_RID,
                               DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0,
                               &group)) {
    CheckTokenMembership(nullptr, group, &is_admin);
    FreeSid(group);
  }
  return is_admin != FALSE;
}

static int RelaunchElevated(const std::wstring& command_line) {
  std::wstring exe = ModuleRoot() + L"\\WeaselCliInstaller.exe";
  SHELLEXECUTEINFOW info{};
  info.cbSize = sizeof(info);
  info.lpVerb = L"runas";
  info.lpFile = exe.c_str();
  info.lpParameters = command_line.c_str();
  info.nShow = SW_HIDE;
  info.fMask = SEE_MASK_NOCLOSEPROCESS;
  if (!ShellExecuteExW(&info) || info.hProcess == nullptr)
    return 1;
  WaitForSingleObject(info.hProcess, INFINITE);
  DWORD exit_code = 1;
  GetExitCodeProcess(info.hProcess, &exit_code);
  CloseHandle(info.hProcess);
  return static_cast<int>(exit_code);
}

static void Print(const std::wstring& text) {
  HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  if (output != INVALID_HANDLE_VALUE && output != nullptr &&
      GetConsoleMode(output, &mode)) {
    std::wstring line = text + L"\n";
    DWORD written = 0;
    WriteConsoleW(output, line.c_str(), static_cast<DWORD>(line.size()),
                  &written, nullptr);
  } else {
    fwprintf(stdout, L"%s\n", text.c_str());
  }
}

// [CLI-002 PROCESS-RUNNER]
static int RunFile(const std::wstring& root, const std::wstring& file,
                   const std::wstring& arguments) {
  std::wstring executable = root + L"\\" + file;
  std::wstring command_line = L"\"" + executable + L"\" " + arguments;
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW;
  startup.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(executable.c_str(), command_line.data(), nullptr,
                      nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                      root.c_str(), &startup, &process)) {
    Print(L"[ERROR] cannot start " + executable + L" error=" +
          std::to_wstring(GetLastError()));
    return 1;
  }
  WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 0;
  GetExitCodeProcess(process.hProcess, &exit_code);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  Print(file + L" exit=" + std::to_wstring(exit_code));
  return static_cast<int>(exit_code);
}

static int StartServer(const std::wstring& root) {
  std::wstring executable = root + L"\\WeaselServer.exe";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW;
  startup.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(executable.c_str(), nullptr, nullptr, nullptr, FALSE,
                      CREATE_NO_WINDOW, nullptr, root.c_str(), &startup,
                      &process)) {
    Print(L"[ERROR] cannot start WeaselServer.exe error=" +
          std::to_wstring(GetLastError()));
    return 1;
  }
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  Print(L"WeaselServer.exe started");
  return 0;
}

static int SetRegistry(const std::wstring& root) {
  HKEY key = nullptr;
  LONG result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Software\\Rime\\Weasel", 0,
                              KEY_SET_VALUE, &key);
  if (result == ERROR_SUCCESS) {
    RegSetValueExW(key, L"WeaselRoot", 0, REG_SZ,
                   reinterpret_cast<const BYTE*>(root.c_str()),
                   static_cast<DWORD>((root.size() + 1) * sizeof(wchar_t)));
    const std::wstring server = L"WeaselServer.exe";
    RegSetValueExW(key, L"ServerExecutable", 0, REG_SZ,
                   reinterpret_cast<const BYTE*>(server.c_str()),
                   static_cast<DWORD>((server.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
  }
  return result == ERROR_SUCCESS ? 0 : 1;
}

static void PrintUsage() {
  Print(L"WeaselCliInstaller [command]");
  Print(L"  install   - silent install from this package root");
  Print(L"  uninstall - silent uninstall of registered Weasel TSF");
  Print(L"  deploy    - deploy Rime user workspace");
  Print(L"  status    - print current installation status");
}

// [CLI-003 ACTIONS]
static int DoInstall(const std::wstring& root) {
  Print(L"[install] root=" + root);
  if (RunFile(root, L"WeaselServer.exe", L"/q") != 0)
    Print(L"[warn] existing server quit returned nonzero");
  int setup = RunFile(root, L"WeaselSetup.exe", L"/s");
  if (setup != 0)
    return setup;
  int deployer = RunFile(root, L"WeaselDeployer.exe", L"/install");
  if (deployer != 0)
    return deployer;
  if (SetRegistry(root) != 0)
    Print(L"[warn] WeaselRoot registry update failed");
  return StartServer(root);
}

static int DoUninstall(const std::wstring& root) {
  Print(L"[uninstall] root=" + root);
  RunFile(root, L"WeaselServer.exe", L"/q");
  return RunFile(root, L"WeaselSetup.exe", L"/u");
}

static int DoDeploy(const std::wstring& root) {
  Print(L"[deploy] root=" + root);
  return RunFile(root, L"WeaselDeployer.exe", L"/deploy");
}

static int DoStatus(const std::wstring& root) {
  Print(L"[status] root=" + root);
  wchar_t value[MAX_PATH]{};
  DWORD value_size = sizeof(value);
  HKEY key = nullptr;
  LONG result =
      RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Software\\Rime\\Weasel", 0,
                    KEY_QUERY_VALUE, &key);
  if (result == ERROR_SUCCESS) {
    result = RegQueryValueExW(key, L"WeaselRoot", nullptr, nullptr,
                              reinterpret_cast<LPBYTE>(value), &value_size);
    RegCloseKey(key);
  }
  if (result == ERROR_SUCCESS && value[0])
    Print(L"[status] installed root=" + std::wstring(value));
  else
    Print(L"[status] installed root=(not set)");
  return 0;
}

// [CLI-004 ENTRY]
int wmain() {
  int argc = 0;
  LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  std::wstring command = argc > 1 ? argv[1] : L"";
  if (argv)
    LocalFree(argv);

  if (command.empty() || command == L"/?" || command == L"/help" ||
      command == L"-h" || command == L"help") {
    PrintUsage();
    return command.empty() ? 2 : 0;
  }

  std::wstring root = ModuleRoot();
  if (!IsAdmin() && command != L"status") {
    Print(L"[auth] requiring administrator privileges");
    return RelaunchElevated(command);
  }

  if (command == L"install")
    return DoInstall(root);
  if (command == L"uninstall")
    return DoUninstall(root);
  if (command == L"deploy")
    return DoDeploy(root);
  if (command == L"status")
    return DoStatus(root);

  Print(L"unknown command: " + command);
  return 2;
}
