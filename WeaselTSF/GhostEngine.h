#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <Windows.h>

#pragma comment(lib, "ws2_32.lib")
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iterator>
#include <cstdint>
#include <algorithm>
#include <cwctype>
#include <utility>
#include <memory>
#include <atomic>
#include <cstdlib>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <cstdio>
#include <functional>

#include "CandidateTree.h"

namespace weasel {

namespace ghost {

constexpr int kDefaultContextChars = 256;
constexpr int kMaxDisplayChars = 24;
constexpr int kTreeBranchChars = 10;
constexpr int kTreeWidth = 20;
constexpr int kDefaultIdleMilliseconds = 800;
// [P3-3] Adaptive trigger: the idle threshold moves between these bounds
// according to the engine service's accept rate.
constexpr int kMinIdleMilliseconds = 400;
constexpr int kMaxIdleMilliseconds = 1200;
constexpr size_t kMinContextChars = 2;
constexpr int kPredictionTimeoutMilliseconds = 1200;
// [TREE-012] The preview never times out: it stays until the context changes
// or focus is lost. The expiry mechanism was removed at the user's request.
// Upper bound on model requests per growth run.
constexpr int kMaxFetchesPerRun = 64;
// [TREE-011] While a composition is live, ask for a much wider candidate set
// and let the service filter it by pinyin, so the constrained branch actually
// exists among the proposals.
constexpr int kPinyinProbeWidth = 200;
// [TREE-014] Sibling branches are expanded concurrently; the engine service
// merges the concurrent requests into a single batched forward.
constexpr int kParallelRequests = 8;
// Without a repetition penalty the base model keeps predicting the same
// character along a chain ("呢呢呢呢"). Applied to the prompt window too, so a
// character the user just typed is suppressed as a candidate.
constexpr const char* kRepeatPenaltyJson =
    "\"repeat_penalty\":1.3,\"repeat_last_n\":256,";

inline uint64_t HashBytes(uint64_t seed, const void* data, size_t size) {
  const unsigned char* bytes = static_cast<const unsigned char*>(data);
  uint64_t hash = seed;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

inline uint64_t HashContext(uint64_t document_token,
                            int64_t caret,
                            const std::wstring& text) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  hash = HashBytes(hash, &document_token, sizeof(document_token));
  hash = HashBytes(hash, &caret, sizeof(caret));
  const int length = static_cast<int>(text.size());
  hash = HashBytes(hash, &length, sizeof(length));
  if (!text.empty())
    hash = HashBytes(hash, text.data(), text.size() * sizeof(wchar_t));
  return hash;
}

inline void TraceLine(const std::wstring& line) {
  wchar_t trace_flag[8]{};
  DWORD flag_size = GetEnvironmentVariableW(L"WEASEL_GHOST_TRACE", trace_flag,
                                            static_cast<DWORD>(std::size(trace_flag)));
  if (!(flag_size > 0 && flag_size < std::size(trace_flag) &&
        wcscmp(trace_flag, L"1") == 0))
    return;
  wchar_t temp_root[MAX_PATH]{};
  ExpandEnvironmentStringsW(L"%TEMP%\\rime.weasel", temp_root,
                            MAX_PATH);
  CreateDirectoryW(temp_root, nullptr);
  std::wstring path = std::wstring(temp_root) + L"\\ghost-debug.log";
  FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"a,ccs=UTF-8") != 0 || !file)
    return;
  fwprintf(file, L"[%lu] %s\n", GetCurrentThreadId(), line.c_str());
  fclose(file);
}

struct Snapshot {
  uint64_t document_token = 0;
  int64_t caret = -1;
  uint64_t context_hash = 0;
  std::wstring prefix;
  RECT caret_rect{};

  bool SameAnchor(const Snapshot& other) const {
    return document_token == other.document_token && caret == other.caret &&
           context_hash == other.context_hash && prefix == other.prefix;
  }
};

class Panel {
 public:
  ~Panel() { ShutdownUi(); }

  void Prepare() { StartUi(); }

  void PostShow(const std::wstring& text, const RECT& caret_rect) {
    if (text.empty())
      return;
    if (!StartUi())
      return;

    auto* message = new ShowMessage{text, caret_rect};
    if (!PostMessageW(hwnd_, kShowMessage, 0,
                      reinterpret_cast<LPARAM>(message))) {
      delete message;
      return;
    }
    visible_.store(true, std::memory_order_release);
  }

  void PostHide() {
    if (hwnd_ && IsWindow(hwnd_))
      PostMessageW(hwnd_, kHideMessage, 0, 0);
    visible_.store(false, std::memory_order_release);
  }

  bool Visible() const { return visible_.load(std::memory_order_acquire); }

 private:
  bool StartUi() {
    std::lock_guard<std::mutex> start_lock(start_mutex_);
    if (ui_thread_.joinable())
      return ui_ready_.load(std::memory_order_acquire);

    ui_running_.store(true, std::memory_order_release);
    ui_thread_ = std::thread([this] { UiLoop(); });
    std::unique_lock<std::mutex> lock(init_mutex_);
    init_cv_.wait(lock, [this] {
      return ui_ready_.load(std::memory_order_acquire) ||
             !ui_running_.load(std::memory_order_acquire);
    });
    return ui_ready_.load(std::memory_order_acquire);
  }

  void UiLoop() {
    ui_thread_id_ = GetCurrentThreadId();
    bool created = EnsureWindow();
    {
      std::lock_guard<std::mutex> lock(init_mutex_);
      ui_ready_.store(created, std::memory_order_release);
      ui_running_.store(created, std::memory_order_release);
      init_cv_.notify_all();
    }
    if (!created)
      return;

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
      TranslateMessage(&message);
      DispatchMessageW(&message);
    }
  }

  void ShutdownUi() {
    if (!ui_thread_.joinable())
      return;
    if (ui_thread_id_)
      PostThreadMessageW(ui_thread_id_, WM_QUIT, 0, 0);
    ui_thread_.join();
  }

  struct ShowMessage {
    std::wstring text;
    RECT caret_rect;
  };

  static constexpr UINT kShowMessage = WM_APP + 0x5a01;
  static constexpr UINT kHideMessage = WM_APP + 0x5a02;

  static LRESULT CALLBACK WindowProc(HWND hwnd, UINT message,
                                     WPARAM wparam, LPARAM lparam) {
    auto* panel = reinterpret_cast<Panel*>(
        GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    switch (message) {
      case kShowMessage: {
        auto* payload = reinterpret_cast<ShowMessage*>(lparam);
        if (panel && payload) {
          panel->RenderShow(payload->text, payload->caret_rect);
        }
        delete payload;
        return 0;
      }
      case kHideMessage:
        if (panel) {
          panel->RenderHide();
        }
        return 0;
      default:
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }
  }

  void RenderShow(const std::wstring& text, const RECT& caret_rect) {
    const int width = MeasureWidth(text) + 24;
    const int height = 36;
    HDC screen_dc = GetDC(NULL);
    HDC memory_dc = CreateCompatibleDC(screen_dc);

    BITMAPV5HEADER header{};
    header.bV5Size = sizeof(header);
    header.bV5Width = width;
    header.bV5Height = -height;
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5RedMask = 0x00ff0000;
    header.bV5GreenMask = 0x0000ff00;
    header.bV5BlueMask = 0x000000ff;
    header.bV5AlphaMask = 0xff000000;

    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(memory_dc,
                                      reinterpret_cast<BITMAPINFO*>(&header),
                                      DIB_RGB_COLORS, &bits, NULL, 0);
    HGDIOBJ old_bitmap = SelectObject(memory_dc, bitmap);
    if (!bitmap || !old_bitmap) {
      Cleanup(memory_dc, bitmap);
      ReleaseDC(NULL, screen_dc);
      return;
    }

    RECT fill{0, 0, width, height};
    HBRUSH background = CreateSolidBrush(RGB(250, 250, 250));
    FillRect(memory_dc, &fill, background);
    DeleteObject(background);

    HFONT font = CreateFontW(-22, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                             DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                             CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                             DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei UI");
    HGDIOBJ old_font = SelectObject(memory_dc, font);
    SetBkMode(memory_dc, TRANSPARENT);
    SetTextColor(memory_dc, RGB(110, 110, 110));
    RECT text_rect{12, 6, width - 12, height - 6};
    DrawTextW(memory_dc, text.c_str(), static_cast<int>(text.size()),
              &text_rect, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    SelectObject(memory_dc, old_font);
    DeleteObject(font);

    auto* pixels = static_cast<unsigned char*>(bits);
    const size_t byte_count = static_cast<size_t>(width) * height * 4;
    for (size_t i = 3; i < byte_count; i += 4)
      pixels[i] = 235;

    POINT position{caret_rect.left, caret_rect.bottom + 2};
    HMONITOR monitor = MonitorFromPoint(position, MONITOR_DEFAULTTONEAREST);
    MONITORINFO info{};
    info.cbSize = sizeof(info);
    if (GetMonitorInfoW(monitor, &info)) {
      if (position.y + height > info.rcWork.bottom)
        position.y = max(info.rcWork.top, caret_rect.top - height - 2);
      position.x = min(position.x, info.rcWork.right - width - 4);
      position.x = max(info.rcWork.left + 4, position.x);
    }

    SIZE size{width, height};
    POINT zero{0, 0};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    UpdateLayeredWindow(hwnd_, screen_dc, &position, &size, memory_dc, &zero,
                        RGB(0, 0, 0), &blend, ULW_ALPHA);

    Cleanup(memory_dc, bitmap);
    ReleaseDC(NULL, screen_dc);
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
  }

  void RenderHide() {
    if (hwnd_) {
      ShowWindow(hwnd_, SW_HIDE);
    }
  }

 private:
  static void Cleanup(HDC memory_dc, HBITMAP bitmap) {
    if (memory_dc)
      DeleteDC(memory_dc);
    if (bitmap)
      DeleteObject(bitmap);
  }

  static int MeasureWidth(const std::wstring& text) {
    HDC dc = GetDC(NULL);
    HFONT font = CreateFontW(-22, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                             DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                             CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                             DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei UI");
    HGDIOBJ old_font = SelectObject(dc, font);
    SIZE size{};
    GetTextExtentPoint32W(dc, text.c_str(),
                          static_cast<int>(text.size()), &size);
    SelectObject(dc, old_font);
    DeleteObject(font);
    ReleaseDC(NULL, dc);
    return max(48, size.cx);
  }

  bool EnsureWindow() {
    if (hwnd_ && IsWindow(hwnd_))
      return true;

    WNDCLASSEXW description{};
    description.cbSize = sizeof(description);
    description.lpfnWndProc = Panel::WindowProc;
    description.hInstance = GetModuleHandleW(nullptr);
    description.lpszClassName = L"WeaselGhostPanel";
    RegisterClassExW(&description);

    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE |
            WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        description.lpszClassName, L"", WS_POPUP, 0, 0, 1, 1, NULL, NULL,
        description.hInstance, nullptr);
    SetWindowLongPtrW(hwnd_, GWLP_USERDATA,
                      reinterpret_cast<LONG_PTR>(this));
    return hwnd_ != nullptr;
  }

  HWND hwnd_ = nullptr;
  std::atomic<bool> visible_{false};
  std::thread ui_thread_;
  std::mutex start_mutex_;
  std::mutex init_mutex_;
  std::condition_variable init_cv_;
  std::atomic<bool> ui_ready_{false};
  std::atomic<bool> ui_running_{false};
  DWORD ui_thread_id_ = 0;
};


// [ENGINE-HTTP-LOCAL-CLIENT]
namespace localhttp {

inline bool EnsureWinsock() {
  static const bool initialized = [] {
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
  }();
  return initialized;
}

inline std::string NarrowAscii(const std::wstring& value) {
  std::string result;
  result.reserve(value.size());
  for (wchar_t ch : value)
    result.push_back(static_cast<char>(ch));
  return result;
}

inline bool SplitHttpEndpoint(const std::wstring& endpoint,
                              std::string& host,
                              std::string& port,
                              std::string& path) {
  const std::wstring prefix = L"http://";
  if (endpoint.compare(0, prefix.size(), prefix) != 0)
    return false;
  const std::wstring rest = endpoint.substr(prefix.size());
  const size_t slash = rest.find(L'/');
  const std::wstring host_port =
      slash == std::wstring::npos ? rest : rest.substr(0, slash);
  const std::wstring raw_path =
      slash == std::wstring::npos ? L"/" : rest.substr(slash);
  const size_t colon = host_port.rfind(L':');
  if (colon == std::wstring::npos) {
    host = NarrowAscii(host_port);
    port = "80";
  } else {
    host = NarrowAscii(host_port.substr(0, colon));
    port = NarrowAscii(host_port.substr(colon + 1));
  }
  path = NarrowAscii(raw_path);
  return !host.empty() && !port.empty() && !path.empty();
}

inline bool PostJson(const std::wstring& endpoint,
                     const std::string& request,
                     int timeout_milliseconds,
                     std::string& status_line,
                     std::string& body,
                     DWORD& error) {
  error = 0;
  if (!EnsureWinsock()) {
    error = WSAGetLastError();
    return false;
  }

  std::string host, port, path;
  if (!SplitHttpEndpoint(endpoint, host, port, path)) {
    error = ERROR_INVALID_PARAMETER;
    return false;
  }

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  addrinfo* addresses = nullptr;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0 ||
      addresses == nullptr) {
    error = WSAGetLastError();
    return false;
  }

  SOCKET socket_handle = INVALID_SOCKET;
  for (addrinfo* address = addresses; address; address = address->ai_next) {
    socket_handle = socket(address->ai_family, address->ai_socktype,
                           address->ai_protocol);
    if (socket_handle == INVALID_SOCKET)
      continue;
    if (connect(socket_handle, address->ai_addr,
                static_cast<int>(address->ai_addrlen)) == 0)
      break;
    closesocket(socket_handle);
    socket_handle = INVALID_SOCKET;
  }
  freeaddrinfo(addresses);
  if (socket_handle == INVALID_SOCKET) {
    error = WSAGetLastError();
    return false;
  }

  DWORD timeout = static_cast<DWORD>(timeout_milliseconds);
  setsockopt(socket_handle, SOL_SOCKET, SO_SNDTIMEO,
             reinterpret_cast<const char*>(&timeout), sizeof(timeout));
  setsockopt(socket_handle, SOL_SOCKET, SO_RCVTIMEO,
             reinterpret_cast<const char*>(&timeout), sizeof(timeout));

  std::string packet;
  packet.reserve(192 + request.size());
  packet += "POST " + path + " HTTP/1.1\r\n";
  packet += "Host: " + host + ":" + port + "\r\n";
  packet += "Content-Type: application/json\r\n";
  packet += "Content-Length: " + std::to_string(request.size()) + "\r\n";
  packet += "Connection: close\r\n\r\n";
  packet += request;

  size_t sent = 0;
  while (sent < packet.size()) {
    int sent_now = send(socket_handle, packet.data() + sent,
                        static_cast<int>(packet.size() - sent), 0);
    if (sent_now <= 0) {
      error = WSAGetLastError();
      closesocket(socket_handle);
      return false;
    }
    sent += static_cast<size_t>(sent_now);
  }

  std::string response;
  char buffer[4096];
  for (;;) {
    int received = recv(socket_handle, buffer, sizeof(buffer), 0);
    if (received > 0) {
      response.append(buffer, static_cast<size_t>(received));
    } else {
      if (received < 0)
        error = WSAGetLastError();
      break;
    }
    if (response.size() > 10u * 1024u * 1024u)
      break;
  }
  closesocket(socket_handle);

  const size_t header_end = response.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    error = ERROR_INVALID_DATA;
    return false;
  }
  status_line = response.substr(0, response.find("\r\n"));
  body = response.substr(header_end + 4);
  return true;
}

}  // namespace localhttp

class Engine {
 public:
  explicit Engine(std::wstring endpoint = EndpointFromEnvironment())
      : endpoint_(std::move(endpoint)) {}

  ~Engine() { Shutdown(); }

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void OnSnapshot(Snapshot snapshot) {
    LlmLog(L"engine OnSnapshot prefix=" + snapshot.prefix);
    if (!Enabled())
      return;
    // UI thread only publishes the snapshot; tree matching/expansion runs on
    // the worker thread so the IME never blocks on engine locks.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_snapshot_ = std::move(snapshot);
      ++generation_;
      pending_ = true;
    }
    EnsureWorker();
    condition_.notify_all();
  }

  void OnFocusLost() { Hide(); }

  void Hide() {
    std::lock_guard<std::mutex> lock(mutex_);
    visible_text_.clear();
    /* panel disabled */
  }

  void HideForLayoutChange() { Hide(); }

  enum class Decision { Ignore, Hide, HideAndEat, Accept };

  // [FEEDBACK-002] Report the ghost text that is actually on screen, together
  // with the tree's cumulative probability for that path. The RL side scores
  // the user's typing against THIS string, not against a tree of its own.
  void ReportShownAsync(const std::wstring& text, double cum) {
    if (!Enabled() || text.empty())
      return;
    std::string body = "{\"kind\":\"shown\",\"text\":";
    AppendJsonString(body, text);
    body += ",\"p\":" + std::to_string(cum) + ",\"keys\":" +
            std::to_string(keystrokes_.load()) + ",\"app\":\"" +
            AppNameForFeedback() + "\"}";
    std::wstring endpoint = FeedbackEndpoint();
    std::thread([endpoint, body]() {
      std::string status_line;
      std::string response;
      DWORD error = 0;
      localhttp::PostJson(endpoint, body, 1000, status_line, response, error);
    }).detach();
  }

  // [FEEDBACK-001] Report accept/shown events back to the engine service.
  void ReportFeedbackAsync(const std::string& kind, int chars) {
    if (!Enabled())
      return;
    std::string body = "{\"kind\":\"" + kind + "\",\"chars\":" +
                       std::to_string(chars) + ",\"keys\":" +
                       std::to_string(keystrokes_.load()) + ",\"app\":\"" +
                       AppNameForFeedback() + "\"}";
    std::wstring endpoint = FeedbackEndpoint();
    std::thread([endpoint, body]() {
      std::string status_line;
      std::string response;
      DWORD error = 0;
      localhttp::PostJson(endpoint, body, 1000, status_line, response, error);
    }).detach();
  }

  Decision HandleKey(WPARAM virtual_key, bool key_up, bool test_only) {
    (void)test_only;
    if (key_up)
      return Decision::Ignore;
    keystrokes_.fetch_add(1);

    std::lock_guard<std::mutex> lock(mutex_);

    if (virtual_key == VK_ESCAPE && VisibleLocked()) {
      visible_text_.clear();
      return Decision::HideAndEat;
    }

    if (virtual_key == VK_TAB && VisibleLocked()) {
      if (!test_only)
        committed_text_ = visible_text_;
      return Decision::Accept;
    }

    if (VisibleLocked()) {
      visible_text_.clear();
      visible_snapshot_ = Snapshot{};
      /* panel disabled */
      return Decision::Hide;
    }

    return Decision::Ignore;
  }

  // [TREE-010 PINYIN] Latest Rime composition (pinyin) for the current input
  // session. When non-empty, the visible continuation is constrained to
  // candidates whose pinyin starts with it.
  void SetPreedit(const std::wstring& preedit) {
    std::wstring normalized;
    for (wchar_t ch : preedit) {
      if (ch >= L'a' && ch <= L'z')
        normalized.push_back(ch);
      else if (ch >= L'A' && ch <= L'Z')
        normalized.push_back(static_cast<wchar_t>(ch - L'A' + L'a'));
      else if (ch >= L'0' && ch <= L'9')
        normalized.push_back(ch);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (normalized == preedit_)
      return;
    preedit_ = normalized;
    if (VisibleLocked())
      TryMatchLocked(visible_snapshot_);
  }

  void Prepare() {}

  bool Visible() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return VisibleLocked();
  }

  bool HasVisiblePrediction() {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return false;
    return !visible_text_.empty();
  }

  std::wstring VisiblePrediction() {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return L"";
    return visible_text_;
  }

  const std::wstring& EndpointForTrace() const { return endpoint_; }

  static std::string AppNameForFeedback() {
    static const std::string name = []() -> std::string {
      wchar_t path[MAX_PATH]{};
      DWORD size = GetModuleFileNameW(nullptr, path, MAX_PATH);
      if (size == 0)
        return std::string();
      std::wstring full(path, size);
      size_t slash = full.find_last_of(L"\\/");
      std::wstring base =
          (slash == std::wstring::npos) ? full : full.substr(slash + 1);
      return ToUtf8(base);
    }();
    return name;
  }

  std::wstring MetricsEndpoint() const {
    std::wstring wide(endpoint_);
    size_t position = wide.rfind(L"/completion");
    if (position == std::wstring::npos)
      return wide + L"/metrics";
    return wide.substr(0, position) + L"/metrics";
  }

  std::wstring FeedbackEndpoint() const {
    std::wstring wide(endpoint_);
    size_t position = wide.rfind(L"/completion");
    if (position == std::wstring::npos)
      return wide + L"/feedback";
    return wide.substr(0, position) + L"/feedback";
  }

  std::wstring TakeCommittedText() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(committed_text_);
  }

 private:
  static std::wstring EndpointFromEnvironment() {
    wchar_t buffer[512]{};
    DWORD size = GetEnvironmentVariableW(L"WEASEL_GHOST_URL", buffer,
                                         static_cast<DWORD>(std::size(buffer)));
    if (size > 0 && size < std::size(buffer))
      return buffer;
    return L"http://127.0.0.1:8081/completion";
  }

  static bool Enabled() {
    wchar_t value[8]{};
    DWORD size = GetEnvironmentVariableW(L"WEASEL_GHOST_DISABLE", value,
                                         static_cast<DWORD>(std::size(value)));
    return !(size > 0 && size < std::size(value) && wcscmp(value, L"1") == 0);
  }

  bool VisibleLocked() const { return !visible_text_.empty(); }

  bool TryMatchLocked(const Snapshot& snapshot) {
    if (tree_.Empty())
      return false;

    int matched = tree_.MatchPath(snapshot.prefix);
    if (matched < 0)
      return false;

    int end_node = matched;
    std::wstring remaining = tree_.BestSuffix(matched, preedit_, &end_node);
    auto stop = std::find_if(remaining.begin(), remaining.end(),
                             [](wchar_t ch) { return IsBranchStop(ch); });
    if (stop != remaining.end())
      remaining.resize(static_cast<size_t>(stop - remaining.begin()));
    if (remaining.size() > kMaxDisplayChars)
      remaining.resize(kMaxDisplayChars);
    bool was_empty = visible_text_.empty();
    bool changed = (remaining != visible_text_);
    visible_text_ = std::move(remaining);
    if (changed)
      LlmLog(L"engine visible=" + visible_text_);
    if (was_empty && !visible_text_.empty())
      ReportShownAsync(visible_text_, tree_.Cum(end_node));
    visible_snapshot_ = snapshot;
    committed_text_.clear();
    if (VisibleLocked()) {
      /* prediction consumed by IME candidate UI */
    } else {
      /* hidden */
    }
    return true;
  }

  void EnsureWorker() {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (worker_.joinable())
      return;
    worker_ = std::thread([this] { WorkerLoop(); });
    poller_ = std::thread([this] { MetricsLoop(); });
  }

  // [P3-3] Poll the engine service for its accept rate and adapt the trigger.
  void MetricsLoop() {
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_)
          return;
      }
      std::string body;
      std::string status_line;
      DWORD error = 0;
      if (localhttp::PostJson(MetricsEndpoint(), "{}", 1000, status_line, body,
                              error) &&
          status_line.find(" 200 ") != std::string::npos) {
        double accept_rate = -1.0;
        size_t position = body.find("\"accept_rate\":");
        if (position != std::string::npos)
          accept_rate = std::strtod(body.c_str() + position + 14, nullptr);
        int idle = kDefaultIdleMilliseconds;
        if (accept_rate >= 0.20)
          idle = kMinIdleMilliseconds;
        else if (accept_rate >= 0.0 && accept_rate < 0.05)
          idle = kMaxIdleMilliseconds;
        idle_milliseconds_.store(idle, std::memory_order_relaxed);
      }
      for (int tick = 0; tick < 300; ++tick) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_)
          return;
      }
    }
  }

  void WorkerLoop() {
    for (;;) {
      Snapshot snapshot;
      uint64_t generation = 0;
      bool has_pending = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(500),
                            [this] { return pending_ || !running_; });
        if (!running_)
          return;
        if (pending_) {
          snapshot = latest_snapshot_;
          generation = generation_;
          pending_ = false;
          has_pending = true;
        }
      }

      if (!has_pending)
        continue;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || generation != generation_)
          continue;
      }

      if (snapshot.prefix.size() < kMinContextChars) {
        // [P3-3] too little context: a prediction here is mostly noise.
        continue;
      }

      TraceLine(L"engine predicting after idle; pid="+std::to_wstring(GetCurrentProcessId())+L"; chars="+std::to_wstring(snapshot.prefix.size())+L"; prefix="+snapshot.prefix);

      std::this_thread::sleep_for(std::chrono::milliseconds(
          idle_milliseconds_.load(std::memory_order_relaxed)));

      // Match the new prefix against the existing tree first. If the user is
      // typing along an already predicted path, rebase the tree at the matched
      // node (new root, depth 0, base prefix advanced) so the remaining subtree
      // is reused instead of rebuilt and the depth budget is restored.
      int start_node = -1;
      {
        std::lock_guard<std::mutex> tree_lock(tree_mutex_);
        int matched = tree_.MatchPath(snapshot.prefix);
        if (matched >= 0) {
          int matched_depth = tree_.Depth(matched);
          start_node = tree_.PruneToRebased(matched, snapshot.prefix);
          LlmLog(L"engine rebase matched depth=" +
                 std::to_wstring(matched_depth));
        } else {
          tree_.Reset(snapshot.prefix);
          start_node = 0;
          LlmLog(L"engine reset tree");
        }
      }

      std::atomic<int> fetches{0};

      auto aborted = [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        return !running_ || generation != generation_;
      };
      auto publish = [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || generation != generation_)
          return;
        if (!TryMatchLocked(latest_snapshot_)) {
          visible_text_.clear();
          visible_snapshot_ = Snapshot{};
          /* panel disabled */
        }
      };
      auto expand = [&](int node, bool force_root) -> bool {
        std::wstring prompt;
        int depth = 0;
        {
          std::lock_guard<std::mutex> tree_lock(tree_mutex_);
          if (!tree_.NeedsExpansion(node, force_root))
            return false;
          prompt = tree_.BasePrefix() + tree_.PathFromRoot(node);
          depth = tree_.Depth(node);
        }
        LlmLog(L"engine fetch begin depth=" + std::to_wstring(depth) +
               L" prompt=" + prompt);
        std::vector<Candidate> candidates =
            FetchNextCandidates(prompt, depth == 0);
        LlmLog(L"engine fetch done count=" +
               std::to_wstring(candidates.size()));
        {
          std::lock_guard<std::mutex> tree_lock(tree_mutex_);
          tree_.SetCandidates(node, candidates);
        }
        ++fetches;
        return true;
      };

      // [TREE-014] expand a set of sibling nodes concurrently.
      auto expand_wave = [&](const std::vector<int>& nodes, bool force_root) {
        std::vector<std::future<void>> pending;
        pending.reserve(nodes.size());
        for (int node : nodes) {
          if (fetches.load() >= kMaxFetchesPerRun || aborted())
            break;
          pending.push_back(std::async(std::launch::async,
                                       [&, node, force_root]() {
            if (aborted())
              return;
            if (expand(node, force_root))
              publish();
          }));
        }
        for (auto& item : pending)
          item.wait();
      };

      // Phase A: expand the root, then complete the whole first level before
      // going deeper. Whatever the user types next is then almost certainly a
      // first-level node that already has a continuation, which is also what
      // gives the reinforcement signal something to hit. First-level nodes are
      // expanded best-first so the visible preview still grows immediately.
      if (expand(start_node, true))
        publish();

      std::vector<int> level_one = tree_.Children(start_node);
      std::sort(level_one.begin(), level_one.end(), [this](int a, int b) {
        return tree_.Cum(a) > tree_.Cum(b);
      });
      for (size_t offset = 0; offset < level_one.size();
           offset += kParallelRequests) {
        if (fetches.load() >= kMaxFetchesPerRun || aborted())
          break;
        size_t end = min(level_one.size(), offset + kParallelRequests);
        expand_wave(std::vector<int>(level_one.begin() + offset,
                                     level_one.begin() + end),
                    false);
      }

      // Phase B: once the first level is complete, walk the most probable
      // chain down to full depth so a long continuation is always available.
      int chain = tree_.BestChild(start_node);
      for (int depth = 1; depth < kTreeMaxDepth && chain >= 0; ++depth) {
        if (fetches >= kMaxFetchesPerRun || aborted())
          break;
        if (expand(chain, false))
          publish();
        chain = tree_.BestChild(chain);
      }

      // Phase C: widen whatever is left, highest cumulative probability first,
      // until the per-run request budget is exhausted.
      std::vector<int> frontier = tree_.Frontier();
      while (fetches.load() < kMaxFetchesPerRun && !frontier.empty() &&
             !aborted()) {
        std::vector<int> wave;
        while (wave.size() < static_cast<size_t>(kParallelRequests) &&
               !frontier.empty()) {
          size_t best_index = 0;
          for (size_t i = 1; i < frontier.size(); ++i) {
            if (tree_.Cum(frontier[i]) > tree_.Cum(frontier[best_index]))
              best_index = i;
          }
          wave.push_back(frontier[best_index]);
          frontier.erase(frontier.begin() + best_index);
        }
        expand_wave(wave, false);
        for (int target : wave) {
          std::vector<int> children = tree_.Children(target);
          for (int child : children) {
            if (tree_.NeedsExpansion(child, false))
              frontier.push_back(child);
          }
        }
      }

      publish();
    }
  }

  void Shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    pending_ = true;
    visible_text_.clear();
    visible_snapshot_ = Snapshot{};
    /* panel disabled */
    }
    condition_.notify_all();
    std::lock_guard<std::mutex> worker_lock(worker_mutex_);
    if (worker_.joinable())
      worker_.join();
    if (poller_.joinable())
      poller_.join();
  }

  static std::string ToUtf8(const std::wstring& value) {
    if (value.empty())
      return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                                   static_cast<int>(value.size()), nullptr, 0,
                                   nullptr, nullptr);
    std::string result(size, '\0');
    if (size > 0) {
      WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                          static_cast<int>(value.size()), result.data(), size,
                          nullptr, nullptr);
    }
    return result;
  }

  static std::wstring FromUtf8(const std::string& value) {
    if (value.empty())
      return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0);
    std::wstring result(size, L'\0');
    if (size > 0) {
      MultiByteToWideChar(CP_UTF8, 0, value.data(),
                          static_cast<int>(value.size()), result.data(), size);
    }
    return result;
  }

  static void AppendJsonString(std::string& target, const std::wstring& value) {
    target.push_back('"');
    const std::string utf8 = ToUtf8(value);
    for (unsigned char byte : utf8) {
      switch (byte) {
        case '"': target += "\\\""; break;
        case '\\': target += "\\\\"; break;
        case '\n': target += "\\n"; break;
        case '\r': target += "\\r"; break;
        case '\t': target += "\\t"; break;
        default:
          if (byte < 0x20) {
            char buffer[8];
            snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
            target += buffer;
          } else {
            target.push_back(static_cast<char>(byte));
          }
      }
    }
    target.push_back('"');
  }

  static std::wstring ExtractJsonString(const std::wstring& json,
                                        const std::wstring& key) {
    const std::wstring needle = L"\"" + key + L"\":";
    size_t position = json.find(needle);
    if (position == std::wstring::npos)
      return {};
    position += needle.size();
    while (position < json.size() && json[position] == L' ')
      ++position;
    if (position >= json.size() || json[position] != L'"')
      return {};
    ++position;
    std::wstring result;
    while (position < json.size()) {
      wchar_t current = json[position++];
      if (current != L'\\') {
        if (current == L'"')
          break;
        result.push_back(current);
        continue;
      }
      if (position >= json.size())
        break;
      wchar_t escaped = json[position++];
      switch (escaped) {
        case L'"': result.push_back(L'"'); break;
        case L'\\': result.push_back(L'\\'); break;
        case L'/': result.push_back(L'/'); break;
        case L'n': result.push_back(L'\n'); break;
        case L'r': result.push_back(L'\r'); break;
        case L't': result.push_back(L'\t'); break;
        case L'b': result.push_back(L'\b'); break;
        case L'f': result.push_back(L'\f'); break;
        case L'u': {
          if (position + 4 > json.size())
            return result;
          result.push_back(
              static_cast<wchar_t>(wcstoul(json.substr(position, 4).c_str(),
                                           nullptr, 16)));
          position += 4;
          break;
        }
        default: result.push_back(escaped); break;
      }
    }
    return result;
  }

  static std::wstring CleanText(std::wstring value, int max_chars) {
    const std::wstring forbidden[] = {L"```", L"\r", L"\n"};
    for (const auto& item : forbidden) {
      size_t position;
      while ((position = value.find(item)) != std::wstring::npos)
        value.erase(position, item.size());
    }
    size_t first = value.find_first_not_of(L" \t");
    size_t last = value.find_last_not_of(L" \t");
    if (first == std::wstring::npos)
      return {};
    value = value.substr(first, last - first + 1);
    if (value.size() > static_cast<size_t>(max_chars))
      value.resize(static_cast<size_t>(max_chars));
    return value;
  }

  static std::wstring TruncateAtBranchStop(std::wstring value) {
    for (size_t i = 0; i < value.size(); ++i) {
      wchar_t ch = value[i];
      if (ch == L'.' || ch == L',' || ch == L'!' || ch == L'?' ||
          ch == L'。' || ch == L'，' || ch == L'！' || ch == L'？' ||
          ch == L'、' || ch == L'；' || ch == L'：') {
        return value.substr(0, i + 1);
      }
    }
    return value;
  }

  static double ExtractJsonNumber(const std::wstring& json,
                                  const std::wstring& key) {
    const std::wstring needle = L"\"" + key + L"\":";
    size_t position = json.find(needle);
    if (position == std::wstring::npos)
      return 0.0;
    position += needle.size();
    while (position < json.size() &&
           (json[position] == L' ' || json[position] == L'\t'))
      ++position;
    size_t end = position;
    while (end < json.size() &&
           (iswdigit(json[end]) || json[end] == L'-' || json[end] == L'+' ||
            json[end] == L'.' || json[end] == L'e' || json[end] == L'E'))
      ++end;
    if (end <= position)
      return 0.0;
    return wcstod(json.substr(position, end - position).c_str(), nullptr);
  }

  static std::vector<Candidate> ExtractTopCandidates(
      const std::wstring& body) {
    std::vector<Candidate> candidates;
    size_t completion_position =
        body.find(L"\"completion_probabilities\"");
    TraceLine(L"parser completion_pos="+std::to_wstring(completion_position));
    if (completion_position == std::wstring::npos)
      return candidates;

    size_t top_position = body.find(L"\"top_probs\"", completion_position);
    TraceLine(L"parser top_pos="+std::to_wstring(top_position));
    if (top_position == std::wstring::npos)
      return candidates;

    size_t array_start = body.find(L'[', top_position);
    if (array_start == std::wstring::npos)
      return candidates;
    size_t array_end = std::wstring::npos;
    int bracket_depth = 0;
    for (size_t position = array_start; position < body.size(); ++position) {
      if (body[position] == L'[')
        ++bracket_depth;
      if (body[position] == L']' && --bracket_depth == 0) {
        array_end = position;
        break;
      }
    }
    if (array_end == std::wstring::npos || array_end <= array_start)
      return candidates;

    size_t array_object = body.find(L'{', array_start + 1);
    TraceLine(L"parser array="+std::to_wstring(array_start)+
              L"; end="+std::to_wstring(array_end)+
              L"; object="+std::to_wstring(array_object)+
              L"; around="+body.substr(max(0, static_cast<int>(top_position)),
                                       min(500, static_cast<int>(body.size()-top_position))));
    size_t cursor = array_start + 1;
    while (cursor < array_end) {
      size_t object_start = body.find(L'{', cursor);
      if (object_start == std::wstring::npos || object_start >= array_end)
        break;
      size_t object_end = body.find(L'}', object_start);
      if (object_end == std::wstring::npos || object_end >= array_end)
        break;

      std::wstring object =
          body.substr(object_start, object_end - object_start + 1);
      std::wstring token = ExtractJsonString(object, L"token");
      double probability = ExtractJsonNumber(object, L"prob");
      std::wstring pinyin = ExtractJsonString(object, L"pinyin");
      TraceLine(L"parser object; token="+token+L"; prob="+
                std::to_wstring(probability));
      if (!token.empty() && probability > 0.0 &&
          static_cast<DWORD>(token.front()) >= 0x20) {
        bool duplicate = false;
        for (auto& candidate : candidates) {
          if (candidate.ch == token.front()) {
            candidate.prob = max(candidate.prob, probability);
            if (!pinyin.empty())
              candidate.pinyin = pinyin;
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          Candidate entry;
          entry.ch = token.front();
          entry.prob = probability;
          entry.pinyin = pinyin;
          candidates.push_back(std::move(entry));
        }
      }
      cursor = object_end + 1;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right) {
                return left.prob > right.prob;
              });
    if (candidates.size() > static_cast<size_t>(kTreeWidth))
      candidates.resize(static_cast<size_t>(kTreeWidth));
    return candidates;
  }

  // [TREE-015] |use_pinyin| is true only for the node the user is actually at
  // (depth 0). Deeper nodes must NOT be pinyin-constrained, otherwise the
  // chain keeps re-selecting the same character and the preview degenerates
  // into "天天天天...".
  std::vector<Candidate> FetchNextCandidates(const std::wstring& prompt,
                                             bool use_pinyin) {
    std::string request = "{\"prompt\":";
    AppendJsonString(request, prompt);
    std::wstring preedit;
    if (use_pinyin) {
      std::lock_guard<std::mutex> lock(mutex_);
      preedit = preedit_;
    }
    request += ",\"n_predict\":1,\"n_probs\":";
    request += std::to_string(preedit.empty() ? kTreeWidth : kPinyinProbeWidth);
    if (!preedit.empty()) {
      request += ",\"pinyin\":";
      AppendJsonString(request, preedit);
    }
    request += ",\"temperature\":1.0,\"top_k\":0,\"top_p\":1.0,\"min_p\":0.0,";
    request += kRepeatPenaltyJson;
    request += "\"app\":\"" + AppNameForFeedback() + "\",";
    request += "\"post_sampling_probs\":true}";

    std::string body;
    std::string status_line;
    DWORD error = 0;
    if (!localhttp::PostJson(endpoint_, request,
                             kPredictionTimeoutMilliseconds, status_line,
                             body, error)) {
      TraceLine(L"tree local http failed; error=" +
                std::to_wstring(error) + L"; status=" +
                std::wstring(status_line.begin(), status_line.end()));
      return {};
    }
    if (status_line.find(" 200 ") == std::string::npos) {
      TraceLine(L"tree local http status; line=" +
                std::wstring(status_line.begin(), status_line.end()));
      return {};
    }

    auto candidates = ExtractTopCandidates(FromUtf8(body));
    TraceLine(L"tree local http done; bytes=" +
              std::to_wstring(body.size()) + L"; parsed=" +
              std::to_wstring(candidates.size()));
    return candidates;
  }

  std::wstring CompleteOnce(const std::wstring& prompt, int seed,
                            float temperature) {
    std::string request = "{\"prompt\":";
    AppendJsonString(request, prompt);
    request += ",\"n_predict\":24,\"seed\":";
    request += std::to_string(seed);
    request += ",\"temperature\":";
    request += std::to_string(temperature);
    request += ",\"top_k\":60,\"top_p\":0.9,\"stop\":[\"\\n\",\"\\r\"]}";
    TraceLine(L"http request built; prefix_chars=" +
              std::to_wstring(prompt.size()) +
              L", request_bytes=" + std::to_wstring(request.size()));

    URL_COMPONENTS components{};
    components.dwStructSize = sizeof(components);
    wchar_t host[256]{};
    wchar_t path[1024]{};
    components.lpszHostName = host;
    components.dwHostNameLength = std::size(host);
    components.lpszUrlPath = path;
    components.dwUrlPathLength = std::size(path);
    if (!WinHttpCrackUrl(endpoint_.c_str(),
                         static_cast<DWORD>(endpoint_.size()), 0, &components))
    {
      TraceLine(L"http crack url failed; error=" +
                std::to_wstring(GetLastError()));
      return {};
    }

    HINTERNET session = WinHttpOpen(L"WeaselGhost/1.0",
                                    WINHTTP_ACCESS_TYPE_NO_PROXY,
                                    WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session)
    {
      TraceLine(L"http session failed; error=" +
                std::to_wstring(GetLastError()));
      return {};
    }
    WinHttpSetTimeouts(session, kPredictionTimeoutMilliseconds,
                       kPredictionTimeoutMilliseconds,
                       kPredictionTimeoutMilliseconds,
                       kPredictionTimeoutMilliseconds);
    HINTERNET connection = WinHttpConnect(
        session, host, components.nPort, 0);
    if (!connection)
      TraceLine(L"http connect failed; error=" +
                std::to_wstring(GetLastError()));
    HINTERNET http_request = nullptr;
    if (connection) {
      http_request = WinHttpOpenRequest(
          connection, L"POST", path, NULL, WINHTTP_NO_REFERER,
          WINHTTP_DEFAULT_ACCEPT_TYPES,
          (components.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0);
    }
    if (!http_request)
      TraceLine(L"http request handle failed; error=" +
                std::to_wstring(GetLastError()));

    std::wstring result;
    if (http_request) {
      const std::wstring headers =
          L"Content-Type: application/json\r\nContent-Length: " +
          std::to_wstring(request.size()) + L"\r\n";
      if (WinHttpSendRequest(http_request, headers.c_str(),
                             static_cast<DWORD>(headers.size()),
                             request.data(),
                             static_cast<DWORD>(request.size()),
                             static_cast<DWORD>(request.size()), 0) &&
          WinHttpReceiveResponse(http_request, nullptr)) {
        TraceLine(L"http send/receive succeeded");
        DWORD status = 0;
        DWORD status_size = sizeof(status);
        DWORD available = 0;
        std::string body;
        if (WinHttpQueryHeaders(http_request,
                                WINHTTP_QUERY_STATUS_CODE |
                                    WINHTTP_QUERY_FLAG_NUMBER,
                                WINHTTP_HEADER_NAME_BY_INDEX, &status,
                                &status_size, WINHTTP_NO_HEADER_INDEX) &&
            status == HTTP_STATUS_OK) {
          for (;;) {
            if (!WinHttpQueryDataAvailable(http_request, &available) ||
                available == 0)
              break;
            std::string chunk(available, '\0');
            DWORD read_size = 0;
            if (!WinHttpReadData(http_request, chunk.data(), available,
                                 &read_size) ||
                read_size == 0)
              break;
            chunk.resize(read_size);
            body += chunk;
          }
          result = ExtractJsonString(FromUtf8(body), L"content");
          TraceLine(L"http response ok; bytes=" +
                    std::to_wstring(body.size()) +
                    L", utf16_chars=" +
                    std::to_wstring(result.size()));
        } else {
          TraceLine(L"http response status not ok; status=" +
                    std::to_wstring(status));
        }
      } else {
        TraceLine(L"http send/receive failed; error=" +
                  std::to_wstring(GetLastError()));
      }
      WinHttpCloseHandle(http_request);
    }
    if (connection)
      WinHttpCloseHandle(connection);
    if (session)
      WinHttpCloseHandle(session);

    result = TruncateAtBranchStop(result);
    result = CleanText(result, kTreeBranchChars);
    TraceLine(L"cleaned result chars=" + std::to_wstring(result.size()));
    if (!result.empty() && prompt.size() >= result.size() &&
        prompt.compare(prompt.size() - result.size(),
                                result.size(), result) == 0)
    {
      TraceLine(L"cleaned result ignored as prefix suffix");
      result.clear();
    }
    return result;
  }

  std::wstring endpoint_;
  Panel panel_;
  mutable std::mutex mutex_;
  std::mutex tree_mutex_;
  std::mutex worker_mutex_;
  std::condition_variable condition_;
  std::thread worker_;
  std::thread poller_;
  Snapshot latest_snapshot_;
  Snapshot visible_snapshot_;
  CandidateTree tree_;
  std::wstring visible_text_;
  std::wstring committed_text_;
  std::wstring preedit_;
  std::atomic<long long> keystrokes_{0};
  std::atomic<int> idle_milliseconds_{kDefaultIdleMilliseconds};
  uint64_t generation_ = 0;
  bool pending_ = false;
  bool running_ = true;
};

}  // namespace ghost

}  // namespace weasel
