#pragma once

#include <Windows.h>

#include <cstdint>
#include <cwctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace weasel {
namespace inline_ghost {

inline std::string Utf8FromWide(const std::wstring& text) {
  if (text.empty())
    return std::string();
  int bytes = ::WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                    static_cast<int>(text.size()), nullptr, 0,
                                    nullptr, nullptr);
  if (bytes <= 0)
    return std::string();
  std::string output(static_cast<size_t>(bytes), '\0');
  ::WideCharToMultiByte(CP_UTF8, 0, text.data(),
                        static_cast<int>(text.size()), &output[0], bytes,
                        nullptr, nullptr);
  return output;
}

inline std::wstring WideFromUtf8(const std::string& text) {
  if (text.empty())
    return std::wstring();
  int chars = ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                    static_cast<int>(text.size()), nullptr, 0);
  if (chars <= 0)
    return std::wstring();
  std::wstring output(static_cast<size_t>(chars), L'\0');
  ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                        static_cast<int>(text.size()), &output[0], chars);
  return output;
}

inline bool IsInvisibleCodePoint(wchar_t value) {
  switch (value) {
    case 0x200B:  // zero width space
    case 0x200C:  // zero width non-joiner
    case 0x200D:  // zero width joiner
    case 0x2060:  // word joiner
    case 0x2063:  // invisible separator (ghost open sentinel)
    case 0x2064:  // invisible plus (ghost close sentinel)
    case 0xFEFF:  // BOM / zero width no-break space
    case 0x202A:  // LRE
    case 0x202B:  // RLE
    case 0x202C:  // PDF
    case 0x202D:  // LRO
    case 0x202E:  // RLO
      return true;
    default:
      return false;
  }
}

inline std::wstring CleanContext(const std::wstring& text) {
  std::wstring cleaned;
  cleaned.reserve(text.size());
  for (const wchar_t value : text) {
    if (!IsInvisibleCodePoint(value))
      cleaned.push_back(value);
  }
  while (!cleaned.empty()) {
    const wchar_t tail = cleaned.back();
    if ((tail >= L'A' && tail <= L'Z') || (tail >= L'a' && tail <= L'z'))
      cleaned.pop_back();
    else
      break;
  }
  while (!cleaned.empty() && iswspace(cleaned.back()) != 0)
    cleaned.pop_back();
  return cleaned;
}

inline uint64_t HashContext(const std::wstring& text) {
  const std::wstring cleaned = CleanContext(text);
  const std::string bytes = Utf8FromWide(cleaned);
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline void TraceLine(const std::wstring& line) {
  wchar_t flag[8]{};
  const DWORD size =
      ::GetEnvironmentVariableW(L"WEASEL_INLINE_GHOST_TRACE", flag, 8);
  if (size == 0 || size >= 8 || wcscmp(flag, L"1") != 0)
    return;
  wchar_t temp[MAX_PATH]{};
  ::ExpandEnvironmentStringsW(L"%TEMP%\\rime.weasel", temp, MAX_PATH);
  ::CreateDirectoryW(temp, nullptr);
  FILE* file = nullptr;
  if (_wfopen_s(&file, (std::wstring(temp) + L"\\inline-ghost-debug.log").c_str(),
                L"a,ccs=UTF-8") == 0 && file != nullptr) {
    fwprintf(file, L"[%lu] %s\n", ::GetCurrentThreadId(), line.c_str());
    fclose(file);
  }
}

inline std::wstring TsContextFilePath() {
  wchar_t appdata[MAX_PATH]{};
  const DWORD length =
      ::GetEnvironmentVariableW(L"APPDATA", appdata, MAX_PATH);
  if (length == 0 || length >= MAX_PATH)
    return L"";
  return std::wstring(appdata) + L"\\Rime\\llm_tsf_context.log";
}

constexpr wchar_t kGhostOpenSentinel = 0x2063;  // INVISIBLE SEPARATOR
constexpr wchar_t kGhostCloseSentinel = 0x2064;  // INVISIBLE PLUS

inline size_t GhostSentinelRemoveCount(const std::wstring& text) {
  size_t count = 0;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t open = text.find(kGhostOpenSentinel, pos);
    if (open == std::wstring::npos)
      break;
    const size_t close = text.find(kGhostCloseSentinel, open + 1);
    if (close == std::wstring::npos)
      break;
    count += close - open + 1;
    pos = close + 1;
  }
  return count;
}

inline std::wstring CleanGhostSentinelBlocks(const std::wstring& text) {
  std::wstring cleaned = text;
  while (true) {
    const size_t open = cleaned.find(kGhostOpenSentinel);
    if (open == std::wstring::npos)
      break;
    const size_t close = cleaned.find(kGhostCloseSentinel, open + 1);
    if (close == std::wstring::npos)
      break;
    cleaned.erase(open, close - open + 1);
  }
  return cleaned;
}

inline size_t TrailingGhostMarkerCount(const std::wstring& text) {
  size_t count = 0;
  size_t pos = text.size();
  while (pos > 0 && text[pos - 1] == L'>') {
    size_t end = pos - 1;
    size_t start = end;
    while (start > 0 && text[start - 1] != L'<') --start;
    if (start == 0 || start == end)
      break;
    if (text[start - 1] != L'<')
      break;
    // Reject text containing another marker inside this segment.
    const size_t segment_len = end - start;
    if (segment_len == 0 || segment_len > 200)
      break;
    bool inner_marker = false;
    for (size_t i = start; i < end; ++i) {
      if (text[i] == L'<') {
        inner_marker = true;
        break;
      }
    }
    if (inner_marker)
      break;
    count += segment_len + 2;
    pos = start - 1;
  }
  return count;
}

inline std::wstring CleanGhostMarkers(const std::wstring& text) {
  const size_t count = TrailingGhostMarkerCount(text);
  if (count == 0)
    return text;
  return text.substr(0, text.size() - count);
}

inline std::wstring GhostFilePath() {
  wchar_t appdata[MAX_PATH]{};
  const DWORD length =
      ::GetEnvironmentVariableW(L"APPDATA", appdata, MAX_PATH);
  if (length == 0 || length >= MAX_PATH)
    return L"";
  return std::wstring(appdata) + L"\\Rime\\llm_inline_ghost.txt";
}

inline bool ReadSuggestion(const std::wstring& path, uint64_t* context_hash,
                           std::wstring* suggestion) {
  if (path.empty() || context_hash == nullptr || suggestion == nullptr)
    return false;
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file.is_open())
    return false;
  std::string hash_line;
  std::string text_line;
  if (!std::getline(file, hash_line))
    return false;
  if (!std::getline(file, text_line))
    return false;
  if (!text_line.empty() && text_line.back() == '\r')
    text_line.pop_back();
  uint64_t hash = 0;
  std::istringstream parser(hash_line);
  parser >> std::hex >> hash;
  if (parser.fail())
    return false;
  *context_hash = hash;
  *suggestion = WideFromUtf8(text_line);
  return !suggestion->empty();
}

inline void ClearGhostFile() {
  const std::wstring path = GhostFilePath();
  if (!path.empty())
    ::DeleteFileW(path.c_str());
}

}  // namespace inline_ghost
}  // namespace weasel