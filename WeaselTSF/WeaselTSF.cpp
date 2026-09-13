#include "stdafx.h"

#include <WeaselIPCData.h>
#include <thread>
#include <shellapi.h>
#include <tlhelp32.h>
#include "WeaselTSF.h"
#include "EditSession.h"
#include "CandidateList.h"
#include "LanguageBar.h"
#include "Compartment.h"
#include "ResponseParser.h"
#include "GhostEngine.h"

static void error_message(const WCHAR* msg) {
  static DWORD next_tick = 0;
  DWORD now = GetTickCount();
  if (now > next_tick) {
    next_tick = now + 10000;  // (ms)
    MessageBox(NULL, msg, get_weasel_ime_name().c_str(), MB_ICONERROR | MB_OK);
  }
}

WeaselTSF::WeaselTSF() {
  LlmLog(L"WeaselTSF ctor");
  _cRef = 1;

  m_ghostEngine = std::make_unique<weasel::ghost::Engine>();
  LlmLog(L"WeaselTSF ghost engine created");

  _dwThreadMgrEventSinkCookie = TF_INVALID_COOKIE;

  _dwTextEditSinkCookie = TF_INVALID_COOKIE;
  _dwTextLayoutSinkCookie = TF_INVALID_COOKIE;
  _dwThreadFocusSinkCookie = TF_INVALID_COOKIE;
  _fTestKeyDownPending = FALSE;
  _fTestKeyUpPending = FALSE;

  _fCUASWorkaroundTested = _fCUASWorkaroundEnabled = FALSE;

  _cand = new CCandidateList(this);

  DllAddRef();
}

WeaselTSF::~WeaselTSF() {
  m_ghostEngine.reset();
  DllRelease();
}

STDMETHODIMP WeaselTSF::QueryInterface(REFIID riid, void** ppvObject) {
  if (ppvObject == NULL)
    return E_INVALIDARG;

  *ppvObject = NULL;

  if (IsEqualIID(riid, IID_IUnknown) ||
      IsEqualIID(riid, IID_ITfTextInputProcessor))
    *ppvObject = (ITfTextInputProcessor*)this;
  else if (IsEqualIID(riid, IID_ITfTextInputProcessorEx))
    *ppvObject = (ITfTextInputProcessorEx*)this;
  else if (IsEqualIID(riid, IID_ITfThreadMgrEventSink))
    *ppvObject = (ITfThreadMgrEventSink*)this;
  else if (IsEqualIID(riid, IID_ITfTextEditSink))
    *ppvObject = (ITfTextEditSink*)this;
  else if (IsEqualIID(riid, IID_ITfTextLayoutSink))
    *ppvObject = (ITfTextLayoutSink*)this;
  else if (IsEqualIID(riid, IID_ITfKeyEventSink))
    *ppvObject = (ITfKeyEventSink*)this;
  else if (IsEqualIID(riid, IID_ITfCompositionSink))
    *ppvObject = (ITfCompositionSink*)this;
  else if (IsEqualIID(riid, IID_ITfEditSession))
    *ppvObject = (ITfEditSession*)this;
  else if (IsEqualIID(riid, IID_ITfThreadFocusSink))
    *ppvObject = (ITfThreadFocusSink*)this;
  else if (IsEqualIID(riid, IID_ITfDisplayAttributeProvider))
    *ppvObject = (ITfDisplayAttributeProvider*)this;

  if (*ppvObject) {
    AddRef();
    return S_OK;
  }
  return E_NOINTERFACE;
}

STDMETHODIMP_(ULONG) WeaselTSF::AddRef() {
  return ++_cRef;
}

STDMETHODIMP_(ULONG) WeaselTSF::Release() {
  LONG cr = --_cRef;

  assert(_cRef >= 0);

  if (_cRef == 0)
    delete this;

  return cr;
}

STDMETHODIMP WeaselTSF::Activate(ITfThreadMgr* pThreadMgr,
                                 TfClientId tfClientId) {
  return ActivateEx(pThreadMgr, tfClientId, 0U);
}

STDMETHODIMP WeaselTSF::Deactivate() {
  LlmLog(L"Deactivate");
  m_client.EndSession();

  _InitTextEditSink(com_ptr<ITfDocumentMgr>());

  _UninitThreadMgrEventSink();

  _UninitKeyEventSink();
  _UninitPreservedKey();

  _UninitLanguageBar();

  _UninitCompartment();

  _UninitThreadMgrEventSink();

  _pThreadMgr = NULL;

  _tfClientId = TF_CLIENTID_NULL;

  _cand->DestroyAll();

  return S_OK;
}

STDMETHODIMP WeaselTSF::ActivateEx(ITfThreadMgr* pThreadMgr,
                                   TfClientId tfClientId,
                                   DWORD dwFlags) {
  LlmLog(L"ActivateEx begin");
  com_ptr<ITfDocumentMgr> pDocMgrFocus;
  _activateFlags = dwFlags;

  _pThreadMgr = pThreadMgr;
  _tfClientId = tfClientId;

  if (!_InitThreadMgrEventSink())
    goto ExitError;

  if ((_pThreadMgr->GetFocus(&pDocMgrFocus) == S_OK) &&
      (pDocMgrFocus != NULL)) {
    _InitTextEditSink(pDocMgrFocus);
  }

  if (!_InitKeyEventSink())
    goto ExitError;

  // if (!_InitDisplayAttributeGuidAtom())
  //	goto ExitError;
  //	some app might init failed because it not provide DisplayAttributeInfo,
  // like some opengl stuff
  _InitDisplayAttributeGuidAtom();

  if (!_InitPreservedKey())
    goto ExitError;

  if (!_InitLanguageBar())
    goto ExitError;

  if (!_IsKeyboardOpen())
    _SetKeyboardOpen(TRUE);

  if (!_InitCompartment())
    goto ExitError;
  if (!_InitThreadFocusSink())
    goto ExitError;

  _EnsureServerConnected();

  return S_OK;

ExitError:
  Deactivate();
  return E_FAIL;
}

STDMETHODIMP WeaselTSF::OnSetThreadFocus() {
  std::wstring _ToggleImeOnOpenClose{};
  RegGetStringValue(HKEY_CURRENT_USER, L"Software\\Rime\\weasel",
                    L"ToggleImeOnOpenClose", _ToggleImeOnOpenClose);
  _isToOpenClose = (_ToggleImeOnOpenClose == L"yes");
  if (m_client.Echo()) {
    m_client.ProcessKeyEvent(0);
    weasel::ResponseParser parser(NULL, NULL, &_status, NULL, &_cand->style());
    bool ok = m_client.GetResponseData(std::ref(parser));
    if (ok)
      _UpdateLanguageBar(_status);
  }
  return S_OK;
}
STDMETHODIMP WeaselTSF::OnKillThreadFocus() {
  _AbortComposition();
  return S_OK;
}
BOOL WeaselTSF::_InitThreadFocusSink() {
  com_ptr<ITfSource> pSource;
  if (FAILED(_pThreadMgr->QueryInterface(&pSource)))
    return FALSE;
  if (FAILED(pSource->AdviseSink(IID_ITfThreadFocusSink,
                                 (ITfThreadFocusSink*)this,
                                 &_dwThreadFocusSinkCookie)))
    return FALSE;
  return TRUE;
}
void WeaselTSF::_UninitThreadFocusSink() {
  com_ptr<ITfSource> pSource;
  if (FAILED(_pThreadMgr->QueryInterface(&pSource)))
    return;
  if (FAILED(pSource->UnadviseSink(_dwThreadFocusSinkCookie)))
    return;
}

STDMETHODIMP WeaselTSF::OnActivated(REFCLSID clsid,
                                    REFGUID guidProfile,
                                    BOOL isActivated) {
  if (!IsEqualCLSID(clsid, c_clsidTextService)) {
    return S_OK;
  }

  if (isActivated) {
    _ShowLanguageBar(TRUE);
    _UpdateLanguageBar(_status);
  } else {
    _DeleteCandidateList();
    _ShowLanguageBar(FALSE);
  }
  return S_OK;
}

void WeaselTSF::_Reconnect() {
  m_client.Disconnect();
  m_client.Connect(NULL);
  m_client.StartSession();
  weasel::ResponseParser parser(NULL, NULL, &_status, NULL, &_cand->style());
  bool ok = m_client.GetResponseData(std::ref(parser));
  if (ok) {
    _UpdateLanguageBar(_status);
  }
}

static unsigned int retry = 0;

bool WeaselTSF::_EnsureServerConnected() {
  if (!m_client.Echo()) {
    _Reconnect();
    retry++;
    if (retry >= 6) {
      HANDLE hMutex = CreateMutex(NULL, TRUE, L"WeaselDeployerExclusiveMutex");
      const auto count_server_process = []() -> int {
        int count = 0;
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap == INVALID_HANDLE_VALUE)
          return 0;
        PROCESSENTRY32 pe;
        pe.dwSize = sizeof(pe);
        if (Process32First(snap, &pe)) {
          do {
            if (_wcsicmp(pe.szExeFile, L"WeaselServer.exe") == 0)
              count++;
          } while (Process32Next(snap, &pe));
        }
        CloseHandle(snap);
        return count;
      };
      if (!m_client.Echo() && GetLastError() != ERROR_ALREADY_EXISTS &&
          !count_server_process()) {
        std::wstring dir = _GetRootDir();
        std::thread th([dir, this]() {
          ShellExecuteW(NULL, L"open", (dir + L"\\start_service.bat").c_str(),
                        NULL, dir.c_str(), SW_HIDE);
          // wait 500ms, then reconnect
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          _Reconnect();
        });
        th.detach();
      }
      if (hMutex) {
        CloseHandle(hMutex);
      }
      retry = 0;
    }
    return (m_client.Echo() != 0);
  } else {
    return true;
  }
}

// [GHOST-TSF-011 SNAPSHOT-COLLECTOR] in-process engine variant
// [GHOST-FIX-012] 光标框判据与 HRESULT 文本化。
// 判据必须是"矩形退化"而不是 "left==0 && top==0"：要么右<=左且下<=上（全零/退化），
// 要么整个矩形落在虚拟屏幕之外，才算不可用。屏幕原点附近的合法位置不能被误杀。
namespace {
inline BOOL GhostRectUsable(const RECT& r) {
  if (r.right <= r.left && r.bottom <= r.top)
    return FALSE;
  const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
  const int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
  RECT virt{vx, vy, vx + GetSystemMetrics(SM_CXVIRTUALSCREEN),
            vy + GetSystemMetrics(SM_CYVIRTUALSCREEN)};
  RECT out{};
  return IntersectRect(&out, &r, &virt);
}

inline std::wstring GhostHrText(HRESULT hr) {
  wchar_t buf[16]{};
  swprintf_s(buf, L"0x%08X", static_cast<unsigned>(hr));
  return buf;
}
}  // namespace

bool WeaselTSF::_ReadGhostPrefix(ITfContext* pContext,
                                TfEditCookie ecReadOnly,
                                std::wstring* prefix, LONG* caret,
                                RECT* caret_rect) {
  TF_SELECTION selection{};
  ULONG selection_count = 0;
  if (FAILED(pContext->GetSelection(ecReadOnly, TF_DEFAULT_SELECTION, 1,
                                    &selection, &selection_count)) ||
      selection_count < 1 || selection.range == nullptr) {
    weasel::ghost::TraceLine(L"snapshot hidden: no selection");
    return false;
  }

  BOOL empty = TRUE;
  if (FAILED(selection.range->IsEmpty(ecReadOnly, &empty)) || !empty) {
    weasel::ghost::TraceLine(L"snapshot hidden: non-empty selection");
    return false;
  }

  com_ptr<ITfRangeACP> acp_range;
  LONG caret_pos = -1;
  LONG selection_length = 0;
  if (FAILED(selection.range->QueryInterface(IID_ITfRangeACP,
                                             (LPVOID*)&acp_range)) ||
      acp_range == nullptr ||
      FAILED(acp_range->GetExtent(&caret_pos, &selection_length)) ||
      caret_pos < 0 || selection_length != 0) {
    weasel::ghost::TraceLine(L"snapshot hidden: ACP range unavailable");
    return false;
  }

  LONG context_start = max(
      0L, static_cast<LONG>(caret_pos - weasel::ghost::kDefaultContextChars));
  LONG context_length = caret_pos - context_start;
  if (FAILED(acp_range->SetExtent(context_start, context_length))) {
    weasel::ghost::TraceLine(L"snapshot hidden: ACP context set failed");
    return false;
  }

  wchar_t buffer[weasel::ghost::kDefaultContextChars]{};
  ULONG prefix_length = 0;
  if (FAILED(acp_range->GetText(ecReadOnly, 0, buffer,
                                weasel::ghost::kDefaultContextChars,
                                &prefix_length))) {
    weasel::ghost::TraceLine(L"snapshot hidden: get text failed");
    return false;
  }

  RECT rect{};
  // [GHOST-FIX-012] 光标框获取重写。原来的写法有三个坑，正是 Chrome / Electron
  // （DSH Desktop）联想失效的根因：
  //   1) `GetTextExt` 的返回值被**丢弃**了 —— 失败与"成功但矩形全零"分不开，
  //      出错时连一条日志都没有；
  //   2) 有效性判据写成 `left == 0 && top == 0`：既会误杀屏幕原点附近的合法位置，
  //      又分不清"退化矩形"，而且 Chromium 给不出时返回的正是全零矩形；
  //   3) 唯一的兜底 `GetCaretPos` 需要 **Win32 原生光标**（CreateCaret/ShowCaret）。
  //      Chromium 自己画光标、从不创建原生光标，所以这条在 Chrome / Electron 里
  //      必然失败；记事本是经典 Win32 控件，有原生光标，于是只有它能用。
  HRESULT view_hr = E_FAIL;
  HRESULT ext_hr = E_FAIL;
  BOOL clipped = FALSE;
  BOOL rect_ok = FALSE;
  com_ptr<ITfContextView> context_view;
  view_hr = pContext->GetActiveView(&context_view);
  if (SUCCEEDED(view_hr) && context_view != nullptr) {
    ext_hr = context_view->GetTextExt(ecReadOnly, selection.range, &rect, &clipped);
    rect_ok = GhostRectUsable(rect);
    if (!rect_ok) {
      // 空区间取不到框时，退一步用单字符区间：不少文本存储（Chromium 尤其明显）
      // 对空区间返回全零或 TS_E_NOLAYOUT，而对单字符区间能给真实位置。
      LONG from = caret_pos > 0 ? caret_pos - 1 : caret_pos;
      if (acp_range->SetExtent(from, 1) == S_OK) {
        RECT r2{};
        BOOL c2 = FALSE;
        HRESULT hr2 = context_view->GetTextExt(ecReadOnly, acp_range, &r2, &c2);
        if (GhostRectUsable(r2)) {
          rect = r2;
          clipped = c2;
          ext_hr = hr2;
          rect_ok = TRUE;
        }
      }
      acp_range->SetExtent(caret_pos, 0);  // 还原成空区间
    }
  }
  if (!rect_ok) {
    // 原生光标兜底：GetGUIThreadInfo 比 GetCaretPos 更通用（后者只对调用线程自己的
    // 光标有效）。对 Chromium 无效，但对经典控件是有效的补充。
    GUITHREADINFO gti{};
    gti.cbSize = sizeof(gti);
    if (GetGUIThreadInfo(0, &gti) && gti.hwndCaret) {
      POINT pt{gti.rcCaret.left, gti.rcCaret.bottom};
      if (ClientToScreen(gti.hwndCaret, &pt)) {
        rect = {pt.x, pt.y, pt.x + 2, pt.y + 20};
        rect_ok = GhostRectUsable(rect);
      }
    }
  }
  // 这一段常开日志（LlmLog 带 pid/tid），Chromium 类应用出问题时能直接定位，
  // 不必再去开 WEASEL_GHOST_TRACE。
  LlmLog(L"ghost caret rect view_hr=" + GhostHrText(view_hr) + L" ext_hr=" +
         GhostHrText(ext_hr) + L" clipped=" + std::to_wstring(clipped ? 1 : 0) +
         L" rect=" + std::to_wstring(rect.left) + L"," + std::to_wstring(rect.top) +
         L"," + std::to_wstring(rect.right) + L"," + std::to_wstring(rect.bottom) +
         L" ok=" + std::to_wstring(rect_ok ? 1 : 0));
  if (!rect_ok) {
    weasel::ghost::TraceLine(L"snapshot hidden: caret rect unavailable");
    return false;
  }

  if (prefix != nullptr)
    prefix->assign(buffer, prefix_length);
  if (caret != nullptr)
    *caret = caret_pos;
  if (caret_rect != nullptr)
    *caret_rect = rect;
  return true;
}

// [GHOST-TSF-011 SNAPSHOT-COLLECTOR] in-process engine variant
void WeaselTSF::_UpdateGhostSnapshot(ITfContext* pContext,
                                     TfEditCookie ecReadOnly) {
  LlmLog(L"snapshot enter pending=" + std::to_wstring(_expSnapshotPending) +
         L" composing=" + std::to_wstring(_IsComposing() ? 1 : 0) +
         L"/" + std::to_wstring(_status.composing ? 1 : 0));
  if (!_expSnapshotPending)
    return;
  if (_IsComposing() || _status.composing)
    return;
  if (!m_ghostEngine)
    return;

  std::wstring prefix;
  LONG caret = -1;
  RECT caret_rect{};
  if (!_ReadGhostPrefix(pContext, ecReadOnly, &prefix, &caret, &caret_rect)) {
    _HideGhostPrediction();
    // [GHOST-FIX-012] 在编辑回调里拿不到光标框（Chromium 常见）→ 另起一个
    // **只读编辑会话**重试。不能就地重试：同一个 edit cookie 下布局还没更新。
    _RequestGhostSnapshot(pContext);
    return;
  }

  weasel::ghost::Snapshot snapshot;
  snapshot.document_token = reinterpret_cast<uint64_t>(pContext);
  snapshot.caret = caret;
  snapshot.prefix = prefix;
  snapshot.context_hash = weasel::ghost::HashContext(
      snapshot.document_token, snapshot.caret, snapshot.prefix);
  snapshot.caret_rect = caret_rect;
  weasel::ghost::TraceLine(
      L"snapshot accepted; prefix_chars=" +
      std::to_wstring(snapshot.prefix.size()) + L"; prefix=" + snapshot.prefix);
  LlmLog(L"snapshot accepted prefix=" + snapshot.prefix);
  m_ghostEngine->OnSnapshot(snapshot);
  _expSnapshotPending = FALSE;
}

// [GHOST-FIX-012] 在**独立申请的只读编辑会话**里采集快照。
// 与候选窗定位（Composition.cpp 的 CGetTextExtentEditSession，
// 用 TF_ES_ASYNCDONTCARE | TF_ES_READ 申请）是同一条已验证在 Chromium 里可用的路子。
class CCollectGhostSnapshotEditSession : public CEditSession {
 public:
  CCollectGhostSnapshotEditSession(com_ptr<WeaselTSF> pTextService,
                                   com_ptr<ITfContext> pContext)
      : CEditSession(pTextService, pContext) {}

  STDMETHODIMP DoEditSession(TfEditCookie ec) {
    // com_ptr 就是 ATL::CComPtr，没有 .get()，靠隐式转换拿裸指针
    _pTextService->_UpdateGhostSnapshot(_pContext, ec);
    return S_OK;
  }
};

void WeaselTSF::_RequestGhostSnapshot(ITfContext* pContext) {
  if (!m_ghostEngine || pContext == nullptr)
    return;
  if (_ghost_retry_budget <= 0) {
    LlmLog(L"ghost retry skipped: budget exhausted");
    return;
  }
  com_ptr<ITfContext> context;
  if (FAILED(pContext->QueryInterface(IID_ITfContext, (LPVOID*)&context)) ||
      context == nullptr) {
    LlmLog(L"ghost retry skipped: QI ITfContext failed");
    return;
  }
  _ghost_retry_budget--;
  com_ptr<CCollectGhostSnapshotEditSession> session;
  session.Attach(new CCollectGhostSnapshotEditSession(this, context));
  if (session == nullptr)
    return;
  HRESULT hr = E_FAIL;
  context->RequestEditSession(_tfClientId, session,
                              TF_ES_ASYNCDONTCARE | TF_ES_READ, &hr);
  LlmLog(L"ghost retry edit session hr=" + GhostHrText(hr) +
         L" budget_left=" + std::to_wstring(_ghost_retry_budget));
}

// [GHOST-020 STALE-DROP] A snapshot is only collected after an IME commit, so
// edits made by plain typing, backspace or spaces never reached the engine and
// it kept advertising a prediction for text that had already moved on - Tab
// then inserted that stale text. Compare the caret prefix against what the
// prediction was made for and drop it as soon as they differ. Composing is
// skipped on purpose: while the user is typing pinyin the document itself is
// unchanged and the constrained preview must survive.
void WeaselTSF::_SyncGhostDocument(ITfContext* pContext,
                                   TfEditCookie ecReadOnly) {
  if (!m_ghostEngine)
    return;
  if (_IsComposing() || _status.composing)
    return;

  std::wstring prefix;
  LONG caret = -1;
  RECT caret_rect{};
  if (!_ReadGhostPrefix(pContext, ecReadOnly, &prefix, &caret, &caret_rect)) {
    _HideGhostPrediction();
    return;
  }
  m_ghostEngine->OnDocumentPrefix(prefix);
}

void WeaselTSF::_SetGhostPreedit(const std::wstring& preedit) {
  if (m_ghostEngine)
    m_ghostEngine->SetPreedit(preedit);
}

std::wstring WeaselTSF::_GetGhostPrediction() {
  if (!m_ghostEngine)
    return L"";
  return m_ghostEngine->VisiblePrediction();
}

bool WeaselTSF::_HasGhostPrediction() {
  return m_ghostEngine && m_ghostEngine->HasVisiblePrediction();
}

void WeaselTSF::_HideGhostPrediction() {
  if (m_ghostEngine)
    m_ghostEngine->Hide();
}

