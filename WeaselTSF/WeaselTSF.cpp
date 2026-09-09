#include "stdafx.h"

#include <WeaselIPCData.h>
#include <thread>
#include <shellapi.h>
#include <tlhelp32.h>
#include "WeaselTSF.h"
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

  TF_SELECTION selection{};
  ULONG selection_count = 0;
  if (FAILED(pContext->GetSelection(ecReadOnly, TF_DEFAULT_SELECTION, 1,
                                    &selection, &selection_count)) ||
      selection_count < 1 || selection.range == nullptr) {
    weasel::ghost::TraceLine(L"snapshot hidden: no selection");
    _HideGhostPrediction();
    return;
  }

  BOOL empty = TRUE;
  if (FAILED(selection.range->IsEmpty(ecReadOnly, &empty)) || !empty) {
    weasel::ghost::TraceLine(L"snapshot hidden: non-empty selection");
    _HideGhostPrediction();
    return;
  }

  com_ptr<ITfRangeACP> acp_range;
  LONG caret = -1;
  LONG selection_length = 0;
  if (FAILED(selection.range->QueryInterface(IID_ITfRangeACP,
                                             (LPVOID*)&acp_range)) ||
      acp_range == nullptr ||
      FAILED(acp_range->GetExtent(&caret, &selection_length)) ||
      caret < 0 || selection_length != 0) {
    weasel::ghost::TraceLine(L"snapshot hidden: ACP range unavailable");
    _HideGhostPrediction();
    return;
  }

  LONG context_start = max(
      0L, static_cast<LONG>(caret - weasel::ghost::kDefaultContextChars));
  LONG context_length = caret - context_start;
  if (FAILED(acp_range->SetExtent(context_start, context_length))) {
    weasel::ghost::TraceLine(L"snapshot hidden: ACP context set failed");
    _HideGhostPrediction();
    return;
  }

  wchar_t prefix[weasel::ghost::kDefaultContextChars]{};
  ULONG prefix_length = 0;
  if (FAILED(acp_range->GetText(
          ecReadOnly, 0, prefix, weasel::ghost::kDefaultContextChars,
          &prefix_length))) {
    weasel::ghost::TraceLine(L"snapshot hidden: get text failed");
    _HideGhostPrediction();
    return;
  }

  RECT caret_rect{};
  com_ptr<ITfContextView> context_view;
  if (SUCCEEDED(pContext->GetActiveView(&context_view)) &&
      context_view != nullptr) {
    BOOL clipped = FALSE;
    context_view->GetTextExt(ecReadOnly, selection.range,
                             &caret_rect, &clipped);
  }
  if (caret_rect.left == 0 && caret_rect.top == 0) {
    POINT caret_point{};
    HWND foreground = GetForegroundWindow();
    if (foreground && GetCaretPos(&caret_point) &&
        ClientToScreen(foreground, &caret_point)) {
      caret_rect = {caret_point.x, caret_point.y, caret_point.x + 2,
                    caret_point.y + 20};
    }
  }
  if (caret_rect.left == 0 && caret_rect.top == 0) {
    weasel::ghost::TraceLine(L"snapshot hidden: caret rect unavailable");
    _HideGhostPrediction();
    return;
  }

  weasel::ghost::Snapshot snapshot;
  snapshot.document_token = reinterpret_cast<uint64_t>(pContext);
  snapshot.caret = caret;
  snapshot.prefix.assign(prefix, prefix_length);
  snapshot.context_hash = weasel::ghost::HashContext(
      snapshot.document_token, snapshot.caret, snapshot.prefix);
  snapshot.caret_rect = caret_rect;
  weasel::ghost::TraceLine(
      L"snapshot accepted; prefix_chars=" +
      std::to_wstring(prefix_length) + L"; prefix=" + snapshot.prefix);
  LlmLog(L"snapshot accepted prefix=" + snapshot.prefix);
  m_ghostEngine->OnSnapshot(snapshot);
  _expSnapshotPending = FALSE;
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

