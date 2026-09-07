#include "stdafx.h"
#include "WeaselTSF.h"
#include "EditSession.h"
#include "ResponseParser.h"
#include "CandidateList.h"

/* Start Composition */
class CStartCompositionEditSession : public CEditSession {
 public:
  CStartCompositionEditSession(com_ptr<WeaselTSF> pTextService,
                               com_ptr<ITfContext> pContext,
                               BOOL fCUASWorkaroundEnabled)
      : CEditSession(pTextService, pContext) {
    _fCUASWorkaroundEnabled = fCUASWorkaroundEnabled;
  }

  /* ITfEditSession */
  STDMETHODIMP DoEditSession(TfEditCookie ec);

 private:
  BOOL _fCUASWorkaroundEnabled;
};

STDMETHODIMP CStartCompositionEditSession::DoEditSession(TfEditCookie ec) {
  HRESULT hr = E_FAIL;
  com_ptr<ITfInsertAtSelection> pInsertAtSelection;
  com_ptr<ITfRange> pRangeComposition;
  if (_pContext->QueryInterface(IID_ITfInsertAtSelection,
                                (LPVOID*)&pInsertAtSelection) != S_OK)
    return hr;
  if (pInsertAtSelection->InsertTextAtSelection(ec, TF_IAS_QUERYONLY, NULL, 0,
                                                &pRangeComposition) != S_OK)
    return hr;

  com_ptr<ITfContextComposition> pContextComposition;
  com_ptr<ITfComposition> pComposition;
  if (_pContext->QueryInterface(IID_ITfContextComposition,
                                (LPVOID*)&pContextComposition) != S_OK)
    return hr;
  if ((pContextComposition->StartComposition(
           ec, pRangeComposition, _pTextService, &pComposition) == S_OK) &&
      (pComposition != NULL)) {
    _pTextService->_SetComposition(pComposition);

    /* set selection */
    TF_SELECTION tfSelection;
    pRangeComposition->Collapse(ec, TF_ANCHOR_END);
    tfSelection.range = pRangeComposition;
    tfSelection.style.ase = TF_AE_NONE;
    tfSelection.style.fInterimChar = FALSE;
    _pContext->SetSelection(ec, 1, &tfSelection);

    // The old composition's range is still visible while its asynchronous
    // end session is pending. Position only after the new composition has
    // actually been created, not from the response handler's stale range.
    _pTextService->_UpdateCompositionWindow(_pContext);
  }

  return hr;
}

void WeaselTSF::_StartComposition(com_ptr<ITfContext> pContext,
                                  BOOL fCUASWorkaroundEnabled) {
  com_ptr<CStartCompositionEditSession> pStartCompositionEditSession;
  pStartCompositionEditSession.Attach(
      new CStartCompositionEditSession(this, pContext, fCUASWorkaroundEnabled));
  _cand->StartUI();
  if (pStartCompositionEditSession != nullptr) {
    HRESULT hr;
    pContext->RequestEditSession(_tfClientId, pStartCompositionEditSession,
                                 TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
  }
}

/* End Composition */
class CEndCompositionEditSession : public CEditSession {
 public:
  CEndCompositionEditSession(com_ptr<WeaselTSF> pTextService,
                             com_ptr<ITfContext> pContext,
                             com_ptr<ITfComposition> pComposition,
                             BOOL clear = TRUE,
                             BOOL move_caret_to_start = FALSE)
      : CEditSession(pTextService, pContext),
        _clear(clear),
        _moveCaretToStart(move_caret_to_start) {
    _pComposition = pComposition;
  }

  /* ITfEditSession */
  STDMETHODIMP DoEditSession(TfEditCookie ec);

 private:
  com_ptr<ITfComposition> _pComposition;
  BOOL _clear;
  BOOL _moveCaretToStart;
};

STDMETHODIMP CEndCompositionEditSession::DoEditSession(TfEditCookie ec) {
  /* Clear the dummy text we set before, if any. */
  if (_pComposition == nullptr)
    return S_OK;
  // Avoid null pointer dereference
  if (!_pTextService || !_pContext)
    return S_OK;

  _pTextService->_ClearCompositionDisplayAttributes(ec, _pContext);

  com_ptr<ITfRange> pCompositionRange;
  if (_clear && _pComposition->GetRange(&pCompositionRange) == S_OK) {
    if (_moveCaretToStart) {
      pCompositionRange->Collapse(ec, TF_ANCHOR_START);
      TF_SELECTION selection{};
      selection.range = pCompositionRange;
      selection.style.ase = TF_AE_NONE;
      selection.style.fInterimChar = FALSE;
      _pContext->SetSelection(ec, 1, &selection);
    }
    pCompositionRange->SetText(ec, 0, L"", 0);
  }

  // Drop ownership before EndComposition(). Some applications notify
  // OnCompositionTerminated synchronously while the old composition ends.
  // Keeping it as the current composition makes that normal notification
  // look like an external abort and can clear a new Rime composition during
  // auto-commit.
  if (_pTextService && _pTextService->_IsCurrentComposition(_pComposition))
    _pTextService->_FinalizeComposition();
  _pComposition->EndComposition(ec);
  return S_OK;
}

void WeaselTSF::_EndComposition(com_ptr<ITfContext> pContext,
                                BOOL clear,
                                BOOL endUI) {
  CEndCompositionEditSession* pEditSession;
  HRESULT hr;
  com_ptr<ITfComposition> pComposition = _pComposition;

  if (endUI)
    _cand->EndUI();
  if ((pEditSession = new CEndCompositionEditSession(
           this, pContext, pComposition, clear)) != NULL) {
    pContext->RequestEditSession(_tfClientId, pEditSession,
                                 TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
    pEditSession->Release();
  }
}

/* Inline LLM Ghost */
namespace {
constexpr UINT kInlineGhostCheckMessage = WM_APP + 0x4a51;
constexpr UINT kInlineGhostFileChangedMessage = WM_APP + 0x4a52;
constexpr UINT kInlineGhostHideMessage = WM_APP + 0x4a53;
constexpr LONG kInlineGhostContextChars = 100;
constexpr DWORD kInlineGhostDelayMilliseconds = 800;
}  // namespace

class CInlineGhostCheckEditSession : public CEditSession {
 public:
  CInlineGhostCheckEditSession(com_ptr<WeaselTSF> pTextService,
                               com_ptr<ITfContext> pContext)
      : CEditSession(pTextService, pContext) {}

  STDMETHODIMP DoEditSession(TfEditCookie ec) override {
    weasel::inline_ghost::TraceLine(L"check session begin");
    if (_pTextService == nullptr || _pContext == nullptr)
      return S_OK;
    const bool ghost_active = _pTextService->_IsGhostInlineActive();
    if (_pTextService->_HasRimeComposition()) {
      weasel::inline_ghost::TraceLine(L"skip: rime composing");
      return S_OK;
    }
    if (!ghost_active &&
        (_pTextService->_IsComposing() ||
         _pTextService->_IsGhostBusy() ||
         _pTextService->_IsGhostSuppressedAfterCancel())) {
      weasel::inline_ghost::TraceLine(
          L"skip: busy or suppressed after cancel");
      return S_OK;
    }

    const std::wstring path = weasel::inline_ghost::GhostFilePath();
    uint64_t expected_hash = 0;
    std::wstring suggestion;
    if (!weasel::inline_ghost::ReadSuggestion(path, &expected_hash,
                                              &suggestion)) {
      weasel::inline_ghost::TraceLine(L"skip: no suggestion file");
      return S_OK;
    }

    TF_SELECTION selection{};
    ULONG selection_count = 0;
    if (FAILED(_pContext->GetSelection(ec, TF_DEFAULT_SELECTION, 1,
                                       &selection, &selection_count)) ||
        selection_count < 1 || selection.range == nullptr)
      return S_OK;

    BOOL is_empty = FALSE;
    if (FAILED(selection.range->IsEmpty(ec, &is_empty)) || !is_empty)
      return S_OK;

    com_ptr<ITfRange> prefix_range;
    if (FAILED(selection.range->Clone(&prefix_range)) ||
        prefix_range == nullptr)
      return S_OK;
    if (FAILED(prefix_range->Collapse(ec, TF_ANCHOR_END)))
      return S_OK;

    LONG shifted = 0;
    if (FAILED(prefix_range->ShiftStart(
            ec, -kInlineGhostContextChars, &shifted, nullptr)))
      return S_OK;

    wchar_t buffer[kInlineGhostContextChars + 1]{};
    ULONG fetched = 0;
    if (FAILED(prefix_range->GetText(ec, 0, buffer,
                                     kInlineGhostContextChars, &fetched)))
      return S_OK;

    std::wstring context(buffer, fetched);
    const size_t sentinel_count =
        weasel::inline_ghost::GhostSentinelRemoveCount(context);
    if (ghost_active && sentinel_count > 0) {
      // An active ghost already owns this text; do not delete it here.
      context = weasel::inline_ghost::CleanGhostSentinelBlocks(context);
      const uint64_t ghost_hash =
          weasel::inline_ghost::HashContext(context);
      if (ghost_hash != expected_hash)
        return S_OK;
      if (suggestion == _pTextService->_GetGhostCommitText())
        return S_OK;
      weasel::inline_ghost::TraceLine(L"replace active ghost suggestion");
      _pTextService->_SetInlineGhostCombinedHash(
          weasel::inline_ghost::HashContext(context + suggestion));
      _pTextService->_ReplaceGhostInline(ec, _pContext, suggestion);
      weasel::inline_ghost::ClearGhostFile();
      return S_OK;
    }

    size_t remove_count = sentinel_count;
    if (remove_count == 0)
      remove_count = weasel::inline_ghost::TrailingGhostMarkerCount(context);
    if (remove_count > 0) {
      com_ptr<ITfRange> cleanup_range;
      if (SUCCEEDED(selection.range->Clone(&cleanup_range)) &&
          cleanup_range != nullptr) {
        cleanup_range->Collapse(ec, TF_ANCHOR_END);
        LONG removed = 0;
        if (SUCCEEDED(cleanup_range->ShiftStart(
                ec, -static_cast<LONG>(remove_count), &removed, nullptr))) {
          cleanup_range->SetText(ec, 0, L"", 0);
          cleanup_range->Collapse(ec, TF_ANCHOR_END);
          TF_SELECTION caret{};
          caret.range = cleanup_range;
          caret.style.ase = TF_AE_NONE;
          caret.style.fInterimChar = FALSE;
          _pContext->SetSelection(ec, 1, &caret);
          weasel::inline_ghost::TraceLine(L"cleaned stale ghost markers");
        }
      }
      if (sentinel_count > 0)
        context = weasel::inline_ghost::CleanGhostSentinelBlocks(context);
      else
        context = weasel::inline_ghost::CleanGhostMarkers(context);
    }

    const uint64_t actual_hash = weasel::inline_ghost::HashContext(context);
    if (actual_hash != expected_hash) {
      wchar_t trace[512]{};
      swprintf_s(trace, L"skip: hash mismatch expected=%016llx actual=%016llx",
                 expected_hash, actual_hash);
      weasel::inline_ghost::TraceLine(trace);
      return S_OK;
    }

    weasel::inline_ghost::TraceLine(L"show ghost inline");
    _pTextService->_SetInlineGhostBaseContextHash(actual_hash);
    const uint64_t combined_hash =
        weasel::inline_ghost::HashContext(context + suggestion);
    _pTextService->_SetInlineGhostCombinedHash(combined_hash);
    _pTextService->_ShowGhostInline(ec, _pContext, suggestion);
    weasel::inline_ghost::ClearGhostFile();
    return S_OK;
  }
};

void WeaselTSF::_ShowGhostInline(TfEditCookie ec, ITfContext* pContext,
                                 const std::wstring& text) {
  weasel::inline_ghost::TraceLine(L"ShowGhostInline begin");
  if (pContext == nullptr || text.empty())
    return;
  if (_IsComposing() || _HasRimeComposition() || _ghostBusy) {
    weasel::inline_ghost::TraceLine(L"ShowGhostInline skip busy/composing");
    return;
  }
  _ghostBusy = TRUE;

  com_ptr<ITfInsertAtSelection> insert_at_selection;
  com_ptr<ITfRange> composition_range;
  if (FAILED(pContext->QueryInterface(IID_ITfInsertAtSelection,
                                      (LPVOID*)&insert_at_selection)) ||
      FAILED(insert_at_selection->InsertTextAtSelection(
          ec, TF_IAS_QUERYONLY, NULL, 0, &composition_range)) ||
      composition_range == nullptr) {
    weasel::inline_ghost::TraceLine(L"ShowGhostInline no insert range");
    _ghostBusy = FALSE;
    return;
  }

  com_ptr<ITfContextComposition> context_composition;
  com_ptr<ITfComposition> composition;
  if (FAILED(pContext->QueryInterface(IID_ITfContextComposition,
                                      (LPVOID*)&context_composition)) ||
      FAILED(context_composition->StartComposition(
          ec, composition_range, this, &composition)) ||
      composition == nullptr) {
    weasel::inline_ghost::TraceLine(L"ShowGhostInline start composition failed");
    _ghostBusy = FALSE;
    return;
  }

  _SetComposition(composition);
  _pEditSessionContext = pContext;
  _inlineGhostCommitText = text;

  const std::wstring display_text =
      std::wstring(1, weasel::inline_ghost::kGhostOpenSentinel) + L"<" +
      text + L">" +
      std::wstring(1, weasel::inline_ghost::kGhostCloseSentinel);
  if (FAILED(composition_range->SetText(
          ec, 0, display_text.c_str(),
          static_cast<LONG>(display_text.size())))) {
    _ghostBusy = FALSE;
    return;
  }
  composition_range->Collapse(ec, TF_ANCHOR_END);

  TF_SELECTION selection{};
  selection.range = composition_range;
  selection.style.ase = TF_AE_NONE;
  selection.style.fInterimChar = FALSE;
  pContext->SetSelection(ec, 1, &selection);

  _SetCompositionDisplayAttributes(ec, pContext, composition_range);
  _inlineGhostActive = TRUE;
  _cand->EndUI();
  _ScheduleGhostHide();
  weasel::inline_ghost::TraceLine(L"ShowGhostInline ok");
}

void WeaselTSF::_ReplaceGhostInline(TfEditCookie ec, ITfContext* pContext,
                                      const std::wstring& text) {
  if (pContext == nullptr || text.empty() || !_inlineGhostActive ||
      !_IsComposing())
    return;
  com_ptr<ITfRange> range;
  if (FAILED(_pComposition->GetRange(&range)) || range == nullptr)
    return;

  _inlineGhostCommitText = text;
  _ClearCompositionDisplayAttributes(ec, pContext);
  const std::wstring display_text =
      std::wstring(1, weasel::inline_ghost::kGhostOpenSentinel) + L"<" +
      text + L">" +
      std::wstring(1, weasel::inline_ghost::kGhostCloseSentinel);
  if (FAILED(range->SetText(ec, 0, display_text.c_str(),
                            static_cast<LONG>(display_text.size()))))
    return;
  range->Collapse(ec, TF_ANCHOR_END);

  TF_SELECTION selection{};
  selection.range = range;
  selection.style.ase = TF_AE_NONE;
  selection.style.fInterimChar = FALSE;
  pContext->SetSelection(ec, 1, &selection);
  _SetCompositionDisplayAttributes(ec, pContext, range);
  _ScheduleGhostHide();
  weasel::inline_ghost::TraceLine(L"ReplaceGhostInline ok");
}

class CInlineGhostCommitEditSession : public CEditSession {
 public:
  CInlineGhostCommitEditSession(com_ptr<WeaselTSF> pTextService,
                                com_ptr<ITfContext> pContext,
                                com_ptr<ITfComposition> pComposition,
                                const std::wstring& text)
      : CEditSession(pTextService, pContext),
        _pComposition(pComposition),
        _text(text) {}

  STDMETHODIMP DoEditSession(TfEditCookie ec) override {
    if (_pComposition == nullptr || _pContext == nullptr) {
      _pTextService->_SetGhostBusy(FALSE);
      return S_OK;
    }
    com_ptr<ITfRange> range;
    if (FAILED(_pComposition->GetRange(&range)) || range == nullptr) {
      _pTextService->_SetGhostBusy(FALSE);
      return S_OK;
    }

    _pTextService->_ClearCompositionDisplayAttributes(ec, _pContext);
    if (!_text.empty() &&
        FAILED(range->SetText(ec, 0, _text.c_str(),
                              static_cast<LONG>(_text.size())))) {
      _pTextService->_SetGhostBusy(FALSE);
      return S_OK;
    }

    range->Collapse(ec, TF_ANCHOR_END);
    TF_SELECTION selection{};
    selection.range = range;
    selection.style.ase = TF_AE_NONE;
    selection.style.fInterimChar = FALSE;
    _pContext->SetSelection(ec, 1, &selection);

    if (_pTextService->_IsCurrentComposition(_pComposition))
      _pTextService->_FinalizeComposition();
    _pComposition->EndComposition(ec);
    _pTextService->_SetGhostBusy(FALSE);
    return S_OK;
  }

 private:
  com_ptr<ITfComposition> _pComposition;
  std::wstring _text;
};

void WeaselTSF::_CommitGhostComposition(ITfContext* pContext) {
  if (!_inlineGhostActive || !_IsComposing())
    return;
  weasel::inline_ghost::TraceLine(L"Commit ghost composition");
  _CancelGhostHideTimer();
  _inlineGhostActive = FALSE;
  _ghostKeyAcceptPending = FALSE;
  _ghostSuppressAfterCancel = FALSE;
  _ghostBusy = TRUE;
  if (pContext != nullptr)
    _pEditSessionContext = pContext;
  if (_pEditSessionContext == nullptr)
    return;

  com_ptr<ITfComposition> composition = _pComposition;
  std::wstring commit_text = _inlineGhostCommitText;
  _inlineGhostCommitText.clear();
  CInlineGhostCommitEditSession* session =
      new CInlineGhostCommitEditSession(this, _pEditSessionContext,
                                        composition, commit_text);
  HRESULT hr = E_FAIL;
  _pEditSessionContext->RequestEditSession(
      _tfClientId, session, TF_ES_SYNC | TF_ES_READWRITE, &hr);
  session->Release();
  if (!SUCCEEDED(hr))
    _ghostBusy = FALSE;
}

void WeaselTSF::_CancelGhostComposition(ITfContext* pContext) {
  if (!_inlineGhostActive || !_IsComposing())
    return;
  weasel::inline_ghost::TraceLine(L"Cancel ghost composition");
  _inlineGhostActive = FALSE;
  _ghostKeyAcceptPending = FALSE;
  _inlineGhostCombinedHash = 0;
  _inlineGhostCommitText.clear();
  _ghostSuppressAfterCancel = TRUE;
  _ghostBusy = TRUE;
  _CancelGhostHideTimer();
  if (pContext == nullptr)
    pContext = _pEditSessionContext;
  if (pContext == nullptr) {
    _ghostBusy = FALSE;
    return;
  }

  com_ptr<ITfComposition> composition = _pComposition;
  com_ptr<ITfContext> cancel_context = pContext;
  CEndCompositionEditSession* session =
      new CEndCompositionEditSession(this, cancel_context, composition, TRUE,
                                     TRUE);
  HRESULT hr = E_FAIL;
  pContext->RequestEditSession(_tfClientId, session,
                               TF_ES_SYNC | TF_ES_READWRITE, &hr);
  session->Release();
  if (!SUCCEEDED(hr))
    _ghostBusy = FALSE;
}

BOOL WeaselTSF::_HandleGhostKey(ITfContext* pContext, WPARAM wParam,
                                LPARAM lParam, BOOL* pfEaten,
                                bool test_only) {
  if (!_inlineGhostActive || !_IsComposing())
    return false;
  const bool key_up = (lParam & (1 << 31)) != 0;
  if (key_up)
    return false;
  if (wParam == VK_TAB) {
    *pfEaten = TRUE;
    if (test_only) {
      _ghostKeyAcceptPending = TRUE;
    } else {
      _CommitGhostComposition(pContext);
    }
    return true;
  }
  if (wParam == VK_ESCAPE) {
    _CancelGhostComposition(pContext);
    *pfEaten = TRUE;
    return true;
  }
  _CancelGhostComposition(pContext);
  return false;
}

static VOID CALLBACK InlineGhostTimerProc(PVOID param, BOOLEAN) {
  HWND hwnd = static_cast<HWND>(param);
  if (hwnd != nullptr && ::IsWindow(hwnd))
    ::PostMessageW(hwnd, kInlineGhostCheckMessage, 0, 0);
}

BOOL WeaselTSF::_EnsureGhostMessageWindow() {
  if (_ghostMessageWindow != nullptr && ::IsWindow(_ghostMessageWindow))
    return TRUE;

  WNDCLASSEXW window_class{};
  window_class.cbSize = sizeof(window_class);
  window_class.lpfnWndProc = WeaselTSF::_GhostMessageWindowProc;
  window_class.hInstance = ::GetModuleHandleW(nullptr);
  window_class.lpszClassName = L"WeaselInlineGhostTimerWindow";
  ::RegisterClassExW(&window_class);

  _ghostMessageWindow = ::CreateWindowExW(
      0, window_class.lpszClassName, L"", WS_POPUP, 0, 0, 0, 0,
      HWND_MESSAGE, nullptr, window_class.hInstance, nullptr);
  if (_ghostMessageWindow == nullptr)
    return FALSE;
  ::SetWindowLongPtrW(_ghostMessageWindow, GWLP_USERDATA,
                      reinterpret_cast<LONG_PTR>(this));
  _StartGhostFileWatcher();
  return TRUE;
}

struct GhostWatcherParams {
  HWND hwnd = nullptr;
  HANDLE stop = nullptr;
};

DWORD WINAPI InlineGhostFileWatcherProc(LPVOID param) {
  auto* params = static_cast<GhostWatcherParams*>(param);
  if (params == nullptr)
    return 0;

  wchar_t appdata[MAX_PATH]{};
  const DWORD length = ::GetEnvironmentVariableW(L"APPDATA", appdata, MAX_PATH);
  if (length == 0 || length >= MAX_PATH) {
    delete params;
    return 0;
  }
  const std::wstring dir = std::wstring(appdata) + L"\\Rime";
  ::CreateDirectoryW(dir.c_str(), nullptr);

  HANDLE notify = ::FindFirstChangeNotificationW(
      dir.c_str(), FALSE,
      FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_FILE_NAME |
          FILE_NOTIFY_CHANGE_SIZE);
  if (notify == INVALID_HANDLE_VALUE) {
    delete params;
    return 0;
  }

  const HANDLE events[] = {notify, params->stop};
  while (true) {
    const DWORD wait =
        ::WaitForMultipleObjects(2, events, FALSE, INFINITE);
    if (wait == WAIT_OBJECT_0) {
      if (params->hwnd != nullptr && ::IsWindow(params->hwnd))
        ::PostMessageW(params->hwnd, kInlineGhostFileChangedMessage, 0, 0);
      ::FindNextChangeNotification(notify);
    } else {
      break;
    }
  }

  ::FindCloseChangeNotification(notify);
  delete params;
  return 0;
}

static VOID CALLBACK InlineGhostHideTimerProc(PVOID param, BOOLEAN) {
  HWND hwnd = static_cast<HWND>(param);
  if (hwnd != nullptr && ::IsWindow(hwnd))
    ::PostMessageW(hwnd, kInlineGhostHideMessage, 0, 0);
}

void WeaselTSF::_StartGhostFileWatcher() {
  if (_ghostWatcherThread != nullptr)
    return;
  if (_ghostWatcherStop == nullptr)
    _ghostWatcherStop = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  auto* params = new GhostWatcherParams();
  params->hwnd = _ghostMessageWindow;
  params->stop = _ghostWatcherStop;
  _ghostWatcherThread = ::CreateThread(
      nullptr, 0, InlineGhostFileWatcherProc, params, 0, nullptr);
}

void WeaselTSF::_StopGhostFileWatcher() {
  if (_ghostWatcherStop != nullptr) {
    ::SetEvent(_ghostWatcherStop);
    if (_ghostWatcherThread != nullptr) {
      ::WaitForSingleObject(_ghostWatcherThread, 2000);
      ::CloseHandle(_ghostWatcherThread);
      _ghostWatcherThread = nullptr;
    }
    ::CloseHandle(_ghostWatcherStop);
    _ghostWatcherStop = nullptr;
  }
}

void WeaselTSF::_ScheduleGhostHide() {
  if (_ghostTimerQueue == nullptr)
    _ghostTimerQueue = ::CreateTimerQueue();
  if (_ghostTimerQueue == nullptr)
    return;
  if (_ghostHideTimer != nullptr) {
    ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostHideTimer, nullptr);
    _ghostHideTimer = nullptr;
  }
  ::CreateTimerQueueTimer(&_ghostHideTimer, _ghostTimerQueue,
                          InlineGhostHideTimerProc, _ghostMessageWindow, 8000,
                          0, WT_EXECUTEDEFAULT);
}

void WeaselTSF::_CancelGhostHideTimer() {
  if (_ghostTimerQueue != nullptr && _ghostHideTimer != nullptr) {
    ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostHideTimer, nullptr);
    _ghostHideTimer = nullptr;
  }
}

LRESULT CALLBACK WeaselTSF::_GhostMessageWindowProc(
    HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
  if (message == kInlineGhostCheckMessage ||
      message == kInlineGhostFileChangedMessage) {
    auto* service = reinterpret_cast<WeaselTSF*>(
        ::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (service != nullptr)
      service->_RunGhostScheduledCheck();
    return 0;
  }
  if (message == kInlineGhostHideMessage) {
    auto* service = reinterpret_cast<WeaselTSF*>(
        ::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (service != nullptr)
      service->_CancelGhostComposition(service->_pEditSessionContext);
    return 0;
  }
  return ::DefWindowProcW(hwnd, message, wparam, lparam);
}

void WeaselTSF::_ScheduleInlineGhostCheck() {
  const bool ghost_active = _IsGhostInlineActive();
  if (_HasRimeComposition() ||
      (_ghostBusy && !ghost_active) ||
      (_IsComposing() && !ghost_active))
    return;
  _ghostCheckRetries = 60;
  if (!_EnsureGhostMessageWindow()) {
    weasel::inline_ghost::TraceLine(L"Schedule no message window");
    return;
  }
  if (_ghostTimerQueue == nullptr)
    _ghostTimerQueue = ::CreateTimerQueue();
  if (_ghostTimerQueue == nullptr)
    return;
  if (_ghostTimer != nullptr) {
    ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostTimer, nullptr);
    _ghostTimer = nullptr;
  }
  ::CreateTimerQueueTimer(
      &_ghostTimer, _ghostTimerQueue, InlineGhostTimerProc,
      _ghostMessageWindow, kInlineGhostDelayMilliseconds, 0,
      WT_EXECUTEDEFAULT);
}

void WeaselTSF::_CancelInlineGhostScheduler() {
  _StopGhostFileWatcher();
  if (_ghostTimerQueue != nullptr) {
    if (_ghostTimer != nullptr) {
      ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostTimer, nullptr);
      _ghostTimer = nullptr;
    }
    if (_ghostHideTimer != nullptr) {
      ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostHideTimer, nullptr);
      _ghostHideTimer = nullptr;
    }
    ::DeleteTimerQueueEx(_ghostTimerQueue, INVALID_HANDLE_VALUE);
    _ghostTimerQueue = nullptr;
  }
  if (_ghostMessageWindow != nullptr) {
    ::DestroyWindow(_ghostMessageWindow);
    _ghostMessageWindow = nullptr;
  }
}

void WeaselTSF::_RunGhostScheduledCheck() {
  weasel::inline_ghost::TraceLine(L"Scheduled check fired");
  if (_ghostTimer != nullptr) {
    ::DeleteTimerQueueTimer(_ghostTimerQueue, _ghostTimer, nullptr);
    _ghostTimer = nullptr;
  }
  if (_HasRimeComposition())
    return;
  if (_IsComposing() && !_IsGhostInlineActive())
    return;
  if (_pTextEditSinkContext == nullptr) {
    weasel::inline_ghost::TraceLine(L"Scheduled check no sink context");
    return;
  }

  CInlineGhostCheckEditSession* session =
      new CInlineGhostCheckEditSession(this, _pTextEditSinkContext);
  HRESULT hr = E_FAIL;
  _pTextEditSinkContext->RequestEditSession(
      _tfClientId, session, TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
  session->Release();

  if (_ghostCheckRetries > 0 && !_IsGhostInlineActive() &&
      !_HasRimeComposition()) {
    --_ghostCheckRetries;
    ::CreateTimerQueueTimer(
        &_ghostTimer, _ghostTimerQueue, InlineGhostTimerProc,
        _ghostMessageWindow, kInlineGhostDelayMilliseconds, 0,
        WT_EXECUTEDEFAULT);
  }
}

/* Get Text Extent */
class CGetTextExtentEditSession : public CEditSession {
 public:
  CGetTextExtentEditSession(com_ptr<WeaselTSF> pTextService,
                            com_ptr<ITfContext> pContext,
                            com_ptr<ITfContextView> pContextView,
                            com_ptr<ITfComposition> pComposition,
                            bool enhancedPosition)
      : CEditSession(pTextService, pContext) {
    _pContextView = pContextView;
    _pComposition = pComposition;
    _enhancedPosition = enhancedPosition;
  }

  /* ITfEditSession */
  STDMETHODIMP DoEditSession(TfEditCookie ec);

 private:
  com_ptr<ITfContextView> _pContextView;
  com_ptr<ITfComposition> _pComposition;
  bool _enhancedPosition;
};

STDMETHODIMP CGetTextExtentEditSession::DoEditSession(TfEditCookie ec) {
  com_ptr<ITfInsertAtSelection> pInsertAtSelection;
  com_ptr<ITfRange> pRangeComposition;
  ITfRange* pRange;
  RECT rc;
  BOOL fClipped;
  TF_SELECTION selection;
  ULONG nSelection;

  if (FAILED(_pContext->QueryInterface(IID_ITfInsertAtSelection,
                                       (LPVOID*)&pInsertAtSelection)))
    return E_FAIL;
  if (FAILED(_pContext->GetSelection(ec, TF_DEFAULT_SELECTION, 1, &selection,
                                     &nSelection)))
    return E_FAIL;

  if (_pComposition != nullptr && _pComposition->GetRange(&pRange) == S_OK) {
    pRange->Collapse(ec, TF_ANCHOR_START);
  } else {
    // composition end
    // note: selection.range is always an empty range
    pRange = selection.range;
  }

  if ((_pContextView->GetTextExt(ec, pRange, &rc, &fClipped)) == S_OK &&
      (rc.left != 0 || rc.top != 0)) {
    // get the foreground window pos and check if rc from GetTextExt is out of
    // window
    if (_enhancedPosition) {
      HWND hwnd;
      RECT rcForegroundWindow;
      hwnd = GetForegroundWindow();
      ::GetWindowRect(hwnd, &rcForegroundWindow);

      if (rc.left < rcForegroundWindow.left ||
          rc.left > rcForegroundWindow.right ||
          rc.top < rcForegroundWindow.top ||
          rc.top > rcForegroundWindow.bottom) {
        POINT pt;
        bool hasCaret = ::GetCaretPos(&pt);
        int offsetx = rcForegroundWindow.left - rc.left + (hasCaret ? pt.x : 0);
        int offsety = rcForegroundWindow.top - rc.top + (hasCaret ? pt.y : 0);
        rc.left += offsetx;
        rc.right += offsetx;
        rc.top += offsety;
        rc.bottom += offsety;
      }
    }
    _pTextService->_SetCompositionPosition(rc);
  }
  return S_OK;
}

/* Composition Window Handling */
BOOL WeaselTSF::_UpdateCompositionWindow(com_ptr<ITfContext> pContext) {
  com_ptr<ITfContextView> pContextView;
  if (pContext->GetActiveView(&pContextView) != S_OK)
    return FALSE;
  com_ptr<CGetTextExtentEditSession> pEditSession;
  pEditSession.Attach(
      new CGetTextExtentEditSession(this, pContext, pContextView, _pComposition,
                                    _cand->style().enhanced_position));
  if (pEditSession == NULL) {
    return FALSE;
  }
  HRESULT hr;
  pContext->RequestEditSession(_tfClientId, pEditSession,
                               TF_ES_ASYNCDONTCARE | TF_ES_READ, &hr);
  return SUCCEEDED(hr);
}

void WeaselTSF::_SetCompositionPosition(const RECT& rc) {
  /* Test if rect is valid.
   * If it is invalid during CUAS test, we need to apply CUAS workaround */
  if (!_fCUASWorkaroundTested) {
    _fCUASWorkaroundTested = TRUE;
    if (rc.top == rc.bottom) {
      _fCUASWorkaroundEnabled = TRUE;
      return;
    }
  }
  RECT _rc;
  _rc.left = _rc.right = rc.left;
  _rc.top = _rc.bottom = rc.bottom;
  m_client.UpdateInputPosition(rc);
  _cand->UpdateInputPosition(rc);
}

/* Inline Preedit */
class CInlinePreeditEditSession : public CEditSession {
 public:
  CInlinePreeditEditSession(com_ptr<WeaselTSF> pTextService,
                            com_ptr<ITfContext> pContext,
                            com_ptr<ITfComposition> pComposition,
                            const std::shared_ptr<weasel::Context> context)
      : CEditSession(pTextService, pContext),
        _pComposition(pComposition),
        _context(context) {}

  /* ITfEditSession */
  STDMETHODIMP DoEditSession(TfEditCookie ec);

 private:
  com_ptr<ITfComposition> _pComposition;
  const std::shared_ptr<weasel::Context> _context;
};

STDMETHODIMP CInlinePreeditEditSession::DoEditSession(TfEditCookie ec) {
  std::wstring preedit = _context->preedit.str;

  com_ptr<ITfRange> pRangeComposition;
  if (_pComposition == nullptr)
    return E_FAIL;
  if ((_pComposition->GetRange(&pRangeComposition)) != S_OK)
    return E_FAIL;

  if ((pRangeComposition->SetText(ec, 0, preedit.c_str(),
                                  static_cast<LONG>(preedit.length()))) != S_OK)
    return E_FAIL;

  /* TODO: Check the availability and correctness of these values */
  int sel_cursor = -1;
  for (size_t i = 0; i < _context->preedit.attributes.size(); i++) {
    if (_context->preedit.attributes.at(i).type == weasel::HIGHLIGHTED) {
      sel_cursor = _context->preedit.attributes.at(i).range.cursor;
      break;
    }
  }

  _pTextService->_SetCompositionDisplayAttributes(ec, _pContext,
                                                  pRangeComposition);

  /* Set caret */
  LONG cch;
  TF_SELECTION tfSelection;
  if (sel_cursor < 0) {
    pRangeComposition->Collapse(ec, TF_ANCHOR_END);
  } else {
    pRangeComposition->Collapse(ec, TF_ANCHOR_START);
    pRangeComposition->ShiftStart(ec, sel_cursor, &cch, NULL);
  }
  tfSelection.range = pRangeComposition;
  tfSelection.style.ase = TF_AE_NONE;
  tfSelection.style.fInterimChar = FALSE;
  _pContext->SetSelection(ec, 1, &tfSelection);

  return S_OK;
}

BOOL WeaselTSF::_ShowInlinePreedit(
    com_ptr<ITfContext> pContext,
    const std::shared_ptr<weasel::Context> context) {
  com_ptr<CInlinePreeditEditSession> pEditSession;
  pEditSession.Attach(
      new CInlinePreeditEditSession(this, pContext, _pComposition, context));
  if (pEditSession != NULL) {
    HRESULT hr;
    pContext->RequestEditSession(_tfClientId, pEditSession,
                                 TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
  }
  return TRUE;
}

/* Update Composition */
class CInsertTextEditSession : public CEditSession {
 public:
  CInsertTextEditSession(com_ptr<WeaselTSF> pTextService,
                         com_ptr<ITfContext> pContext,
                         com_ptr<ITfComposition> pComposition,
                         const std::wstring& text)
      : CEditSession(pTextService, pContext),
        _text(text),
        _pComposition(pComposition) {}

  /* ITfEditSession */
  STDMETHODIMP DoEditSession(TfEditCookie ec);

 private:
  std::wstring _text;
  com_ptr<ITfComposition> _pComposition;
};

STDMETHODIMP CInsertTextEditSession::DoEditSession(TfEditCookie ec) {
  com_ptr<ITfRange> pRange;
  TF_SELECTION tfSelection;
  HRESULT hRet = S_OK;

  if (_pComposition == nullptr)
    return E_FAIL;
  if (FAILED(_pComposition->GetRange(&pRange)))
    return E_FAIL;

  if (FAILED(pRange->SetText(ec, 0, _text.c_str(),
                             static_cast<LONG>(_text.length()))))
    return E_FAIL;

  /* update the selection to an insertion point just past the inserted text. */
  pRange->Collapse(ec, TF_ANCHOR_END);

  tfSelection.range = pRange;
  tfSelection.style.ase = TF_AE_NONE;
  tfSelection.style.fInterimChar = FALSE;

  _pContext->SetSelection(ec, 1, &tfSelection);

  return hRet;
}

BOOL WeaselTSF::_InsertText(com_ptr<ITfContext> pContext,
                            const std::wstring& text) {
  CInsertTextEditSession* pEditSession;
  HRESULT hr;

  if ((pEditSession = new CInsertTextEditSession(this, pContext, _pComposition,
                                                 text)) != NULL) {
    pContext->RequestEditSession(_tfClientId, pEditSession,
                                 TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
    pEditSession->Release();
  }

  return TRUE;
}

void WeaselTSF::_UpdateComposition(com_ptr<ITfContext> pContext) {
  HRESULT hr;

  _pEditSessionContext = pContext;

  _pEditSessionContext->RequestEditSession(
      _tfClientId, this, TF_ES_ASYNCDONTCARE | TF_ES_READWRITE, &hr);
  _async_edit = !!(hr == TF_S_ASYNC);
}

/* Composition State */
STDMETHODIMP WeaselTSF::OnCompositionTerminated(TfEditCookie ecWrite,
                                                ITfComposition* pComposition) {
  // NOTE:
  // This will be called when an edit session ended up with an empty composition
  // string, Even if it is closed normally. Silly M$.

  // EndComposition() may generate this callback for the composition we just
  // closed. Only an active, matching composition is an external termination.
  if (!_IsCurrentComposition(pComposition))
    return S_OK;

  // A host may terminate the empty TSF composition used for a non-inline
  // preedit. Keep Rime's composing state; the next key will create a fresh
  // TSF composition. Only an inactive Rime session should be aborted here.
  if (_status.composing) {
    _FinalizeComposition();
    return S_OK;
  }

  _inlineGhostActive = FALSE;
  _ghostKeyAcceptPending = FALSE;
  _ghostSuppressAfterCancel = TRUE;
  weasel::inline_ghost::TraceLine(L"Ghost composition terminated externally");
  _AbortComposition();
  return S_OK;
}

void WeaselTSF::_AbortComposition(bool clear) {
  _inlineGhostActive = FALSE;
  _ghostKeyAcceptPending = FALSE;
  m_client.ClearComposition();
  if (_IsComposing()) {
    _EndComposition(_pEditSessionContext, clear);
  }
  _committed = TRUE;
  _cand->Destroy();
}

void WeaselTSF::_FinalizeComposition() {
  _pComposition = nullptr;
}

void WeaselTSF::_SetComposition(com_ptr<ITfComposition> pComposition) {
  _pComposition = pComposition;
}

BOOL WeaselTSF::_IsComposing() {
  return _pComposition != NULL;
}

BOOL WeaselTSF::_IsCurrentComposition(ITfComposition* pComposition) {
  return _pComposition != nullptr && _pComposition == pComposition;
}
