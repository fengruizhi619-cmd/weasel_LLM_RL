#include "stdafx.h"
#include <fstream>
#include "WeaselTSF.h"

void WeaselTSF::_PublishTsContext(ITfContext* pContext, TfEditCookie ec) {
  if (pContext == nullptr || _IsComposing() || _HasRimeComposition())
    return;

  TF_SELECTION selection{};
  ULONG selection_count = 0;
  if (FAILED(pContext->GetSelection(ec, TF_DEFAULT_SELECTION, 1, &selection,
                                    &selection_count)) ||
      selection_count < 1 || selection.range == nullptr)
    return;

  BOOL is_empty = FALSE;
  if (FAILED(selection.range->IsEmpty(ec, &is_empty)) || !is_empty)
    return;

  com_ptr<ITfRange> prefix_range;
  if (FAILED(selection.range->Clone(&prefix_range)) || prefix_range == nullptr)
    return;
  if (FAILED(prefix_range->Collapse(ec, TF_ANCHOR_END)))
    return;
  LONG shifted = 0;
  if (FAILED(prefix_range->ShiftStart(ec, -100, &shifted, nullptr)))
    return;

  wchar_t buffer[101]{};
  ULONG fetched = 0;
  if (FAILED(prefix_range->GetText(ec, 0, buffer, 100, &fetched)))
    return;

  const std::wstring context =
      weasel::inline_ghost::CleanContext(std::wstring(buffer, fetched));
  if (context.empty())
    return;

  const uint64_t context_hash =
      weasel::inline_ghost::HashContext(context);
  if (_ghostSuppressAfterCancel) {
    if (context_hash == _inlineGhostBaseContextHash)
      return;
    _ghostSuppressAfterCancel = FALSE;
  }
  if (context_hash == _inlineGhostCombinedHash)
    return;

  const std::string utf8 = weasel::inline_ghost::Utf8FromWide(context);
  const std::string line = "[tsf] src=tsf rebuild=yes ctx(" +
                           std::to_string(context.size()) + "/100): " + utf8 +
                           std::string("\n");
  const std::wstring path = weasel::inline_ghost::TsContextFilePath();
  if (path.empty())
    return;
  std::ofstream output(path.c_str(), std::ios::app | std::ios::binary);
  if (output.is_open())
    output.write(line.data(), static_cast<std::streamsize>(line.size()));
}

static BOOL IsRangeCovered(TfEditCookie ec,
                           ITfRange* pRangeTest,
                           ITfRange* pRangeCover) {
  LONG lResult;

  if (pRangeCover->CompareStart(ec, pRangeTest, TF_ANCHOR_START, &lResult) !=
          S_OK ||
      lResult > 0)
    return FALSE;
  if (pRangeCover->CompareEnd(ec, pRangeTest, TF_ANCHOR_END, &lResult) !=
          S_OK ||
      lResult < 0)
    return FALSE;
  return TRUE;
}

STDMETHODIMP WeaselTSF::OnEndEdit(ITfContext* pContext,
                                  TfEditCookie ecReadOnly,
                                  ITfEditRecord* pEditRecord) {
  BOOL fSelectionChanged;
  IEnumTfRanges* pEnumTextChanges;
  ITfRange* pRange;
  bool text_changed = false;

  /* did the selection change? */
  if (pEditRecord->GetSelectionStatus(&fSelectionChanged) == S_OK &&
      fSelectionChanged) {
    if (_IsComposing()) {
      /* if the caret moves out of composition range, stop the composition */
      TF_SELECTION tfSelection;
      ULONG cFetched;

      if (pContext->GetSelection(ecReadOnly, TF_DEFAULT_SELECTION, 1,
                                 &tfSelection, &cFetched) == S_OK &&
          cFetched == 1) {
        ITfRange* pRangeComposition;
        if (_pComposition->GetRange(&pRangeComposition) == S_OK) {
          if (!IsRangeCovered(ecReadOnly, tfSelection.range, pRangeComposition))
            _EndComposition(pContext, true);
          pRangeComposition->Release();
        }
      }
    }
  }

  /* text modification? */
  if (pEditRecord->GetTextAndPropertyUpdates(TF_GTP_INCL_TEXT, NULL, 0,
                                             &pEnumTextChanges) == S_OK) {
    if (pEnumTextChanges->Next(1, &pRange, NULL) == S_OK) {
      text_changed = true;
      pRange->Release();
    }
    pEnumTextChanges->Release();
  }

  if (text_changed && !_IsComposing() && !_HasRimeComposition())
    _PublishTsContext(pContext, ecReadOnly);

  if (!_IsComposing() && !_HasRimeComposition())
    _ScheduleInlineGhostCheck();
  return S_OK;
}

STDMETHODIMP WeaselTSF::OnLayoutChange(ITfContext* pContext,
                                       TfLayoutCode lcode,
                                       ITfContextView* pContextView) {
  if (!_IsComposing())
    return S_OK;

  if (pContext != _pTextEditSinkContext)
    return S_OK;

  if (lcode == TF_LC_CHANGE)
    _UpdateCompositionWindow(pContext);
  return S_OK;
}

BOOL WeaselTSF::_InitTextEditSink(com_ptr<ITfDocumentMgr> pDocMgr) {
  com_ptr<ITfSource> pSource;
  BOOL fRet;

  /* clear out any previous sink first */
  if (_dwTextEditSinkCookie != TF_INVALID_COOKIE) {
    _pTextEditSinkContext->QueryInterface(&pSource);
    if (pSource != nullptr) {
      pSource->UnadviseSink(_dwTextEditSinkCookie);
      pSource->UnadviseSink(_dwTextLayoutSinkCookie);
    }
    _pTextEditSinkContext = nullptr;
    _dwTextEditSinkCookie = TF_INVALID_COOKIE;
  }
  if (pDocMgr == NULL)
    return TRUE;

  if (pDocMgr->GetTop(&_pTextEditSinkContext) != S_OK)
    return FALSE;

  if (_pTextEditSinkContext == NULL)
    return TRUE;

  fRet = FALSE;

  pSource.Release();

  if (_pTextEditSinkContext->QueryInterface(IID_ITfSource, (void**)&pSource) ==
      S_OK) {
    if (pSource->AdviseSink(IID_ITfTextEditSink, (ITfTextEditSink*)this,
                            &_dwTextEditSinkCookie) == S_OK)
      fRet = TRUE;
    else
      _dwTextEditSinkCookie = TF_INVALID_COOKIE;
    if (pSource->AdviseSink(IID_ITfTextLayoutSink, (ITfTextLayoutSink*)this,
                            &_dwTextLayoutSinkCookie) == S_OK) {
      fRet = TRUE;
    } else
      _dwTextLayoutSinkCookie = TF_INVALID_COOKIE;
  }
  if (fRet == FALSE) {
    _pTextEditSinkContext = nullptr;
  }

  return fRet;
}
