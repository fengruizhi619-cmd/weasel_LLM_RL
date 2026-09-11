# -*- coding: utf-8 -*-
"""[CTX-001] Tell what the user just did, given two pre-caret windows.

The hook (tools/WeaselExpContextV0) and the recorder only ever see the last N
characters in front of the caret, and N is a cap. Once the document is longer
than that the window no longer grows, it SLIDES - so the old test, "is prev a
prefix of ctx", quietly stops being true for a plain append. Measured on
diag/corpus.jsonl: 213 records whose context was exactly 100 characters, every
one of them classified as "replace", every one of them an append. The online
loop therefore stopped producing positives - and, through the same prefix test
inside classify_backspace(), negatives - as soon as a document got long.

The fix is to compare on the overlap rather than on the prefix:

  m1 = the largest m with prev[-m:] == ctx[:m]
       the tail of prev is the head of ctx  -> the rest of ctx was APPENDED
  m2 = the largest m with ctx[-m:] == prev[:m]
       the tail of ctx is the head of prev  -> the rest of prev was DELETED

Both cases need m to be large enough to be believable, which is what max_change
does: a 1-character coincidence on two 100-character windows would otherwise
read as "the user typed 99 characters".
"""

# An edit longer than this is not typing - it is a paste, a select-all retype or
# a document switch, and treating it as a training sample is how the live loop
# would otherwise freeze the IME on a 2000-step backward pass.
DEFAULT_MAX_CHANGE = 64


def longest_overlap(left, right):
    """Largest m with left[-m:] == right[:m] (m <= min(len(left), len(right))).

    O(min(len)) slices, and both windows are capped at a few hundred
    characters, so this costs nothing next to a forward pass.
    """
    m = min(len(left), len(right))
    while m > 0 and left[len(left) - m:] != right[:m]:
        m -= 1
    return m


def classify_change(prev, ctx, max_change=DEFAULT_MAX_CHANGE):
    """Return (kind, text) describing the edit between two pre-caret windows.

      ("append",  typed)    the user added `typed` at the caret
      ("delete",  removed)  the user removed `removed` before the caret
      ("replace", removed)  something else (select + retype, paste, jump):
                            `removed` is the old tail that is gone
      ("none",    "")       nothing to report
    """
    if prev is None or prev == ctx:
        return "none", ""

    m1 = longest_overlap(prev, ctx)          # appended candidate
    m2 = longest_overlap(ctx, prev)          # deleted candidate
    typed = ctx[m1:] if m1 > 0 else ""
    removed = prev[m2:] if m2 > 0 else ""

    append_ok = bool(typed) and len(typed) <= max_change
    delete_ok = bool(removed) and len(removed) <= max_change

    if append_ok and (not delete_ok or m1 >= m2):
        return "append", typed
    if delete_ok and (not append_ok or m2 > m1):
        return "delete", removed
    if m1 == 0 and m2 == 0:
        common = 0
        while (common < len(prev) and common < len(ctx)
               and prev[common] == ctx[common]):
            common += 1
        return "replace", prev[common:]
    # the two readings disagree and neither is small enough to trust
    return "replace", prev[min(m1, m2):]
