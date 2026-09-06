# WeaselExpContextV0 (cli_emojiless_exp_v0)

Experiment version on top of cli_emojiless_base. Keeps the Weasel core
untouched and adds context reading as an external hook process.

- No polling: event driven.
- Trigger state machine: user key -> IME commit (UIA TextChanged)
  -> rebuild tree? (exp: always yes) -> read N chars before caret
  -> strip trailing ASCII letters [A-Za-z]+ -> log context (console + file).
- Keyboard hook (WH_KEYBOARD_LL) only annotates whether the change came
  from physical keys (src=key) or elsewhere (src=other).

Usage:
  WeaselExpContextV0.exe [-n 100] [-log <path>]

All future features follow the same rule: implemented as hooks, never by
modifying the Weasel core.
