using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Automation;
using System.Windows.Automation.Text;

// [EXP-001 CONFIG]
// cli_emojiless_exp_v0 : context reader only (hook based, no polling).
// Pipeline: user key -> IME commit (UIA TextChanged) -> rebuild-tree(=yes)
//           -> read N chars before caret -> strip trailing [A-Za-z]+ -> log.

internal static class Native {
  internal const int WH_KEYBOARD_LL = 13;
  internal const int WM_KEYDOWN = 0x0100;
  internal const int WM_SYSKEYDOWN = 0x0104;
  internal const uint LLKHF_INJECTED = 0x10;
  internal const int WM_QUIT = 0x0012;

  internal delegate IntPtr LowLevelKeyboardProc(int nCode, IntPtr wParam, IntPtr lParam);

  [StructLayout(LayoutKind.Sequential)]
  internal struct KBDLLHOOKSTRUCT {
    public uint vkCode;
    public uint scanCode;
    public uint flags;
    public uint time;
    public IntPtr dwExtraInfo;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct MSG {
    public IntPtr hwnd;
    public uint message;
    public IntPtr wParam;
    public IntPtr lParam;
    public uint time;
    public int ptX;
    public int ptY;
  }

  [DllImport("user32.dll", SetLastError = true)]
  internal static extern IntPtr SetWindowsHookExW(int idHook, LowLevelKeyboardProc lpfn,
                                                  IntPtr hMod, uint dwThreadId);
  [DllImport("user32.dll")]
  internal static extern bool UnhookWindowsHookEx(IntPtr hhk);
  [DllImport("user32.dll")]
  internal static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);
  [DllImport("user32.dll")]
  internal static extern bool GetMessageW(out MSG msg, IntPtr hWnd, uint wMsgFilterMin, uint wMsgFilterMax);
  [DllImport("user32.dll")]
  internal static extern bool TranslateMessage(ref MSG msg);
  [DllImport("user32.dll")]
  internal static extern IntPtr DispatchMessageW(ref MSG msg);
  [DllImport("user32.dll")]
  internal static extern bool PostThreadMessageW(uint idThread, uint msg, IntPtr wParam, IntPtr lParam);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  internal static extern IntPtr GetModuleHandleW(string lpModuleName);
  [DllImport("kernel32.dll")]
  internal static extern uint GetCurrentThreadId();
}

// [EXP-002 HOOK-CONTEXT-READER]
internal static class Program {
  private static int _maxChars = 256;
  // [EXP-007] when set, also capture contexts that end in ASCII (English
  // typing). Off by default: with an IME active those are pinyin preedits.
  private static bool _asciiMode = false;
  private static string _logPath = "";
  private static StreamWriter _log;
  private static string _diagPath = "";
  private static StreamWriter _diag;
  private static readonly object _gate = new object();

  private static AutomationElement _subscribed;
  private static AutomationEventHandler _textChangedHandler;
  private static DateTime _lastKey = DateTime.MinValue;
  private static string _lastLogged = null;
  private static int _prevLen = 0;
  private static bool _hasPrev = false;

  private static Native.LowLevelKeyboardProc _keyProc;
  private static IntPtr _hook = IntPtr.Zero;
  private static volatile bool _running = true;

  // [EXP-008 DIAG]
  private static string CodePoints(string text) {
    var sb = new StringBuilder();
    for (int i = 0; i < text.Length; i++) {
      int cp;
      if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length &&
          char.IsLowSurrogate(text[i + 1])) {
        cp = char.ConvertToUtf32(text[i], text[i + 1]);
        i++;
      } else {
        cp = text[i];
      }
      if (sb.Length > 0)
        sb.Append(' ');
      sb.Append("U+").Append(cp.ToString("X4"));
    }
    return sb.ToString();
  }

  private static string RemovedSet(string raw, string clean) {
    var sb = new StringBuilder();
    for (int i = 0; i < raw.Length; i++) {
      string one = raw.Substring(i, 1);
      if (one != RemoveInvisible(one)) {
        if (sb.Length > 0)
          sb.Append(';');
        sb.Append(CodePoints(one));
      }
    }
    return sb.ToString();
  }

  private static void AppendDiag(string line) {
    lock (_gate) {
      if (_diag != null)
        _diag.WriteLine(line);
    }
  }

  // [EXP-003 SELF-TEST]
  private static void SelfTest() {
    string[] samples = {
      "核心文本一​",
      "﻿测试文本",
      "你好‍世界",
      "上下文⁠测试"
    };
    foreach (string sample in samples) {
      string clean = RemoveInvisible(sample);
      Console.WriteLine("raw=" + sample.Length + " clean=" + clean.Length +
                        " text=[" + clean + "]");
    }
    Console.WriteLine("selftest done");
  }

  // [EXP-003 ENTRY]
  [STAThread]
  private static int Main(string[] args) {
    // winexe build has no console: setting the encoding would throw.
    try { Console.OutputEncoding = Encoding.UTF8; } catch { }
    for (int i = 0; i < args.Length; i++) {
      if (args[i] == "-selftest" || args[i] == "--selftest") {
        SelfTest();
        return 0;
      }
    }
    ParseArgs(args);
    Console.WriteLine("[exp-v0] cli_emojiless_exp_v0 context reader (hook, no polling)");
    Console.WriteLine("[exp-v0] n=" + _maxChars + " log=" + (_logPath.Length > 0 ? _logPath : "(console only)"));
    if (_logPath.Length > 0) {
      _log = new StreamWriter(_logPath, true, new UTF8Encoding(false));
      _log.AutoFlush = true;
    }
    if (_diagPath.Length > 0) {
      _diag = new StreamWriter(_diagPath, true, new UTF8Encoding(false));
      _diag.AutoFlush = true;
    }
    try { Console.CancelKeyPress += delegate { _running = false; Native.PostThreadMessageW(Native.GetCurrentThreadId(), Native.WM_QUIT, IntPtr.Zero, IntPtr.Zero); }; } catch { }

    _keyProc = KeyboardProc;
    _hook = Native.SetWindowsHookExW(Native.WH_KEYBOARD_LL, _keyProc, Native.GetModuleHandleW(null), 0);
    if (_hook == IntPtr.Zero) {
      Console.WriteLine("[exp-v0] [warn] low-level keyboard hook failed, error=" + Marshal.GetLastWin32Error());
    } else {
      Console.WriteLine("[exp-v0] keyboard hook armed");
    }

    try {
      Automation.AddAutomationFocusChangedEventHandler(new AutomationFocusChangedEventHandler(OnFocusChanged));
      Console.WriteLine("[exp-v0] focus handler armed (switch to a text field and type)");
    } catch (Exception ex) {
      Console.WriteLine("[exp-v0] [error] cannot arm focus handler: " + ex.Message);
    }

    Native.MSG msg;
    while (_running) {
      if (!Native.GetMessageW(out msg, IntPtr.Zero, 0, 0))
        break;
      Native.TranslateMessage(ref msg);
      Native.DispatchMessageW(ref msg);
    }

    if (_hook != IntPtr.Zero) Native.UnhookWindowsHookEx(_hook);
    try { Automation.RemoveAutomationFocusChangedEventHandler(new AutomationFocusChangedEventHandler(OnFocusChanged)); } catch { }
    if (_log != null) _log.Close();
    if (_diag != null) _diag.Close();
    Console.WriteLine("[exp-v0] stopped");
    return 0;
  }

  private static void ParseArgs(string[] args) {
    for (int i = 0; i < args.Length; i++) {
      if ((args[i] == "-n" || args[i] == "--n") && i + 1 < args.Length) {
        int parsed;
        if (int.TryParse(args[i + 1], out parsed) && parsed > 0 && parsed <= 4096) _maxChars = parsed;
        i++;
      } else if ((args[i] == "-log" || args[i] == "--log") && i + 1 < args.Length) {
        _logPath = args[i + 1];
        i++;
      } else if ((args[i] == "-diag" || args[i] == "--diag") && i + 1 < args.Length) {
        _diagPath = args[i + 1];
        i++;
      } else if (args[i] == "-ascii" || args[i] == "--ascii") {
        _asciiMode = true;
      }
    }
  }

  private static bool IsAsciiLetter(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  }

  // [EXP-007 INVISIBLE-CLEAN]
  // Strip zero-width / format characters an IME or editor may inject into the
  // document (ZWNJ/ZWJ, zero-width space, BOM/FEFF, word joiner, LTR/RTL
  // marks, interlinear annotation, etc.) so visible text and its length are
  // stable across duplicate TextChanged events.
  private static string RemoveInvisible(string text) {
    if (string.IsNullOrEmpty(text))
      return text;
    var sb = new StringBuilder(text.Length);
    foreach (char c in text) {
      if (c == '\u200b' || c == '\u200c' || c == '\u200d' || c == '\u200e' ||
          c == '\u200f' || c == '\u202a' || c == '\u202b' || c == '\u202c' ||
          c == '\u202d' || c == '\u202e' || c == '\u2060' || c == '\u2061' ||
          c == '\u2062' || c == '\u2063' || c == '\u2064' || c == '\u2066' ||
          c == '\u2067' || c == '\u2068' || c == '\u2069' || c == '\u206a' ||
          c == '\u206b' || c == '\u206c' || c == '\u206d' || c == '\u206e' ||
          c == '\u206f' || c == '\ufeff' || c == '\ufff9' || c == '\ufffa' ||
          c == '\ufffb') {
        continue;
      }
      if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.Format)
        continue;
      sb.Append(c);
    }
    return sb.ToString();
  }

  private static void LogLine(string text) {
    lock (_gate) {
      Console.WriteLine(text);
      if (_log != null) _log.WriteLine(text);
    }
  }

  // [EXP-004 FOCUS-REBIND]
  private static void OnFocusChanged(object sender, AutomationFocusChangedEventArgs e) {
    AutomationElement element = null;
    try { element = AutomationElement.FocusedElement; } catch { }
    Rebind(element);
  }

  private static void Rebind(AutomationElement element) {
    if (element == null) return;
    try {
      if (element == _subscribed) return;
      if (IsConsoleHost(element)) return;
      Unsubscribe();
      if (!(bool)element.GetCurrentPropertyValue(AutomationElement.IsTextPatternAvailableProperty))
        return;
      if (IsSelfProcess(element)) return;
      _textChangedHandler = new AutomationEventHandler(OnTextChanged);
      Automation.AddAutomationEventHandler(TextPattern.TextChangedEvent, element,
                                           TreeScope.Element, _textChangedHandler);
      _subscribed = element;
      _prevLen = 0;
      _hasPrev = false;
      LogLine("[focus] subscribed text-changed on " + Describe(element));
    } catch { }
  }

  private static void Unsubscribe() {
    try {
      if (_subscribed != null && _textChangedHandler != null)
        Automation.RemoveAutomationEventHandler(TextPattern.TextChangedEvent, _subscribed, _textChangedHandler);
    } catch { }
    _subscribed = null;
    _textChangedHandler = null;
  }

  private static bool IsSelfProcess(AutomationElement element) {
    try {
      int pid = (int)element.GetCurrentPropertyValue(AutomationElement.ProcessIdProperty);
      return pid == System.Diagnostics.Process.GetCurrentProcess().Id;
    } catch {
      return false;
    }
  }

  // [EXP-009 CONSOLE-GUARD]
  // Never subscribe to the console/terminal that hosts this reader: printing
  // log lines changes the terminal text, which fires TextChanged and creates
  // a self-feedback loop (the observed duplicate lines with CR/LF + spaces).
  private static bool IsConsoleHost(AutomationElement element) {
    try {
      string className = element.Current.ClassName;
      if (className != null) {
        string cl = className.ToLowerInvariant();
        if (cl.Contains("termcontrol") || cl.Contains("consolewindow") ||
            cl.Contains("cascadia"))
          return true;
      }
      int pid = (int)element.GetCurrentPropertyValue(AutomationElement.ProcessIdProperty);
      using (System.Diagnostics.Process proc = System.Diagnostics.Process.GetProcessById(pid)) {
        string name = proc.ProcessName.ToLowerInvariant();
        if (name == "windowsterminal" || name == "windowsterminalpreview" ||
            name == "openconsole" || name == "conhost" || name == "cmd" ||
            name == "powershell" || name == "pwsh" || name == "wezterm" ||
            name == "mintty" || name == "alacritty")
          return true;
      }
    } catch {
      return false;
    }
    return false;
  }

  private static string Describe(AutomationElement element) {
    try {
      return element.Current.Name + " / " + element.Current.ClassName;
    } catch {
      return "(unknown)";
    }
  }

  // [EXP-005 KEYBOARD-HOOK]
  private static IntPtr KeyboardProc(int nCode, IntPtr wParam, IntPtr lParam) {
    if (nCode >= 0) {
      int msg = wParam.ToInt32();
      if (msg == Native.WM_KEYDOWN || msg == Native.WM_SYSKEYDOWN) {
        Native.KBDLLHOOKSTRUCT kbd = (Native.KBDLLHOOKSTRUCT)Marshal.PtrToStructure(lParam, typeof(Native.KBDLLHOOKSTRUCT));
        if ((kbd.flags & Native.LLKHF_INJECTED) == 0) {
          _lastKey = DateTime.Now;
        }
      }
    }
    return Native.CallNextHookEx(_hook, nCode, wParam, lParam);
  }

  // [EXP-006 TEXT-CHANGED PIPELINE]
  private static void OnTextChanged(object sender, AutomationEventArgs e) {
    AutomationElement element = sender as AutomationElement;
    if (element == null || IsSelfProcess(element) || IsConsoleHost(element)) return;

    string context = ReadContext(element);
    if (context == null) return;
    string rawContext = context;
    context = RemoveInvisible(context);

    string trimmed = context;
    if (trimmed.Length > _maxChars) trimmed = trimmed.Substring(trimmed.Length - _maxChars);
    // [EXP-006] commit-only trigger: while IME composition is live the
    // pre-caret text ends with ASCII pinyin letters; only when the candidate
    // is committed onto the document does a non-ASCII (CJK/punct) tail
    // appear. Treat that as the sole "上屏" trigger, unless -ascii is given.
    if (trimmed.Length == 0) return;
    bool asciiTail = IsAsciiLetter(trimmed[trimmed.Length - 1]);
    if (asciiTail && !_asciiMode) return;
    string stripped = Regex.Replace(trimmed, "[A-Za-z]+$", "").TrimEnd();
    if (string.IsNullOrWhiteSpace(stripped)) return;
    string keyed = (DateTime.Now - _lastKey).TotalMilliseconds <= 2500 ? "key" : "other";

    int delta;
    lock (_gate) {
      // [EXP-007] record every change, including backspaces and replacements.
      // The RL side classifies them (predicted-reject / typing-reject /
      // replace); dropping them here silently removed all negative samples.
      if (_lastLogged != null && _lastLogged == stripped) return;
      delta = _hasPrev ? stripped.Length - _prevLen : 0;
      _hasPrev = true;
      _prevLen = stripped.Length;
      _lastLogged = stripped;
    }

    string line = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] src=" + keyed +
                  (asciiTail ? "/ascii" : "") + " delta=" + (delta >= 0 ? "+" : "") + delta +
                  " rebuild=yes ctx(" + stripped.Length + "/" + _maxChars + "): " + stripped;
    LogLine(line);
    AppendDiag("[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] rawN=" + rawContext.Length +
               " cleanN=" + stripped.Length + " removed=[" + RemovedSet(rawContext, trimmed) +
               "]\n  raw_cps=" + CodePoints(rawContext.Length > _maxChars
                   ? rawContext.Substring(rawContext.Length - _maxChars)
                   : rawContext) +
               "\n  clean_cps=" + CodePoints(stripped));
  }

  private static string ReadContext(AutomationElement element) {
    try {
      if ((bool)element.GetCurrentPropertyValue(AutomationElement.IsTextPatternAvailableProperty)) {
        TextPattern text = (TextPattern)element.GetCurrentPattern(TextPattern.Pattern);
        TextPatternRange document = text.DocumentRange;
        TextPatternRange[] selections = text.GetSelection();
        string before = null;
        if (selections != null && selections.Length > 0) {
          TextPatternRange beforeRange = document.Clone();
          beforeRange.MoveEndpointByRange(TextPatternRangeEndpoint.End, selections[0], TextPatternRangeEndpoint.Start);
          before = beforeRange.GetText(-1);
        } else {
          before = document.GetText(-1);
        }
        return before ?? string.Empty;
      }
      if ((bool)element.GetCurrentPropertyValue(AutomationElement.IsValuePatternAvailableProperty)) {
        ValuePattern value = (ValuePattern)element.GetCurrentPattern(ValuePattern.Pattern);
        return value.Current.Value ?? string.Empty;
      }
    } catch {
      return null;
    }
    return null;
  }
}
