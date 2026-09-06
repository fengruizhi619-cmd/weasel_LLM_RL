using System;
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
  private static int _maxChars = 100;
  private static string _logPath = "";
  private static StreamWriter _log;
  private static readonly object _gate = new object();

  private static AutomationElement _subscribed;
  private static AutomationEventHandler _textChangedHandler;
  private static DateTime _lastKey = DateTime.MinValue;
  private static string _lastLogged = null;

  private static Native.LowLevelKeyboardProc _keyProc;
  private static IntPtr _hook = IntPtr.Zero;
  private static volatile bool _running = true;

  // [EXP-003 ENTRY]
  [STAThread]
  private static int Main(string[] args) {
    Console.OutputEncoding = Encoding.UTF8;
    ParseArgs(args);
    Console.WriteLine("[exp-v0] cli_emojiless_exp_v0 context reader (hook, no polling)");
    Console.WriteLine("[exp-v0] n=" + _maxChars + " log=" + (_logPath.Length > 0 ? _logPath : "(console only)"));
    if (_logPath.Length > 0) {
      _log = new StreamWriter(_logPath, true, new UTF8Encoding(false));
      _log.AutoFlush = true;
    }
    Console.CancelKeyPress += delegate { _running = false; Native.PostThreadMessageW(Native.GetCurrentThreadId(), Native.WM_QUIT, IntPtr.Zero, IntPtr.Zero); };

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
      }
    }
  }

  private static bool IsAsciiLetter(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
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
      Unsubscribe();
      if (!(bool)element.GetCurrentPropertyValue(AutomationElement.IsTextPatternAvailableProperty))
        return;
      if (IsSelfProcess(element)) return;
      _textChangedHandler = new AutomationEventHandler(OnTextChanged);
      Automation.AddAutomationEventHandler(TextPattern.TextChangedEvent, element,
                                           TreeScope.Element, _textChangedHandler);
      _subscribed = element;
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
    if (element == null || IsSelfProcess(element)) return;

    string context = ReadContext(element);
    if (context == null) return;

    string trimmed = context;
    if (trimmed.Length > _maxChars) trimmed = trimmed.Substring(trimmed.Length - _maxChars);
    // [EXP-006] commit-only trigger: while IME composition is live the
    // pre-caret text ends with ASCII pinyin letters; only when the candidate
    // is committed onto the document does a non-ASCII (CJK/punct) tail
    // appear. Treat that as the sole "上屏" trigger.
    if (trimmed.Length == 0 || IsAsciiLetter(trimmed[trimmed.Length - 1])) return;
    string stripped = Regex.Replace(trimmed, "[A-Za-z]+$", "");
    string keyed = (DateTime.Now - _lastKey).TotalMilliseconds <= 2500 ? "key" : "other";

    lock (_gate) {
      if (_lastLogged != null && _lastLogged == stripped) return;
      _lastLogged = stripped;
    }

    string line = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] src=" + keyed +
                  " rebuild=yes ctx(" + stripped.Length + "/" + _maxChars + "): " + stripped;
    LogLine(line);
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
