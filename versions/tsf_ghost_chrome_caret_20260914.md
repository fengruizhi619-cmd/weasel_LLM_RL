# TSF 联想失效修复：Chromium 类应用拿不到光标框（GHOST-FIX-012）

日期：2026-09-14　标签：`[GHOST-FIX-012]`

## 症状

- **记事本**里联想正常。
- **Chrome** 和 **DSH Desktop（Electron）** 里联想完全不出现。
- 输入法本身正常（能打字、候选窗正常），只有联想不出。

注意两个失败应用**都是 Chromium**（DSH Desktop 是 Electron），所以这不是两个独立故障，
而是一类应用的同一个问题。

## 证据链

1. **TSF 是活的、DLL 也没错**：Chrome(PID 41848) 与 DSH Desktop(PID 41292) 的模块表里都有
   `weaselx64.dll`，路径都是 `C:\Program Files\Rime\weasel-0.17.4-emoji-off\weaselx64.dll`，
   与注册表里 TSF 文本服务 CLSID `{A3F4CDED-B1E9-41EE-9CA6-7BD40DE6CB0A}` 的
   `InprocServer32` 指向同一个文件。
   → 排除"旧 TSF 残留 / x86-x64 路径错 / DLL 没加载"这几类。

2. **`%TEMP%\rime.weasel\ghost-debug.log` 给出直接死因**（该文件由 `WEASEL_GHOST_TRACE=1` 打开）：

   ```
   [11356] snapshot hidden: caret rect unavailable      ← 连续 106 次
   [11356] snapshot accepted; prefix_chars=0; prefix=
   ```

3. **失败线程归属定位到具体应用**：`Get-Process | ? Threads.Id -eq 11356`
   → **DSH Desktop PID 41292**（就是用户正在打字的窗口）。

4. **代码级定位**（`WeaselTSF.cpp::_ReadGhostPrefix`，原文）：

   ```cpp
   context_view->GetTextExt(ecReadOnly, selection.range, &rect, &clipped);  // 返回值被丢弃
   if (rect.left == 0 && rect.top == 0) {
     ... GetCaretPos(&caret_point) ...        // 需要 Win32 原生光标
   }
   if (rect.left == 0 && rect.top == 0) {
     TraceLine(L"snapshot hidden: caret rect unavailable"); return false;
   }
   ```

## 根因

同一个 `ITfContextView::GetTextExt`，**候选窗定位能用而联想不能用** —— 两者唯一的差别是
调用位置：

| 路径 | 调用位置 | Chromium 下 |
|---|---|---|
| 候选窗定位 `CGetTextExtentEditSession` | **独立申请的只读编辑会话**（`RequestEditSession`） | 正常 |
| 联想快照 `_ReadGhostPrefix` | **`OnEndEdit` 回调内部** | 拿不到光标框 |

三个叠加缺陷：

1. **`GetTextExt` 的返回值被丢弃** —— 失败与"成功但矩形全零"分不开，出问题连日志都没有。
2. **有效性判据写成 `left == 0 && top == 0`** —— 既误杀屏幕原点附近的合法位置，
   又分不清退化矩形，而 Chromium 给不出时返回的正是全零矩形。
3. **唯一的兜底 `GetCaretPos` 需要 Win32 原生光标**（`CreateCaret`/`ShowCaret`）。
   **Chromium 自己画光标、从不创建原生光标**，所以这条在 Chrome / Electron 里必然失败；
   记事本是经典 Win32 控件、有原生光标，于是"只有记事本能用"。
4. 另外 `OnLayoutChange` 第一句就是 `if (!_IsComposing()) return S_OK;` ——
   **布局就绪这个唯一能救回快照的时机被整个跳过**，快照请求（`_expSnapshotPending`）
   于是永远悬着。

## 修复

1. `_ReadGhostPrefix`：保留并**记录** `GetActiveView` / `GetTextExt` 的真实 HRESULT、
   `clipped` 与原始矩形（`LlmLog` 常开，带 pid/tid，不必再开 trace）。
2. 新增判据 `GhostRectUsable()`：矩形退化（右≤左且下≤上）或完全落在虚拟屏幕外才算不可用。
3. 失败时先退一步用**单字符区间**再取一次（空区间在 Chromium 上尤其容易返回全零），
   再兜底 `GetGUIThreadInfo(0,&gti).rcCaret`（比 `GetCaretPos` 通用）。
4. 新增 `_RequestGhostSnapshot()`：另起**只读编辑会话**（`TF_ES_ASYNCDONTCARE | TF_ES_READ`，
   与候选窗定位同一条已验证可用的路子）重新采集。
5. `OnLayoutChange`：非组字态且快照仍挂起时补一次采集。
6. 重试带预算 `_ghost_retry_budget`（每次请求新快照时充满 3 次），
   防在布局迟迟不就绪的应用里无限重试拖住输入。

## 安装与验证

- 构建：`output\build_v7.bat`（只编 `WeaselTSF.vcxproj` x64 Release）。
- 安装：目标 DLL 被 Chrome / DSH Desktop / explorer / maid_kit / SearchHost 占用，
  直接覆盖会失败 → **改名换位**：先 `move` 成 `.bak`（已加载的 DLL 允许重命名），
  再把新 DLL 复制到位。需要 UAC 提权（`Start-Process -Verb RunAs`）。
  因路径含中文，提权脚本走 `%TEMP%` 下的纯 ASCII `.bat`，结果写标记文件回读。
- **必须重启应用才生效**：已加载的进程仍映射旧文件。→ 重启 Chrome 与 DSH Desktop。
- 验证判据：重启后在新进程里打一个字，`llm-ime.log` 应出现
  `ghost caret rect view_hr=… ext_hr=… rect=… ok=1`；`ghost-debug.log` 不再刷
  `caret rect unavailable`。
- 回滚：`%TEMP%\weaselx64.dll.bak` 换回原名即可（提权）。

## 顺带记录

- 这次是**共用云端 pod** 上另一工序的 `cleanup.sh` 把 `/root/distill` 当"上一个工序残留"
  整目录删除（其脚本里明写 `rm -rf /root/distill`），导致两轮实验结果丢失。
  工程教训：外机工作目录要命名空间化并放 `DO_NOT_DELETE.md`；结果要边跑边拉回本地；
  队列脚本要带前置校验，避免"文件缺失 → 每臂秒失败 → 看着像跑过了"。
