#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace weasel {
namespace ghost {

constexpr int kTreeMaxDepth = 10;

// [TREE-001 BRANCH-BOUNDARY]
inline bool IsBranchStop(wchar_t ch) {
  return ch == L'.' || ch == L',' || ch == L'!' || ch == L'?' ||
         ch == L'。' || ch == L'，' || ch == L'！' || ch == L'？' ||
         ch == L'、' || ch == L'；' || ch == L'：';
}

// [TREE-010 PINYIN] One model candidate: surface char + probability + pinyin.
struct Candidate {
  wchar_t ch = 0;
  double prob = 0.0;
  std::wstring pinyin;
};

struct TreeNode {
  wchar_t ch = 0;
  std::wstring pinyin;
  double prob = 0.0;
  double cum = 0.0;
  int parent = -1;
  int depth = 0;
  bool terminal = false;
  bool children_generated = false;
  std::vector<int> children;
};

// [TREE-000 CLASS]
class CandidateTree {
 public:
  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodes_.empty();
  }

  // [TREE-002 RESET]
  void Reset(const std::wstring& base_prefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    nodes_.clear();
    nodes_.push_back(TreeNode{});
    nodes_[0].cum = 1.0;
    base_prefix_ = base_prefix;
  }

  std::wstring BasePrefix() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return base_prefix_;
  }

  // [TREE-003 MATCH]
  int MatchPath(const std::wstring& current_prefix) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.empty())
      return -1;
    if (current_prefix.size() < base_prefix_.size())
      return -1;
    if (base_prefix_.compare(
            0, base_prefix_.size(), current_prefix, 0, base_prefix_.size()) != 0)
      return -1;

    int node = 0;
    for (size_t i = base_prefix_.size(); i < current_prefix.size(); ++i) {
      node = FindChildLocked(node, current_prefix[i]);
      if (node < 0)
        return -1;
    }
    return node;
  }

  int FindChild(int parent, wchar_t ch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return FindChildLocked(parent, ch);
  }

  // [TREE-004 EXPANSION-STATE]
  // |force| lets a terminal (punctuation) node grow when the user actually
  // typed it: display still never crosses punctuation, but the tree must be
  // able to continue after the user commits the punctuation character.
  bool NeedsExpansion(int node, bool force = false) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return false;
    const auto& item = nodes_[node];
    return item.depth < kTreeMaxDepth && (force || !item.terminal) &&
           !item.children_generated;
  }

  // [TREE-008 FRONTIER] Nodes that still need expansion.
  std::vector<int> Frontier() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& item = nodes_[i];
      if (item.depth < kTreeMaxDepth && !item.terminal &&
          !item.children_generated)
        result.push_back(static_cast<int>(i));
    }
    return result;
  }

  bool IsTerminal(int node) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node >= 0 && node < static_cast<int>(nodes_.size()) &&
           nodes_[node].terminal;
  }

  int Depth(int node) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node >= 0 && node < static_cast<int>(nodes_.size())
               ? nodes_[node].depth
               : 0;
  }

  std::wstring PathFromRoot(int node) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::wstring reversed;
    while (node > 0 && node < static_cast<int>(nodes_.size())) {
      reversed.push_back(nodes_[node].ch);
      node = nodes_[node].parent;
    }
    return std::wstring(reversed.rbegin(), reversed.rend());
  }

  // [TREE-005 BEST-SUFFIX]
  // Terminal punctuation is retained for prefix matching but is never selected
  // as visible continuation.
  std::wstring BestSuffix(int node, const std::wstring& pinyin_filter = L"",
                          int* end_node = nullptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::wstring suffix;
    int cursor = node;
    std::wstring remaining = pinyin_filter;
    while (cursor >= 0 && cursor < static_cast<int>(nodes_.size())) {
      const auto& item = nodes_[cursor];
      int best = -1;
      std::wstring consumed;
      // [TREE-011 PINYIN-PATH] Walk the tree consuming the composition pinyin
      // character by character: at every step keep only branches whose pinyin
      // matches the still-unconsumed part, then take the highest cum among
      // them. Once the composition is used up the path is unconstrained.
      if (!remaining.empty()) {
        for (int child : item.children) {
          if (nodes_[child].terminal)
            continue;
          const std::wstring& cp = nodes_[child].pinyin;
          if (cp.empty())
            continue;
          std::wstring take;
          if (remaining.size() >= cp.size() &&
              remaining.compare(0, cp.size(), cp) == 0)
            take = cp;
          else if (cp.size() > remaining.size() &&
                   cp.compare(0, remaining.size(), remaining) == 0)
            take = remaining;
          else
            continue;
          if (best < 0 || nodes_[child].cum > nodes_[best].cum) {
            best = child;
            consumed = take;
          }
        }
        if (best < 0)
          remaining.clear();
      }
      if (best < 0) {
        for (int child : item.children) {
          if (nodes_[child].terminal)
            continue;
          if (best < 0 || nodes_[child].cum > nodes_[best].cum)
            best = child;
        }
      } else {
        remaining = remaining.substr(consumed.size());
      }
      if (best < 0)
        break;
      suffix.push_back(nodes_[best].ch);
      cursor = best;
    }
    if (end_node != nullptr)
      *end_node = cursor;
    return suffix;
  }

  // [TREE-006 BEST-CHILD]
  int BestChild(int node, const std::wstring& pinyin_filter = L"") const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return -1;
    if (nodes_[node].children.empty())
      return -1;

    int best = -1;
    if (!pinyin_filter.empty()) {
      for (int child : nodes_[node].children) {
        if (nodes_[child].terminal ||
            !PinyinMatchesLocked(child, pinyin_filter))
          continue;
        if (best < 0 || nodes_[child].cum > nodes_[best].cum)
          best = child;
      }
    }
    if (best < 0) {
      for (int child : nodes_[node].children) {
        if (nodes_[child].terminal)
          continue;
        if (best < 0 || nodes_[child].cum > nodes_[best].cum)
          best = child;
      }
    }
    return best;
  }

  std::vector<int> Children(int node) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return result;
    for (int child : nodes_[node].children) {
      if (!nodes_[child].terminal)
        result.push_back(child);
    }
    return result;
  }

  // Keep only the path from the root to |node| and |node|'s subtree;
  // all sibling branches are dropped. Returns the new index of |node|.
  int PruneTo(int node) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return -1;

    std::vector<int> path;
    int cursor = node;
    while (cursor >= 0) {
      path.push_back(cursor);
      cursor = nodes_[cursor].parent;
    }
    std::reverse(path.begin(), path.end());

    std::vector<bool> keep(nodes_.size(), false);
    for (int item : path)
      keep[item] = true;
    std::vector<int> stack;
    stack.push_back(node);
    while (!stack.empty()) {
      int item = stack.back();
      stack.pop_back();
      keep[item] = true;
      for (int child : nodes_[item].children)
        stack.push_back(child);
    }

    std::vector<int> remap(nodes_.size(), -1);
    std::vector<TreeNode> rebuilt;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (!keep[i])
        continue;
      remap[i] = static_cast<int>(rebuilt.size());
      TreeNode copy = nodes_[i];
      copy.parent = -1;
      copy.children.clear();
      rebuilt.push_back(copy);
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (!keep[i])
        continue;
      int target = remap[i];
      if (nodes_[i].parent >= 0 && remap[nodes_[i].parent] >= 0)
        rebuilt[target].parent = remap[nodes_[i].parent];
      for (int child : nodes_[i].children) {
        if (remap[child] >= 0)
          rebuilt[target].children.push_back(remap[child]);
      }
    }
    nodes_ = std::move(rebuilt);
    return remap[node];
  }

  // [TREE-009 PRUNE-REBASE] Keep only |node| and its subtree, promote |node| to
  // root (index 0), rebase depths so the depth budget is restored, and move
  // base_prefix_ forward to |new_base|. This is what lets the engine follow
  // the user along an already predicted path instead of rebuilding.
  int PruneToRebased(int node, const std::wstring& new_base) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return -1;

    std::vector<int> keep;
    std::vector<int> stack;
    stack.push_back(node);
    while (!stack.empty()) {
      int item = stack.back();
      stack.pop_back();
      keep.push_back(item);
      for (int child : nodes_[item].children)
        stack.push_back(child);
    }

    const int base_depth = nodes_[node].depth;
    std::vector<int> remap(nodes_.size(), -1);
    std::vector<TreeNode> rebuilt;
    for (int item : keep) {
      remap[item] = static_cast<int>(rebuilt.size());
      TreeNode copy = nodes_[item];
      copy.parent = -1;
      copy.children.clear();
      copy.depth = nodes_[item].depth - base_depth;
      rebuilt.push_back(copy);
    }
    for (int item : keep) {
      int target = remap[item];
      if (nodes_[item].parent >= 0 && remap[nodes_[item].parent] >= 0)
        rebuilt[target].parent = remap[nodes_[item].parent];
      for (int child : nodes_[item].children) {
        if (remap[child] >= 0)
          rebuilt[target].children.push_back(remap[child]);
      }
    }
    nodes_ = std::move(rebuilt);
    base_prefix_ = new_base;
    return 0;
  }

  double Cum(int node) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node < 0 || node >= static_cast<int>(nodes_.size()))
      return 0.0;
    return nodes_[node].cum;
  }

  // [TREE-007 SET-CANDIDATES]
  void SetCandidates(int parent, const std::vector<Candidate>& candidates) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (parent < 0 || parent >= static_cast<int>(nodes_.size()))
      return;

    nodes_[parent].children.clear();
    for (const auto& candidate : candidates) {
      if (candidate.ch == 0)
        continue;

      int existing = FindChildLocked(parent, candidate.ch);
      if (existing < 0) {
        TreeNode child{};
        child.ch = candidate.ch;
        child.pinyin = candidate.pinyin;
        child.prob = candidate.prob;
        child.cum = nodes_[parent].cum * candidate.prob;
        child.parent = parent;
        child.depth = nodes_[parent].depth + 1;
        child.terminal = IsBranchStop(child.ch);
        nodes_.push_back(child);
        nodes_[parent].children.push_back(
            static_cast<int>(nodes_.size() - 1));
      } else {
        nodes_[existing].pinyin = candidate.pinyin;
        nodes_[existing].prob = candidate.prob;
        nodes_[existing].cum = nodes_[parent].cum * candidate.prob;
      }
    }
    nodes_[parent].children_generated = true;
  }

 private:
  // [TREE-010 PINYIN]
  bool PinyinMatchesLocked(int child, const std::wstring& filter) const {
    if (filter.empty())
      return true;
    const std::wstring& pinyin = nodes_[child].pinyin;
    if (pinyin.size() < filter.size())
      return false;
    return pinyin.compare(0, filter.size(), filter) == 0;
  }

  int FindChildLocked(int parent, wchar_t ch) const {
    if (parent < 0 || parent >= static_cast<int>(nodes_.size()))
      return -1;
    for (int child : nodes_[parent].children) {
      if (nodes_[child].ch == ch)
        return child;
    }
    return -1;
  }

  mutable std::mutex mutex_;
  std::wstring base_prefix_;
  std::vector<TreeNode> nodes_;
};

}  // namespace ghost
}  // namespace weasel
