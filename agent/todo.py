"""待办事项存储：TodoStore。

被主 Agent 的 `update_todos` @tool 函数使用。
旧版 `UpdateTodosTool`（继承 Tool 基类）已随归档移除。
"""
from __future__ import annotations


_VALID_STATUS = ("pending", "in_progress", "completed")
_STATUS_ICON = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def _render(todos: list[dict]) -> str:
    if not todos:
        return "(当前无待办事项)"
    lines = []
    for t in todos:
        icon = _STATUS_ICON.get(t.get("status", "pending"), "[?]")
        lines.append(f"  {icon} {t.get('id')}. {t.get('content', '')}")
    return "\n".join(lines)


class TodoStore:
    """待办事项存储管理器。

    跨用户回合存活的待办列表。不进入 history，compactor 不会丢失。
    """

    def __init__(self):
        self.todos: list[dict] = []

    def update(self, items: list[dict]) -> str:
        cleaned: list[dict] = []
        for i, t in enumerate(items, start=1):
            content = (t.get("content") or "").strip()
            if not content:
                continue
            status = t.get("status", "pending")
            if status not in _VALID_STATUS:
                status = "pending"
            cleaned.append({
                "id": t.get("id", i),
                "content": content,
                "status": status,
            })

        in_progress_count = sum(1 for t in cleaned if t["status"] == "in_progress")
        if in_progress_count > 1:
            return "Error: 同一时间只能有一个 in_progress 任务，请重新规划。"

        self.todos = cleaned
        print("\n[计划已更新]")
        print(_render(self.todos))
        print()

        completed = sum(1 for t in self.todos if t["status"] == "completed")
        pending = sum(1 for t in self.todos if t["status"] == "pending")
        summary = (
            f"todos updated: total={len(self.todos)}, completed={completed}, "
            f"in_progress={in_progress_count}, pending={pending}"
        )
        return summary + "\n\n当前列表：\n" + _render(self.todos)

    def render(self) -> str:
        return _render(self.todos)
