# 第7章：高级 Agent — 子代理与任务规划

> **学习目标**：掌握多 Agent 协作模式、任务规划机制、安全约束设计

---

## 7.1 为什么需要子代理？

单个 Agent 面对复杂任务时有三个问题：

| 问题 | 表现 | 后果 |
|------|------|------|
| **上下文污染** | 所有信息都塞进同一个上下文 | 关键信息被淹没 |
| **工具争抢** | 10个工具挤在一起，LLM 容易选错 | 调用不合适的工具 |
| **串行执行** | 必须做完一步才能做下一步 | 效率低下 |

**子代理 = 分而治之**：

```
主 Agent: 只负责规划 + 协调（少数关键工具）
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 子Agent A     子Agent B     子Agent C
(专注搜索)    (专注分析)    (专注执行)
 工具: 只有    工具: 只有    工具: 只有
 web_fetch     read_file    write_file
```

> 每个子 Agent 的上下文是独立的——A 的搜索不会污染 B 的分析。

---

## 7.2 子代理的完整实现

### 7.2.1 SubagentSpec — 子代理规格定义

```python
# agent/subagents/spec.py
@dataclass(frozen=True)
class SubagentSpec:
    name: str              # 子代理名称
    description: str       # 描述（供主 Agent 决定用哪个）
    system_prompt: str     # 从模板文件加载的身份定义
    tool_names: tuple[str, ...]  # 工具白名单
    max_turns: int = 15    # 最大迭代次数
```

`frozen=True` 确保规格不可变——一旦定义，运行中不更改。

### 7.2.2 SubagentRegistry — 注册表

```python
# agent/subagents/registry.py
class SubagentRegistry:
    def __init__(self, templates_dir, skills_loader=None):
        self._specs: dict[str, SubagentSpec] = {}
        self._load_all(templates_dir, skills_loader)

    def get(self, name: str) -> SubagentSpec | None:
        return self._specs.get(name)

    def names(self, include_aliases=False) -> list[str]:
        return list(self._specs.keys())
```

5 种子代理的注册信息：

```python
_BUILTIN_SPECS = {
    "quick_helper": {
        "description": "快速助手。轻量只读，适合短命令。",
        "tool_names": ("run_command", "read_file", "glob_tool", "grep_tool"),
        "max_turns": 8,
    },
    "doc_analyzer": {
        "description": "文档分析器。只读，适合阅读代码、查阅文档。",
        "tool_names": ("load_skill", "read_file", "glob_tool", "grep_tool"),
        "max_turns": 12,
    },
    "web_researcher": {
        "description": "网络研究员。只读查访，适合抓网页、查资料。",
        "tool_names": ("run_command", "web_fetch", "load_skill",
                       "read_file", "glob_tool", "grep_tool"),
        "max_turns": 15,
    },
    "validator": {
        "description": "校验员。只读核验，适合盘点文件、检查遗漏。",
        "tool_names": ("run_command", "read_file", "glob_tool", "grep_tool"),
        "max_turns": 12,
    },
    "engine_executor": {
        "description": "工程执行器。可读写可执行，适合修改文件。",
        "tool_names": ("run_command", "web_fetch", "load_skill",
                       "read_file", "write_file", "edit_file",
                       "glob_tool", "grep_tool"),
        "max_turns": 20,
    },
}
```

### 7.2.3 身份模板

每个子代理有一个 Markdown 身份模板（`templates/subagents/*.md`）：

```markdown
<!-- templates/subagents/web_researcher.md -->
你是网络研究员，负责外出查访、搜罗线索、比对资料。

## 身份与口吻
- 你直接向上级汇报，语气干练，不夸大
- 专注于信息收集和分析

## 行为约束
- 你的职责是查询和调研，不是改动
- 可以抓网页、读文件、搜索内容、运行只读查询命令
- 你不能再派遣其他子代理（防止递归！）
- 如果发现需要其他子代理协助，直接向上级汇报请求协调

## 汇报格式
最后用一段简短中文汇报：
1. 关键发现，3-7 条以内
2. 引用来源：URL / 文件路径 / 行号
3. 若证据不足，明确说明缺什么
```

### 7.2.4 dispatch_subagent — 派遣子代理

这是连接"主 Agent"和"子 Agent"的桥梁，**一个 @tool 函数**：

```python
# lc_tools.py 中的 dispatch_subagent
@tool
def dispatch_subagent(agent_type: str, task: str) -> str:
    """派遣子代理执行独立任务。

    Args:
        agent_type: 子代理类型
        task: 任务描述
    """
    spec = _subagent_registry.get(agent_type)

    # 1. 按白名单筛选工具
    tools = [
        _SUBAGENT_TOOL_MAP[name]
        for name in spec.tool_names
        if name in _SUBAGENT_TOOL_MAP
    ]

    # 2. 子代理有自己的独立 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", spec.system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 3. 子代理有自己的独立 AgentExecutor
    agent = create_tool_calling_agent(_llm_ref, tools, prompt)
    executor = ParallelAgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=spec.max_turns,
        handle_parsing_errors=True,
    )

    # 4. 独立上下文：chat_history=[] 空列表开始
    print(f"\n[派遣子代理 · {agent_type}]: {task[:80]}")
    result = executor.invoke({
        "input": task,
        "chat_history": [],  # ← 独立上下文！
    })

    # 5. 只回传结果总结
    final = result["output"]
    print(f"[子代理汇报]: {final[:200]}")
    return final
```

---

## 7.3 关键设计：上下文隔离

```
主 Agent 的上下文:
┌──────────────────────────────────────────────┐
│ system_prompt（含 MEMORY.md）                  │
│ chat_history（全部历史对话）                    │
│ human（当前用户输入）                           │
│ agent_scratchpad（上次工具调用结果）             │
│  → "派遣 web_researcher 搜索香菇资料"            │
│  → "得到结果: 香菇是一种真菌..."                │
└──────┬───────────────────────────────────────┘
       │ dispatch_subagent
       ▼
子 Agent 的上下文（完全独立）:
┌──────────────────────────────────────────────┐
│ system_prompt（子代理身份定义）                 │
│ chat_history = []（从零开始！）                │
│ human = "搜索百度百科关于香菇的资料"            │
│ agent_scratchpad（子代理自己的工具调用记录）    │
│  → web_fetch("https://baike.baidu.com/...")   │
│  → "香菇属于真菌界..."                        │
│  → "总结完毕，返回给主 Agent"                  │
└──────────────────────────────────────────────┘
```

**好处**：
- 子 Agent 调用的所有工具、所有试错，都不会污染主 Agent 的上下文
- 主 Agent 只看到一句总结（`result["output"]`）
- 如果子 Agent 出错重试，父 Agent 完全不知情

---

## 7.4 任务规划（Todolist）

### 7.4.1 为什么需要任务规划？

```
用户: "搜索3个以上网页，整理香菇资料，做成HTML文件"

没有规划时:
  Agent 凭感觉做事，可能先写 HTML 再搜索（做无用功）

有规划时:
  Agent 先思考步骤:
    1. 搜索3个以上网页（in_progress）
    2. 整理总结资料（pending）
    3. 制作成HTML文件（pending）
  然后按步骤执行，每完成一步就更新状态。
```

### 7.4.2 update_todos 工具

```python
# lc_tools.py
@tool
def update_todos(todos: list[dict]) -> str:
    """更新待办列表。每次传入完整数组（全量覆盖）。

    Args:
        todos: 待办列表，每项格式:
            {"id": int, "content": str, "status": "pending"|"in_progress"|"completed"}
    """
    ...
    return "todos updated: total=3, completed=0, in_progress=1, pending=2"
```

**约束**：同一时间只有一个任务可以是 `in_progress`。

### 7.4.3 TodoStore

```python
class TodoStore:
    def __init__(self):
        self.todos: list[dict] = []

    def update(self, todos: list[dict]) -> str:
        """全量覆盖更新"""
        self.todos = todos
        return self._format_summary()

    def current_todos_text(self) -> str:
        """格式化为文本（给 LLM 看）"""
        lines = []
        for t in self.todos:
            mark = {"pending": "[ ]", "in_progress": "[~]", "completed": "[✓]"}[t["status"]]
            lines.append(f"{mark} {t['content']}")
        return "\n".join(lines)
```

### 7.4.4 LLM 使用规划的方式

LLM 不需要特殊的"规划算法"——它只需要 `update_todos` 工具：

```
第1轮: LLM 思考
  "这是一个多步骤任务，我需要先创建计划"
  调用: update_todos(todos=[
    {"id": 1, "content": "搜索3个以上香菇资料网页", "status": "in_progress"},
    {"id": 2, "content": "整理总结资料", "status": "pending"},
    {"id": 3, "content": "生成HTML文件", "status": "pending"},
  ])

第2轮: 派遣子代理
  调用: dispatch_subagent(agent_type="web_researcher", task="...香菇百度百科...")
  调用: dispatch_subagent(agent_type="web_researcher", task="...香菇营养...")
  调用: dispatch_subagent(agent_type="web_researcher", task="...香菇栽培...")

第3轮: 更新规划
  调用: update_todos(todos=[
    {"id": 1, ..., "status": "completed"},
    {"id": 2, ..., "status": "in_progress"},
  ])
  ...
```

**规划的智能来自 LLM 自身的推理能力**，不是硬编码的。

---

## 7.5 安全约束体系

这是生产级 Agent 必须考虑的设计。本项目的安全约束分为三层：

### 第一层：工具白名单

```python
# dispatch_subagent 中
tools = [
    _SUBAGENT_TOOL_MAP[name]
    for name in spec.tool_names
    if name in _SUBAGENT_TOOL_MAP
]

# _SUBAGENT_TOOL_MAP 中显式省略了:
# - dispatch_subagent  → 防止递归派遣
# - update_todos      → 防止篡改主 Agent 的待办列表
```

### 第二层：最大迭代次数

```python
# 不同类型子代理的 max_turns 不同
"quick_helper":     8    # 快速任务限制更严
"doc_analyzer":    12
"web_researcher":  15
"validator":       12
"engine_executor": 20    # 工程任务可能步骤更多
```

### 第三层：身份模板约束

```markdown
<!-- 行为约束写在 prompt 中，是软约束但有效 -->
- 你不能再派遣其他子代理
- 你的职责是查询和调研，不是改动
- 可以抓网页、读文件、搜索内容、运行只读查询命令
```

### 整体安全架构

```
┌──────────────────────────────────────────────┐
│              主 Agent                         │
│  全部 10 个 @tool 函数                         │
│  max_iterations = 50                          │
└──────────────────┬───────────────────────────┘
                   │ dispatch_subagent
                   ▼
┌──────────────────────────────────────────────┐
│              子 Agent                         │
│  工具白名单（最多 8 个工具）                   │
│  ✗ dispatch_subagent — 防递归                │
│  ✗ update_todos — 防篡改待办                  │
│  max_iterations = 8~20                       │
│  chat_history = [] — 独立上下文               │
└──────────────────────────────────────────────┘
```

---

## 7.6 并发执行

### 子 Agent 间并发（主 Agent 层面）

当主 Agent 一次派遣多个子 Agent 时，`AgentExecutor` 按顺序处理：

```
LLM 返回:
  dispatch_subagent(A)  ← 先执行
  dispatch_subagent(B)  ← 再执行
  dispatch_subagent(C)  ← 最后执行
```

> 注意：这里是**顺序**的，每个子 Agent 完全运行完才到下一个。
> 这可能需要等待，但因为每个子 Agent 都有自己的 ReAct 循环，整体效率通常可以接受。

### 子 Agent 内工具级并发（子 Agent 层面）

`ParallelAgentExecutor` 能让子 Agent 内部的多个**只读工具**并发执行：

```
子 Agent 第 1 轮:
  LLM 返回:
    web_fetch(url1)  ← 只读 → 并发！
    web_fetch(url2)  ← 只读 → 并发！
    web_fetch(url3)  ← 只读 → 并发！

  结果: 三个网页同时抓取，等待全部完成后统一返回
```

```python
class ParallelAgentExecutor(AgentExecutor):
    _READ_ONLY_TOOLS = {
        "web_fetch", "read_file", "glob_tool", "grep_tool", "load_skill"
    }

    def _iter_next_step(self, ...):
        # 分组：只读工具放一起并发，写工具保持顺序
        for group in self._group_by_concurrency(actions):
            if len(group) == 1:
                yield self._perform_agent_action(group[0])
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    results = list(pool.map(_run, group))
                for step in results:
                    yield step
```

---

## 7.7 真实场景：香菇任务的完整流程

```
用户: "搜索三个以上网页，整理有关香菇的资料，总结制作成 html 文件"

第1轮: LLM 推理 + 创建 todolist
  → update_todos(todos=[...])

第2轮: 派遣 3 个 web_researcher 并发搜索
  → dispatch_subagent(agent_type="web_researcher",
       task="搜索百度百科关于香菇的分类、形态特征...")
  → dispatch_subagent(agent_type="web_researcher",
       task="搜索香菇的营养成分和药用价值...")
  → dispatch_subagent(agent_type="web_researcher",
       task="搜索香菇的栽培历史和烹饪用途...")

  每个子 Agent 内部:
    web_researcher #1:
      → web_fetch("https://baike.baidu.com/...")
      → 总结 → 返回 "香菇属于真菌界..."
    web_researcher #2:
      → web_fetch("https://...nutrition...")
      → 总结 → 返回 "香菇富含蛋白质..."
    web_researcher #3:
      → web_fetch("https://...cultivation...")
      → 总结 → 返回 "香菇栽培起源于中国..."

第3轮: LLM 收到三个子代理的结果
  → update_todos(推进任务)

第4轮: LLM 决定生成 HTML
  → write_file(path="finished-work/...", content="<!DOCTYPE html>...")

第5轮: 最终回复
  → "任务完成！已生成 HTML 文件..."
```

整个过程中，主 Agent 的上下文只有 **5 轮对话**（每轮约几百 token），而每个子 Agent 内部可能有 **3-10 轮**。如果没有子代理，这些全部会塞进主上下文。

---

## 7.8 设计模式总结

### 主-子代理模式

```
主 Agent: 指挥官，负责"决定做什么"
子 Agent: 执行者，负责"怎么去做"
```

**适用场景**：
- 任务涉及多个不同领域的知识
- 需要并行执行多个独立任务
- 需要严格的安全隔离

**不适用场景**：
- 简单的一两步任务（子代理反而增加开销）
- 需要紧密协作的任务（子代理之间不能直接通信）

### 规划-执行模式

```
第一步: LLM 思考 → 建立任务列表
第二步: 按顺序执行列表中的任务
第三步: 完成后更新列表
第四步: 如果所有任务完成 → 汇报
        否则 → 继续执行
```

### 白名单安全模式

```
不给所有权限，只给完成任务"恰好够用"的权限
```

---

## 7.9 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ 子代理 = 独立的 AgentExecutor              │
│  ☐ 上下文隔离：子代理 chat_history = []       │
│  ☐ SubagentSpec 定义子代理规格                │
│  ☐ SubagentRegistry 管理注册表               │
│  ☐ 安全约束三层：白名单+max_turns+模板约束     │
│  ☐ ParallelAgentExecutor 支持工具级并发       │
│  ☐ TodoStore 管理任务规划                     │
│  ☐ dispatch_subagent 是 @tool 的特殊实现      │
└──────────────────────────────────────────────┘
```

### 思考题

1. 如果子代理需要另一个子代理的结果才能继续，应该怎么设计？
2. 为什么 `dispatch_subagent` 不在 `_SUBAGENT_TOOL_MAP` 中？
3. 如何给子代理添加"技能加载"的能力？需要修改哪些代码？

---

**[下一章：综合实战 →](08-综合实战.md)**
