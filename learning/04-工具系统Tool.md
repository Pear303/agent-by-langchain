# 第4章：Tool — 工具系统

> **学习目标**：理解 LangChain 的工具系统，掌握 `@tool` 装饰器、参数验证、工具注册等核心概念

---

## 4.1 为什么 Agent 需要工具？

LLM 本身有**三大局限**：

| 局限 | 表现 | 工具如何解决 |
|------|------|-------------|
| **信息过时** | 训练数据截止到某个时间点 | 搜索工具获取最新信息 |
| **无法操作外部系统** | 只能生成文本 | 文件工具读写文件、命令工具执行代码 |
| **没有"感知"能力** | 不知道外部世界状态 | 读取文件、抓取网页来"感知" |

> **一句话**：工具是 Agent 的"感官"和"手脚"，让 Agent 从"只能说的聊天机器人"变成"能做事的智能体"。

---

## 4.2 @tool 装饰器：最简单的工具定义

LangChain 提供了 `@tool` 装饰器，让你用**最少的代码**定义工具：

```python
from langchain_core.tools import tool

@tool
def web_fetch(url: str, max_chars: int = 8000) -> str:
    """获取指定 URL 的网页内容。"""
    # 函数体：实现工具的逻辑
    ...
    return text[:max_chars]
```

### @tool 自动做了三件事

```python
# 你只需要写函数，@tool 自动生成：
{
  "name": "web_fetch",
  "description": "获取指定 URL 的网页内容。",
  "parameters": {
    "type": "object",
    "properties": {
      "url":    {"type": "string", "description": "url"},
      "max_chars": {"type": "integer", "default": 8000}
    },
    "required": ["url"]
  }
}
```

1. **函数名 → 工具名**：`web_fetch` → `"name": "web_fetch"`
2. **docstring → 工具描述**：LLM 通过描述来决定何时调用这个工具
3. **类型注解 → 参数 Schema**：`str`、`int`、`bool` 等自动映射为 JSON Schema

### 关键：LLM 如何知道调用哪个工具？

LLM 收到工具列表后，会根据每个工具的 `name` 和 `description` 决定调用哪个。所以**工具描述要写清楚**：

```python
@tool
def run_command(command: str) -> str:
    """在终端执行 shell 命令并返回输出结果。
    适合：运行脚本、编译代码、启动服务、文件操作。
    注意：慎重执行删除或修改命令。"""
    ...

@tool
def web_fetch(url: str) -> str:
    """获取指定 URL 的网页内容。
    适合：查资料、看文档、抓取在线信息。"""
    ...
```

> LLM 的"智能"很大程度取决于你给的描述是否清晰。

---

## 4.3 本项目的工具系统

查看 `agent/lc_tools.py`，主 Agent 注册了 10 个工具：

```python
# lc_agent.py __init__ 中
self.tools = [
    read_file,       # 读文件（支持 offset/limit 分页）
    write_file,      # 写文件（覆盖式）
    edit_file,       # 编辑文件（智能匹配缩进）
    run_command,     # 执行 shell 命令
    web_fetch,       # 抓取网页
    load_skill,      # 加载技能
    glob_tool,       # 文件搜索
    grep_tool,       # 内容搜索
    update_todos,    # 更新待办列表
    dispatch_subagent,  # 派遣子代理
]
```

### 4.3.1 读文件工具 (read_file)

```python
@tool
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """读取工作区文件的内容。支持 offset/limit 分页。
    Args:
        file_path: 文件路径（相对于工作区）
        offset: 起始行号，从 0 开始
        limit: 最多读取行数
    """
    full_path = _resolve_path(file_path)
    if not full_path.exists():
        return f"Error: file not found: {file_path}"
    lines = full_path.read_text(encoding="utf-8").splitlines()
    selected = lines[offset:offset + limit]
    return "\n".join(selected)
```

### 4.3.2 写文件工具 (write_file)

```python
@tool
def write_file(file_path: str, content: str) -> str:
    """写入文件（覆盖式）。会自动创建不存在的目录。"""
    full_path = _resolve_path(file_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {file_path}"
```

### 4.3.3 执行命令工具 (run_command)

```python
@tool
def run_command(command: str, workdir: str = "") -> str:
    """执行 shell 命令。workdir 为空时使用工作区根目录。"""
    ...

    result = subprocess.run(
        command, shell=True, cwd=cwd,
        capture_output=True, text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    return output[:3000]  # 截断防止 token 爆炸
```

### 4.3.4 搜索工具 (glob_tool / grep_tool)

```python
@tool
def glob_tool(pattern: str, path: str = "") -> str:
    """使用 glob 模式搜索文件。
    例如: '**/*.py' 找所有 Python 文件
    """
    ...

@tool
def grep_tool(pattern: str, include: str = "", path: str = "") -> str:
    """在文件内容中搜索指定模式。
    pattern: 正则表达式
    include: 文件匹配模式，如 '*.py'
    """
    ...
```

### 4.3.5 网页抓取工具 (web_fetch)

实现了编码检测和 gzip 解压的完整版：

```python
@tool
def web_fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    """获取指定 URL 的网页内容。
    Args:
        url: 要抓取的网页 URL
        extract_mode: 'text' 纯文本 或 'raw' 原始 HTML
        max_chars: 最多返回字符数
    """
    # 内部实现 _fetch() 包含：
    #   1. 自定义 User-Agent
    #   2. gzip 解压
    #   3. 编码检测（GBK→GB18030 转换）
    #   4. HTML 标签移除
    ...
```

---

## 4.4 dispatch_subagent — 最特殊的工具

`dispatch_subagent` 是本项目中最核心的 @tool，它的特殊之处在于：

**它内部创建了一个完整的 AgentExecutor（子代理）**。

```python
@tool
def dispatch_subagent(agent_type: str, task: str) -> str:
    """派遣一个子代理执行任务，子代理拥有独立上下文。

    Args:
        agent_type: 子代理类型
            - web_researcher: 网络研究员（只读查访）
            - engine_executor: 工程执行器（可读写）
            - doc_analyzer: 文档分析器（只读）
            - validator: 校验员（只读核验）
            - quick_helper: 快速助手（轻量只读）
        task: 给子代理的任务描述
    """
    spec = _subagent_registry.get(agent_type)

    # 子代理的工具从白名单中筛选
    tools = [
        _SUBAGENT_TOOL_MAP[name]
        for name in spec.tool_names
        if name in _SUBAGENT_TOOL_MAP
    ]

    # 子代理有自己独立的 prompt 和 executor
    prompt = ChatPromptTemplate.from_messages([
        ("system", spec.system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(_llm_ref, tools, prompt)
    executor = ParallelAgentExecutor(
        agent=agent, tools=tools,
        max_iterations=spec.max_turns,
    )

    result = executor.invoke({"input": task, "chat_history": []})
    return result["output"]   # ← 只回传总结，子代理内部历史不暴露
```

> 第7章会详细讲解这个机制。

---

## 4.5 工具白名单与安全约束

子代理不能使用所有工具——通过 `_SUBAGENT_TOOL_MAP` 实现白名单控制：

```python
# lc_tools.py — 全局工具映射表
_SUBAGENT_TOOL_MAP: dict[str, Any] = {
    "run_command": run_command,
    "web_fetch": web_fetch,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "glob_tool": glob_tool,
    "grep_tool": grep_tool,
    "load_skill": load_skill,
    # 注意: dispatch_subagent 不在其中！防止子代理递归派遣
    # 注意: update_todos 不在其中！防止子代理修改主 Agent 的 todolist
}
```

不同类型子代理的白名单：

| 子代理类型 | 可用工具 | 安全约束 |
|-----------|---------|---------|
| `quick_helper` | `run_command`, `read_file`, `glob`, `grep` | 只读为主 |
| `doc_analyzer` | `load_skill`, `read_file`, `glob`, `grep` | 完全只读 |
| `web_researcher` | `run_command`, `web_fetch`, `read_file`, ... | 可网络查询 |
| `validator` | `run_command`, `read_file`, `glob`, `grep` | 只读核验 |
| `engine_executor` | 以上全部 + `write_file`, `edit_file` | 可读写执行 |

> **设计原则**：最小权限。只给子代理完成任务所必需的权限。

---

## 4.6 工具级并发执行

本项目的 `ParallelAgentExecutor` 继承自 `AgentExecutor`，增加了工具级并发能力：

```python
class ParallelAgentExecutor(AgentExecutor):
    """AgentExecutor 子类，同一帧内的只读工具调用并发执行。"""

    def _iter_next_step(self, ...):
        # 同一帧内多个只读工具 → 并发执行
        for group in self._group_by_concurrency(actions):
            if len(group) == 1:
                yield self._perform_agent_action(group[0])
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    results = list(pool.map(_run, group))
                for step in results:
                    yield step
```

并发白名单（只读工具可并发）：

```python
_READ_ONLY_TOOLS = {
    "web_fetch",    # 并发抓取多个网页
    "read_file",    # 并发读取多个文件
    "glob_tool",    # 并发文件搜索
    "grep_tool",    # 并发内容搜索
    "load_skill",   # 并发加载技能
}
```

---

## 4.7 @tool 的进阶用法

### 自定义工具名

```python
from langchain_core.tools import tool

@tool(parse_docstring=True)
def my_tool(x: int, y: int) -> str:
    """计算两个数的和。
    Args:
        x: 第一个数
        y: 第二个数
    """
    return f"结果是{x + y}"
```

### 返回结构化数据

```python
from typing import TypedDict, List

class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str

@tool
def search(query: str) -> List[SearchResult]:
    """搜索并返回结构化结果"""
    ...
```

---

## 4.8 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ @tool 自动从函数生成工具定义（name/desc/    │
│     params）                                  │
│  ☐ 工具是 Agent 与外部世界的桥梁               │
│  ☐ 工具描述决定 LLM 会不会用它                 │
│  ☐ dispatch_subagent 是"元工具"——创建子 Agent │
│  ☐ 白名单机制实现安全约束                      │
│  ☐ 只读工具可并发执行（ParallelAgentExecutor） │
└──────────────────────────────────────────────┘
```

### 思考题

1. 为什么 `dispatch_subagent` 和 `update_todos` 不在子代理工具白名单中？
2. 如果让 `run_command` 也加入并发白名单，可能有什么风险？
3. `@tool` 自动生成的 Schema 和你手写 JSON Schema 相比，有什么优劣？

---

**[下一章：Agent — 智能体核心 →](05-Agent智能体核心.md)**
