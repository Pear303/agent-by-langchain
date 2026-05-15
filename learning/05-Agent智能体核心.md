# 第5章：Agent — 智能体核心

> **学习目标**：深入理解 Agent 的工作原理、AgentExecutor 的执行流程、回调系统

---

## 5.1 Agent 的本质：带循环的 Chain

回顾第3章，Chain 是固定的处理流程：

```
输入 → Prompt → LLM → 输出
```

而 Agent 是**带循环的 Chain**：

```
输入 → Prompt → LLM → 有工具调用? → 执行工具 → 回到 Prompt
                  ↓ 否
                 输出
```

> **Agent = Chain + 循环决策能力**

LangChain 用 `AgentExecutor` 来实现这个循环。创建 Agent 分为两步：

```python
# 步骤1：创建 Agent（只是一个 Runnable）
agent = create_tool_calling_agent(llm, tools, prompt)
# 等价于: agent = prompt | llm.bind_tools(tools)

# 步骤2：包装成 AgentExecutor（加入循环）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=50,    # 防止无限循环
    handle_parsing_errors=True,
)
```

---

## 5.2 AgentExecutor 内部执行流程

当调用 `executor.invoke({"input": "...", "chat_history": [...]})` 时：

```
┌─────────────────────────────────────────────────────────────────┐
│ AgentExecutor.invoke({input, chat_history})                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第1步: 准备 Prompt 输入                                          │
│   把 {chat_history} 和 {input} 填入 ChatPromptTemplate           │
│   agent_scratchpad 初始为空                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第2步: 调用 LLM                                                  │
│   agent.invoke({"input": ..., "chat_history": [...],             │
│                 "agent_scratchpad": [...]})                       │
│                                                                  │
│   LLM 返回 → AIMessage                                           │
│     可能包含: tool_calls[]（决定调用工具）                         │
│     或纯文本（最终回答）                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              有 tool_calls       纯文本回复
                    │                 │
                    ▼                 ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ 第3步: 格式检查   │  │ 第5步: 返回结果   │
        │ 验证 tool_calls   │  │ {"output": "..."} │
        │ 是否合法          │  └──────────────────┘
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ 第4步: 执行工具   │
        │ 按顺序/并发执行    │
        │ 收集 tool_result  │
        │                   │
        │ 追加到             │
        │ agent_scratchpad  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ 回到第1步(下一轮)  │
        │ agent_scratchpad  │
        │ 包含了上一轮的     │
        │ 工具调用+结果      │
        └──────────────────┘
```

### 最大迭代限制

```python
max_iterations=50
```

如果第50轮后 LLM 还在调工具，`AgentExecutor` 会强制停止并返回当前结果。这是防止 Agent 无限循环的安全网。

> 本项目设置 50 轮，普通任务通常在 3-8 轮内完成。

---

## 5.3 create_tool_calling_agent 的实现原理

```python
def create_tool_calling_agent(llm, tools, prompt):
    """内部实现（简化版）"""
    # 1. 将工具绑定到 LLM
    #    bind_tools 会让 LLM 知道有哪些工具可用
    llm_with_tools = llm.bind_tools(tools)

    # 2. 返回一个 Runnable
    #    prompt 的输出 → llm_with_tools 的输入
    return prompt | llm_with_tools
```

`bind_tools` 是关键步骤。它做了两件事：

```python
# bind_tools 内部行为（伪代码）
llm.bind_tools(tools)

# 1. 生成 OpenAI 兼容的工具定义
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema.schema() if has tool.args_schema else inferred_schema,
        }
    }
    for tool in tools
]

# 2. 在每次 API 调用时自动附加 tools 参数
# 实际发送给 API 的请求:
{
    "model": "deepseek-v4-flash",
    "messages": [...],
    "tools": tool_definitions,       # ← 自动附加
    "tool_choice": "auto",           # ← LLM 自主决定
}
```

> `bind_tools` 不修改 LLM 本身，它只是让 LLM 在调用时知道"有哪些工具可用"。

---

## 5.4 agent_scratchpad 的秘密

`agent_scratchpad` 是 Agent 能够"记住上一轮干了什么"的关键。

### 它的格式

```python
# agent_scratchpad 的内容是一系列消息
[
    # 第1轮的工具调用
    AIMessage(content="", tool_calls=[
        {"name": "web_fetch", "args": {"url": "https://..."}, "id": "call_1"}
    ]),
    # 第1轮的工具结果
    ToolMessage(content="<网页内容>", tool_call_id="call_1"),

    # 第2轮的工具调用
    AIMessage(content="", tool_calls=[
        {"name": "write_file", "args": {...}, "id": "call_2"}
    ]),
    # 第2轮的工具结果
    ToolMessage(content="文件已写入", tool_call_id="call_2"),
]
```

### 为什么叫 scratchpad（草稿纸）？

因为 LLM 每次调用都是"从头推理"的，需要使用 `agent_scratchpad` 来看到之前已经做过什么。就像你在草稿纸上写计算过程一样。

**关键设计**：`agent_scratchpad` 在 ChatPromptTemplate 中用 `placeholder` 类型，表示它是一系列消息而不是普通文本。

---

## 5.5 回调系统（Callbacks）

LangChain 提供了一套完整的事件钩子——`BaseCallbackHandler`：

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"开始调用 LLM...")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM 调用完成")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"开始执行工具: {serialized['name']}")

    def on_tool_end(self, output, **kwargs):
        print(f"工具执行完成")

    def on_agent_finish(self, finish, **kwargs):
        print(f"Agent 执行结束")
```

### 本项目中的 TokenCallback

`agent/lc_agent.py` 中的 `TokenCallback` 追踪每次 LLM 调用的 token 用量：

```python
class TokenCallback(BaseCallbackHandler):
    def __init__(self, tracker, model):
        self._tracker = tracker
        self._model = model

    def on_llm_end(self, response, **kwargs):
        """每次 LLM 调用结束时触发"""
        usage = response.generations[0][0].message.usage_metadata
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)

        self._tracker.record_raw(
            self._model, input_tokens, output_tokens,
            input_tokens + output_tokens
        )
```

### 回调如何挂载？

```python
# 方法1：在 AgentExecutor 层挂载（本项目方式）
executor = AgentExecutor(
    ...,
    callbacks=[TokenCallback(self.token_tracker, self.model)],
)

# 方法2：在 LLM 层挂载
llm = ChatOpenAI(..., callbacks=[my_callback])

# 方法3：在 invoke 时临时挂载
chain.invoke({"input": "你好"}, config={"callbacks": [my_callback]})
```

> 回调是**非侵入式**的——不改变执行逻辑，只监听事件。

---

## 5.6 Agent 的消息类型

Agent 内部流通的是 LangChain 的标准消息类型：

```python
from langchain_core.messages import (
    HumanMessage,      # 用户消息
    AIMessage,         # AI 回复（可以包含 tool_calls）
    SystemMessage,     # 系统提示词
    ToolMessage,       # 工具执行结果
    BaseMessage,       # 所有消息的基类
)
```

消息结构示例：

```python
# 用户消息
HumanMessage(content="帮我查一下北京的天气")

# AI 消息（带工具调用）
AIMessage(
    content="好的，我查一下天气。",
    tool_calls=[
        {
            "name": "web_fetch",
            "args": {"url": "https://wttr.in/Beijing"},
            "id": "call_abc123",
        }
    ]
)

# 工具消息
ToolMessage(
    content="北京: 晴, 27°C, 湿度40%",
    tool_call_id="call_abc123",
)
```

---

## 5.7 处理解析错误

LLM 偶尔会返回格式错误的 tool_calls。`handle_parsing_errors=True` 让 AgentExecutor 自动处理：

```python
executor = AgentExecutor(
    ...,
    handle_parsing_errors=True,
    # 内部行为：捕获解析异常，把错误信息发给 LLM 让它重试
)

# 相当于帮你写了：
try:
    parsed = parse_llm_output(llm_output)
except Exception as e:
    # 把错误消息返回给 LLM，让它修正
    return f"格式错误，请修正: {e}"
```

---

## 5.8 从入口到循环的完整调用链

```python
# agent.py（入口）
agent = LCAgent(model="deepseek-v4-flash")
agent.run()
```

```python
# lc_agent.py run() 方法
def run(self):
    while True:  # 多轮对话循环（不是 Agent 循环）
        user_input = input("You🫅 : ")
        if user_input == "exit": break

        # ↓ 进入 Agent 内部循环
        result = self.executor.invoke({
            "input": user_input,
            "chat_history": self.memory_store.messages,
        })
        # ↑ AgentExecutor 内部自动迭代直到完成

        reply = result["output"]
        self.memory_store.append_history("user", user_input)
        self.memory_store.append_history("assistant", reply)

        print(f"智能助手🧰: {reply}")
```

注意两个"循环"的区别：

| 循环 | 所在位置 | 作用 | 终止条件 |
|------|---------|------|---------|
| **对话循环** | `agent.py` / `LCAgent.run()` | 反复接收用户输入 | 用户输入 exit |
| **Agent 循环** | `AgentExecutor` 内部 | 推理-行动-观察 | LLM 返回纯文本 / 达到 max_iterations |

---

## 5.9 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ Agent = Chain + 循环决策能力               │
│  ☐ AgentExecutor 管理 ReAct 循环              │
│  ☐ agent_scratchpad 记录中间步骤              │
│  ☐ handle_parsing_errors 处理 LLM 格式错误   │
│  ☐ Callbacks 非侵入式监听事件                 │
│  ☐ 对话循环 vs Agent 循环是两个不同层次        │
└──────────────────────────────────────────────┘
```

### 思考题

1. 如果一个 Agent 在 `max_iterations` 轮后仍然没有完成，你会怎么处理？
2. `agent_scratchpad` 如果超过 token 限制怎么办？
3. 能否同时使用多个 `BaseCallbackHandler`？

---

**[下一章：Memory — 记忆系统 →](06-记忆系统Memory.md)**
