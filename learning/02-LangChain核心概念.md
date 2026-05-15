# 第2章：LangChain 核心概念

> **学习目标**：掌握 LangChain 的四大核心抽象，理解它们之间的关系

---

## 2.1 LangChain 的"乐高哲学"

LangChain 的设计思想是 **"像搭乐高一样搭 AI 应用"**：

```
┌─────────────────────────────────────────┐
│           LangChain 生态系统              │
│                                         │
│  ┌────────┐  ┌────────┐  ┌──────────┐  │
│  │ Models  │  │ Prompts │  │ Memory   │  │
│  │ (模型)  │  │ (提示词) │  │ (记忆)   │  │
│  └────┬───┘  └────┬───┘  └────┬─────┘  │
│       │           │           │         │
│  ┌────┴───────────┴───────────┴──────┐  │
│  │            Chains (链)            │  │
│  └────────────────┬──────────────────┘  │
│                   │                     │
│  ┌────────────────┴──────────────────┐  │
│  │         Agents (智能体)           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

每个组件都是独立的"乐高块"，可以自由组合。

---

## 2.2 四大核心抽象

### ① Model I/O — 模型输入输出

**作用**：统一所有 LLM 的调用接口

不管你用的是 OpenAI、DeepSeek、Claude 还是本地模型，LangChain 都提供一致的接口：

```python
# 不管是哪个模型，调用方式都一样
from langchain_openai import ChatOpenAI

# OpenAI
llm = ChatOpenAI(model="gpt-4")

# DeepSeek（本项目实际使用）
llm = DeepSeekChatOpenAI(model="deepseek-v4-flash")

# 调用方式完全一致
response = llm.invoke("你好")
```

本项目中的实际使用（`agent/lc_agent.py`）：

```python
from langchain_openai import ChatOpenAI

class DeepSeekChatOpenAI(ChatOpenAI):
    """继承自 ChatOpenAI，兼容 DeepSeek API"""
    ...

self.llm = DeepSeekChatOpenAI(
    model=model,
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
```

> LangChain 用 `ChatOpenAI` 封装了 OpenAI 兼容的 API 协议，DeepSeek 也使用相同的协议，所以直接继承即可。

### ② Prompts — 提示词管理

**作用**：结构化地构建给 LLM 的提示词

#### 最简单的 PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

template = "请用{language}写一段{code_type}代码"
prompt = PromptTemplate.from_template(template)

# 格式化
result = prompt.format(language="Python", code_type="排序算法")
# → "请用Python写一段排序算法代码"
```

#### ChatPromptTemplate（本项目使用）

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的AI助手，请用{style}风格回复。"),
    ("placeholder", "{chat_history}"),  # ← 对话历史占位
    ("human", "{input}"),               # ← 用户输入占位
    ("placeholder", "{agent_scratchpad}"),  # ← Agent中间步骤占位
])
```

模板中的**三种占位符**：

| 占位符 | 运行时注入什么 | 来源 |
|--------|--------------|------|
| `{chat_history}` | 历史消息列表 | 从 memory 加载 |
| `{input}` | 用户当前输入 | `executor.invoke({"input": ...})` |
| `{agent_scratchpad}` | 中间步骤（工具调用+结果） | LangChain 内部自动维护 |

> `placeholder` 是特殊类型，它注入的是一个**消息列表**而不是普通文本，这让 Agent 能"记住"之前的工具调用历史。

### ③ Memory — 记忆

**作用**：让 Agent 记住过去的信息

LangChain 的 `BaseChatMessageHistory` 定义了记忆的标准接口：

```python
from langchain_core.chat_history import BaseChatMessageHistory

# 标准接口
class BaseChatMessageHistory:
    messages: list[BaseMessage]  # 存储的消息

    def add_messages(self, messages): ...   # 添加消息
    def clear(self): ...                    # 清空记忆
```

本项目中的 `MemoryStore` 实现了这个接口（`agent/memory.py`）：

```python
class MemoryStore(BaseChatMessageHistory):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages):
        for msg in messages:
            # 将 LangChain 消息序列化到 JSONL 文件
            role = _TYPE_TO_JSONL_ROLE.get(msg.type, "unknown")
            extra = getattr(msg, "additional_kwargs", None) or None
            self.append_history(role, msg.content, additional_kwargs=extra)
```

> 第6章会详细讲解记忆系统的三种层次。

### ④ Chains / Agents — 链与智能体

**Chain** 是**固定的**处理流程：

```
输入 → 步骤A → 步骤B → 步骤C → 输出
```

**Agent** 是**动态的**处理流程（ReAct 循环）：

```
输入 → LLM思考 → 选择工具 → 执行 → 观察 → 再思考 → ... → 输出
```

| | Chain | Agent |
|--|-------|-------|
| 流程 | 固定不变 | 动态决策 |
| 灵活性 | 低 | 高 |
| 适用场景 | 确定的转换流程 | 需要决策的任务 |
| 是否需要工具 | 可选 | 必须 |

> 链条定流程，Agent 定决策。两者不是互斥的——Agent 内部也是用 Chain 来组织提示词的。

---

## 2.3 Runnable 接口：LangChain 的"统一协议"

LangChain 中几乎所有组件都实现了 `Runnable` 接口。这是 LangChain 最精妙的设计之一。

### 三个核心方法

```python
class Runnable[Input, Output]:
    def invoke(self, input: Input) -> Output:
        """同步调用"""

    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""

    def batch(self, inputs: list[Input]) -> list[Output]:
        """批量调用"""
```

### 为什么重要？

因为**所有东西都是 Runnable**：

```python
llm = ChatOpenAI(...)          # 是 Runnable
prompt = ChatPromptTemplate()  # 是 Runnable
chain = prompt | llm           # 也是 Runnable
agent = create_tool_calling_agent(...)  # 是 Runnable
executor = AgentExecutor(...)  # 也是 Runnable

# 所以它们都能用同样的方式调用
llm.invoke("你好")
chain.invoke({"input": "你好"})
executor.invoke({"input": "你好", "chat_history": []})
```

### 管道操作符 `|`

LangChain 的 `|` 操作符（类似 Unix 管道）是组合组件的核心方式：

```python
# 传统写法
formatted = prompt.format(input="你好")
response = llm.invoke(formatted)

# 管道写法（函数式编程）
chain = prompt | llm
response = chain.invoke({"input": "你好"})
# prompt 的输出 → 自动成为 llm 的输入
```

本项目中的使用：

```python
# agent/lc_agent.py
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
# 内部等价于: prompt | llm（加上工具绑定）

executor = AgentExecutor(
    agent=agent,
    tools=self.tools,
    verbose=True,
    max_iterations=50,
)
# AgentExecutor.invoke() 将自动管理 ReAct 循环
```

---

## 2.4 LangChain 生态全景图

```
┌──────────────────────────────────────────────────────────────────┐
│                       LangChain 生态                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  langchain-core ◄── 核心抽象                                     │
│    ├── Runnable 接口                                             │
│    ├── BaseMessage（消息类型）                                    │
│    ├── PromptTemplate                                            │
│    ├── BaseTool / @tool                                          │
│    └── BaseChatMessageHistory                                    │
│                                                                  │
│  langchain-openai ◄── LLM 提供商封装                             │
│    ├── ChatOpenAI（兼容 DeepSeek）                                │
│    └── OpenAI Embeddings                                         │
│                                                                  │
│  langchain-community ◄── 社区贡献的工具/模型                      │
│                                                                  │
│  langgraph ◄── 高级 Agent 编排（多 Agent 流程控制）               │
│                                                                  │
│  本项目使用的子集:                                               │
│    langchain-core + langchain-openai + langgraph                 │
│    + 自己实现的 MemoryStore / SubagentRegistry                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**不需要学整个生态**。本项目只用了 LangChain 的一个子集：
- `langchain-core`：核心抽象（消息、提示词、工具、Runnable）
- `langchain-openai`：DeepSeek LLM 封装
- `langgraph`：高级 Agent 能力（可选）

---

## 2.5 从项目看整体架构

```python
# agent/lc_agent.py 中的 __init__ —— 完整装配流程

def __init__(self, model="deepseek-v4-flash", max_iterations=50):
    # ① LLM
    self.llm = DeepSeekChatOpenAI(model=model, ...)

    # ② 工具
    self.tools = [read_file, write_file, run_command, web_fetch, ...]

    # ③ 记忆
    self.memory_store = MemoryStore(memory_dir=...)

    # ④ 提示词
    self.prompt = ChatPromptTemplate.from_messages([...])

    # ⑤ Agent
    agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
    self.executor = AgentExecutor(agent=agent, tools=self.tools, ...)
```

这个 `__init__` 就是 LangChain 四大核心的**完整装配**。第3-6章会逐一深入每个组件。

---

## 2.6 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ LangChain 四大核心：Model/Prompt/Memory/   │
│     Chain-Agent                              │
│  ☐ Runnable 接口：invoke/ainvoke/batch        │
│  ☐ 管道操作符 | 组合组件                      │
│  ☐ AgentExecutor 封装 ReAct 循环              │
│  ☐ 本项目是 LangChain 子集的完整实战           │
└──────────────────────────────────────────────┘
```

### 思考题

1. `Runnable` 接口为什么要同时提供 `invoke` 和 `ainvoke`？
2. 如果不使用 LangChain，手写一个 `AgentExecutor` 需要处理哪些问题？
3. 在 `ChatPromptTemplate` 中，`placeholder` 和普通的 `system`/`human` 占位符有什么区别？

---

**[下一章：Chain — 链式调用 →](03-Chain链式调用.md)**
