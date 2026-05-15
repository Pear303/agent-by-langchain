# 第3章：Chain — 链式调用

> **学习目标**：掌握 LangChain 的链式调用机制，理解 `Runnable` 接口和管道操作

---

## 3.1 什么是 Chain？

Chain（链）是把多个处理步骤**串联**成一个执行流程。

### 生活中的类比

```
汽车装配链:
  车架 → 装发动机 → 装轮胎 → 装座椅 → 喷漆 → 成品车
```

Chain 在 LangChain 中也是同样的道理：

```
用户输入 → PromptTemplate格式化 → LLM处理 → 输出解析 → 最终结果
```

### 最简单的 Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate.from_template("请用{language}介绍{topic}")

# 用管道操作符 | 创建链
chain = prompt | llm

# 执行链
result = chain.invoke({"language": "中文", "topic": "香菇"})
```

> **核心思想**：`prompt | llm` 的意思是"先把输入传给 prompt 格式化，再把结果传给 llm"。

---

## 3.2 Runnable 接口深入

前一章提到所有组件都实现了 `Runnable` 接口。接口的完整定义如下：

```python
class Runnable[Input, Output]:
    """可运行组件的基础接口"""

    def invoke(self, input: Input) -> Output:
        """同步调用"""

    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""

    def batch(self, inputs: list[Input]) -> list[Output]:
        """批量调用"""

    def stream(self, input: Input) -> Iterator[Output]:
        """流式输出"""

    # 组合方法
    def pipe(self, other: Runnable) -> Runnable:
        """等同于 | 操作符"""

    def bind(self, **kwargs) -> Runnable:
        """绑定额外参数（如 tools、response_format）"""

    def with_config(self, config: dict) -> Runnable:
        """绑定配置（如回调）"""
```

### invoke 的数据流

```python
chain = prompt | llm

# prompt.invoke({"language": "中文", "topic": "香菇"})
# → PromptValue（包含格式化后的文本）

# llm.invoke("请用中文介绍香菇")
# → AIMessage(content="香菇是一种...\n\n香菇的学名是 Lentinula edodes...")

# chain.invoke({"language": "中文", "topic": "香菇"})
# → AIMessage(content="香菇是一种...")
```

管道操作符 `|` 自动处理类型转换：第一个 Runnable 的输出类型匹配第二个的输入类型。

---

## 3.3 构建更复杂的链

### 链 + 输出解析器

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"language": "中文", "topic": "香菇"})
# result 是 str 类型（纯文本），而不是 AIMessage
# "香菇是一种..."
```

### 链 + 多个参数

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，请用{style}的风格回答。"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "role": "营养学专家",
    "style": "通俗易懂",
    "question": "香菇有什么营养价值？"
})
```

### 链 + RunnablePassthrough（传入额外上下文）

`RunnablePassthrough` 允许你在链中传递额外的数据：

```python
from langchain_core.runnables import RunnablePassthrough

# 先处理输入，再传给 prompt
chain = (
    RunnablePassthrough.assign(
        topic=lambda x: f"{x['topic']}的详细介绍"
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 3.4 本项目中的 Chain

查看 `agent/lc_agent.py`，Agent 内部也是用 Chain 来构建的：

```python
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义提示词模板（本质是一个 Chain）
self.prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),          # 系统提示词
    ("placeholder", "{chat_history}"),   # 历史对话（动态注入）
    ("human", "{input}"),               # 用户输入
    ("placeholder", "{agent_scratchpad}"),  # Agent中间步骤
])

# 2. 创建 Agent Chain
# create_tool_calling_agent() 内部实现:
#   bind_tools: 将工具绑定到 LLM，让 LLM 知道可以调用什么工具
#   返回一个 Runnable: prompt | llm.bind_tools(tools)
agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

# 3. 包装成 AgentExecutor（自动管理 ReAct 循环）
self.executor = AgentExecutor(
    agent=agent,  # agent 是一个 Runnable   ← 核心！Runnable 接口统一
    tools=self.tools,
    verbose=True,
    max_iterations=50,
    handle_parsing_errors=True,
    callbacks=[TokenCallback(self.token_tracker, self.model)],
)
```

### create_tool_calling_agent 内部发生了什么？

这个函数是 LangChain 的核心工具——它把 **prompt + llm + tools** 组合成一个 Runnable：

```
create_tool_calling_agent(llm, tools, prompt)
    ↓
返回一个 RunnableChain 内部等价于:
    prompt | llm.bind_tools(tools)
    ↓
AgentExecutor 包装后:
    executor.invoke({"input": ..., "chat_history": [...]})
    ↓
内部自动:
    第1轮: prompt → llm思考（可能有 tool_calls）→ 执行工具 → 观察结果
    第2轮: (同上，带着上一轮结果) → ... → 直到 LLM 返回纯文本
```

---

## 3.5 Runnable 的其他组合方式

### 并行执行：RunnableParallel

```python
from langchain_core.runnables import RunnableParallel

# 两个 chain 并行执行
chain1 = PromptTemplate.from_template("用中文介绍{topic}") | llm
chain2 = PromptTemplate.from_template("用英文介绍{topic}") | llm

parallel_chain = RunnableParallel(
    中文=chain1,
    英文=chain2,
)

result = parallel_chain.invoke({"topic": "香菇"})
# {
#   "中文": AIMessage(content="香菇是一种..."),
#   "英文": AIMessage(content="Shiitake is a..."),
# }
```

### 条件路由：RunnableBranch

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x["input"]) > 100, long_chain),   # 长输入 → 用详细链
    (lambda x: len(x["input"]) < 10, short_chain),    # 短输入 → 用简洁链
    default_chain,                                     # 其他 → 默认链
)
```

---

## 3.6 Chain 在项目各处的体现

虽然本项目的核心是 Agent（动态决策），但 Chain 的概念贯穿始终：

| 位置 | Chain 的体现 |
|------|-------------|
| `lc_agent.py` | `prompt \| llm.bind_tools(tools)` 构成 Agent |
| `lc_tools.py` 中的 `dispatch_subagent` | 内部创建 `ChatPromptTemplate \| ParallelAgentExecutor` |
| `compactor.py` | `system_prompt \| openai_client.chat.completions.create` 压缩历史 |
| `context.py` | 多个模板片段拼接 → 最终 system prompt |

> Agent 是动态的 Chain，Chain 是固定的 Agent。记住这个辩证关系。

---

## 3.7 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ Chain = 多个 Runnable 串联                 │
│  ☐ | 操作符组合组件                           │
│  ☐ invoke/ainvoke/batch/stream 统一接口       │
│  ☐ create_tool_calling_agent 返回 Runnable    │
│  ☐ AgentExecutor 让 Runnable 具备循环能力      │
│  ☐ RunnableParallel 并行、RunnableBranch 分支  │
└──────────────────────────────────────────────┘
```

### 思考题

1. 如果不使用 `|` 操作符，怎么手动实现 `chain.invoke()` 的效果？
2. `RunnablePassthrough` 在什么时候特别有用？
3. 为什么 `create_tool_calling_agent` 不直接把循环逻辑也封装进去，而要交给 `AgentExecutor`？

---

**[下一章：Tool — 工具系统 →](04-工具系统Tool.md)**
