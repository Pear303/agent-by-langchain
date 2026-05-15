# 第6章：Memory — 记忆系统

> **学习目标**：理解 Agent 的三层记忆架构，掌握记忆的持久化、压缩和跨会话保持

---

## 6.1 为什么 Agent 需要记忆？

让我们做一个思想实验。假设你是一个 Agent，你的 LLM 每次对话都是"重置"的：

```
第一轮对话：
  用户: "我的名字是张三"
  你: "好的张三！"
  ← 对话结束，LLM 被完全重置

第二轮对话：
  用户: "我叫什么名字？"
  你: "我不知道，你还没告诉过我..."  ← 尴尬了
```

**没有记忆的 Agent，每次对话都是第一次见面。**

---

## 6.2 本项目的三层记忆架构

本项目实现了一个经典的三层记忆系统：

```
┌──────────────────────────────────────────────────────────┐
│                    三层记忆系统                           │
│                                                          │
│  层级         存储位置            生命周期    每次读/写    │
│  ┌──────┐  ┌───────────────┐  ┌────────┐  ┌──────────┐  │
│  │工作记忆│  │ history.jsonl │  │ 会话级  │  │ 每轮读写  │  │
│  ├──────┤  ├───────────────┤  ├────────┤  ├──────────┤  │
│  │情景记忆│  │ YYYY-MM-DD.md│  │ 按日    │  │ 压缩时写  │  │
│  ├──────┤  ├───────────────┤  ├────────┤  ├──────────┤  │
│  │长期记忆│  │ MEMORY.md    │  │ 永久    │  │ 每轮读取  │  │
│  └──────┘  └───────────────┘  └────────┘  └──────────┘  │
└──────────────────────────────────────────────────────────┘
```

### ① 工作记忆（Working Memory）

**载体**：`memory/history.jsonl`

**作用**：存储完整的对话原始记录

**访问方式**：每轮对话都作为 `chat_history` 注入到 Agent 的 Prompt 中

**结构**：

```json
{"ts": "2026-05-14T08:30:00+08:00", "role": "user", "content": "搜索三个以上网页，整理有关香菇的资料"}
{"ts": "2026-05-14T08:30:05+08:00", "role": "assistant", "content": "好的，我来规划这个任务"}
{"ts": "2026-05-14T08:30:06+08:00", "role": "assistant", "content": "任务完成！...", "additional_kwargs": {"reasoning_content": "..."}}
```

**实现（`agent/memory.py`）**：

```python
class MemoryStore(BaseChatMessageHistory):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: Sequence[BaseMessage]):
        """将 LangChain 消息序列化到 JSONL"""
        for msg in messages:
            role = _TYPE_TO_JSONL_ROLE.get(msg.type, "unknown")
            extra = getattr(msg, "additional_kwargs", None) or None
            self.append_history(role, msg.content, additional_kwargs=extra)

    def append_history(self, role, content, additional_kwargs=None):
        """追加一条记录到 JSONL 文件和时间线"""
        entry = {
            "ts": datetime.now(_UTC8).isoformat(),
            "role": role,
            "content": content,
        }
        if additional_kwargs:
            entry["additional_kwargs"] = additional_kwargs
        # 写入 JSONL 文件
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 同时更新内存中的 messages 列表（作为 BaseChatMessageHistory）
        ...
```

### ② 情景记忆（Episodic Memory）

**载体**：`memory/YYYY-MM-DD.md`（按日期归档）

**作用**：记录"今天发生了什么"的摘要

**触发条件**：压缩发生时写入（详见 6.3 节）

**结构**：

```markdown
# 2026-05-14 情景记忆

## 08:30 香菇资料搜索任务
- 用户要求搜索3个以上网页并整理成HTML
- 派遣了3个web_researcher子代理
- 生成了 烤香菇之动物习性.html

## 14:15 代码审查任务
- 用户要求审查 agent/lc_agent.py
- 发现了2处潜在的 bug
- 已修复并通过测试
```

**实现**：

```python
def today_episode_path(self) -> Path:
    date = datetime.now(_UTC8).strftime("%Y-%m-%d")
    return self.memory_dir / f"{date}.md"

def append_episode(self, content: str) -> None:
    p = self.today_episode_path()
    existing = p.read_text(encoding="utf-8") if p.exists() else f"# {p.stem} 情景记忆\n"
    new_text = existing.rstrip() + "\n\n" + content.strip() + "\n"
    p.write_text(new_text, encoding="utf-8")
```

### ③ 长期记忆（Long-term Memory）

**载体**：`memory/MEMORY.md`

**作用**：存储跨会话的持久知识

**读取时机**：**每轮对话**都注入到 system prompt 中

**写入时机**：压缩时更新

**结构**：

```markdown
# 长期记忆

## 项目信息
- 项目名称: claude-agent-examples
- 语言: Python
- 框架: LangChain
- API 提供商: DeepSeek

## 用户偏好
- 用户喜欢简洁的回答
- 使用中文交流
- 偏好技术导向的解决方案

## 关键决策记录
- 2026-05-14: 子代理从手写系统迁移到纯 LangChain
- 并行执行通过 ParallelAgentExecutor 实现
```

**读取就是文件 IO**，每次构建 system prompt 时：

```python
def build_system_prompt(self) -> str:
    parts = []
    # ... SOUL.md, USER.md, identity.md ...

    # 注入长期记忆
    if self.memory:
        memory = self.memory.read_memory().strip()
        if memory:
            parts.append(f"# Long-term Memory\n\n{memory}")

    return "\n---\n".join(parts)
```

---

## 6.3 记忆压缩（Compaction）

为什么需要压缩？因为工作记忆（`chat_history`）会不断增长：

```
第1轮: chat_history = []
第2轮: chat_history = [msg1, msg2]
第10轮: chat_history = [msg1, msg2, ..., msg20] ← 越来越长
第N轮: chat_history 可能超过 200K tokens ← LLM 上下文窗口满了！
```

### 压缩触发条件

```python
class TokenTracker:
    def should_compact(self, max_context, threshold=0.7):
        """
        当最近一次调用的 输入 token > max_context * threshold 时触发
        默认: 200_000 * 0.7 = 140_000 tokens
        """
        return self._last_input_tokens > max_context * threshold
```

### 压缩流程

```
触发压缩条件（输入 tokens > 140K）
       │
       ▼
┌─────────────────────────────────────────────┐
│ Compactor.compact(history)                   │
│                                              │
│  1. 分割历史:                                 │
│     保留最近 K=10 轮 → 继续作为工作记忆          │
│     之前的全部历史 → 送去压缩                   │
│                                              │
│  2. 调用 LLM 生成摘要：                         │
│     "请根据以下对话历史：                       │
│      [大量对话文本]                             │
│      输出三个 XML 段落：                       │
│      <episode>今日情景摘要</episode>            │
│      <updated_memory>更新后的长期记忆</updated> │
│      <updated_user>更新后的用户档案</updated>"  │
│                                              │
│  3. 解析 LLM 输出（提取 XML 标签）              │
│     写入情景记忆 YYYY-MM-DD.md                 │
│     更新长期记忆 MEMORY.md                      │
│     更新用户档案 USER.md                        │
│                                              │
│  4. 标记已归档                                 │
└─────────────────────────────────────────────┘
```

### Compactor 实现

```python
class Compactor:
    K = 10  # 保留最近 10 轮

    def compact(self, history):
        if len(history) <= self.K:
            return history
        old = history[:-self.K]      # 要压缩的部分
        self._run_compaction(old)
        return history[-self.K:]     # 保留最近 K 轮

    def _run_compaction(self, old_messages):
        # 1. 构建压缩提示词
        prompt = _PROMPT_TEMPLATE.format(
            old_conversation=_messages_to_text(old_messages),
            current_memory=self.memory.read_memory(),
            current_user=self.memory.read_user(),
            today_episode=self.memory.read_today_episode(),
        )

        # 2. 调用 LLM 生成摘要
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content

        # 3. 解析 XML 标签
        if episode := _extract("episode", text):
            self.memory.append_episode(episode)
        if new_memory := _extract("updated_memory", text):
            self.memory.write_memory(new_memory)
        if new_user := _extract("updated_user", text):
            self.memory.write_user(new_user)

        # 4. 标记已归档
        self.memory.append_compact_marker()
```

### 启动时补归档

如果上次会话没有触发压缩就退出了，下次启动时补归档：

```python
# lc_agent.py __init__ 中
unarchived = self.memory_store.load_unarchived_history()
if len(unarchived) >= 2:
    self.compactor.compact_startup(unarchived)
```

---

## 6.4 记忆在 Agent 中的完整生命周期

```
┌──────────────┐
│  用户输入      │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 构建 System Prompt                                      │
│  ① SOUL.md（灵魂，固定）                                 │
│  ② USER.md（用户偏好，压缩时更新）                        │
│  ③ identity.md（工作区信息）                             │
│  ④ MEMORY.md（长期记忆，每轮注入） ← ← ← 关键！          │
│  ⑤ 当前技能列表                                         │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ AgentExecutor.invoke()                                  │
│  输入: system_prompt + chat_history（工作记忆） + input  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ReAct 循环（直到完成）                             │   │
│  │  每轮: LLM 推理 → 工具调用 → 观察结果              │   │
│  │  中间步骤存储在 agent_scratchpad 中                │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 写入工作记忆                                            │
│  user_input → history.jsonl                             │
│  assistant_reply → history.jsonl                        │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ TokenTracker: 检查是否需要压缩                            │
│  输入 tokens > 140K? → Compactor.compact()               │
│                        → 生成情景记忆 + 更新长期记忆      │
│                        → 保留最近 K=10 轮                │
│  否 → 等待下一轮                                         │
└────────────────────────────────────────────────────────┘
```

---

## 6.5 LangChain 内置的 Memory 类型（对比学习）

LangChain 也提供了多种内置 Memory 实现，但本项目的 Memory 是自己实现的。了解内置类型有助于理解设计思路：

| Memory 类型 | 说明 | 与项目对比 |
|------------|------|-----------|
| `ConversationBufferMemory` | 存储全部历史 | 类似工作记忆 |
| `ConversationSummaryMemory` | LLM 定期总结历史 | 类似 Compactor |
| `ConversationBufferWindowMemory` | 只保留最近 K 轮 | 压缩后只留 10 轮 |
| `VectorStoreRetrieverMemory` | 用向量数据库检索相关记忆 | 未使用 |

**为什么本项目自己实现而非直接用内置 Memory？**

1. **需要文件持久化**（JSONL + Markdown）— 内置 Memory 默认在内存中
2. **三层记忆架构** — LangChain 没有现成的三层记忆实现
3. **跨会话恢复** — 启动时读取 JSONL 重构 messages
4. **细粒度控制** — 什么时候压缩、保留多少、如何归档完全可控

---

## 6.6 记忆系统设计原则

### 原则1：写多读少

```
工作记忆   ← 每轮都写，每轮都读
情景记忆   ← 压缩时写，很少读（按需 grep）
长期记忆   ← 压缩时写，每轮都读
```

### 原则2：压缩不丢失信息

压缩不是"丢弃"旧对话——而是**把原始对话提炼成结构化知识**：

```
原始对话: [50轮对话，15万tokens]
     │
     ▼  Compactor
     │
情景记忆: "用户完成了香菇搜索任务"  ← 保留"发生了什么"
长期记忆: "用户偏好技术导向回答"     ← 保留"用户是什么样的人"
```

### 原则3：记忆有层次

```
短期（具体）  →  中期（摘要）  →  长期（知识）
  工作记忆         情景记忆        长期记忆
  详细但量大       折中           精炼但可能丢失细节
```

---

## 6.7 本章小结

```
┌──────────────────────────────────────────────┐
│           核心要点回顾                         │
│                                              │
│  ☐ 三层记忆：工作/情景/长期                     │
│  ☐ 工作记忆 = history.jsonl（全量原始记录）     │
│  ☐ 情景记忆 = YYYY-MM-DD.md（按日摘要）        │
│  ☐ 长期记忆 = MEMORY.md（跨会话核心知识）       │
│  ☐ Compactor 在 token > 140K 时触发压缩        │
│  ☐ 压缩=提炼, 不是丢弃                         │
│  ☐ 启动时自动补归档未压缩的历史                  │
└──────────────────────────────────────────────┘
```

### 思考题

1. 如果长期记忆 `MEMORY.md` 也超过了 token 限制，你会怎么处理？
2. 压缩时为什么保留最近 K=10 轮，而不是全部压缩？
3. 本项目的情景记忆是通过日期归档的，你能想到其他归档方式吗？

---

**[下一章：高级 Agent — 子代理与任务规划 →](07-高级Agent子代理与任务规划.md)**
