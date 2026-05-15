# 🧠 LangChain Agent 从入门到精通

> 从零开始，彻底掌握用 LangChain 构建 AI Agent 的完整知识体系。
> 本教程基于本项目真实代码，由浅入深，每一章都配有可直接运行的代码示例。

---

## 📖 学习路径

本教程分为 **8 章**，建议按顺序学习：

```
第1章 ─── AI Agent 基础认知
   │
   ▼
第2章 ─── LangChain 核心概念
   │
   ▼
第3章 ─── Chain（链式调用）
   │
   ▼
第4章 ─── Tool（工具系统）
   │
   ▼
第5章 ─── Agent（智能体核心）
   │
   ▼
第6章 ─── Memory（记忆系统）
   │
   ▼
第7章 ─── 高级 Agent：子代理与任务规划
   │
   ▼
第8章 ─── 综合实战：读懂并改造本项目
```

---

## 📚 章节内容

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| **第1章** | `01-AI-Agent基础认知.md` | 什么是 AI Agent、和普通 AI 的区别、核心组成要素 |
| **第2章** | `02-LangChain核心概念.md` | LangChain 四大核心抽象、工具链全景图 |
| **第3章** | `03-Chain链式调用.md` | PromptTemplate、LLMChain、Runnable 接口、管道操作 |
| **第4章** | `04-工具系统Tool.md` | @tool 装饰器、参数验证、工具注册、工具并发 |
| **第5章** | `05-Agent智能体核心.md` | AgentExecutor、Agent 类型、推理-行动循环、回调 |
| **第6章** | `06-记忆系统Memory.md` | 工作记忆/情景记忆/长期记忆、压缩策略、历史管理 |
| **第7章** | `07-高级Agent子代理与任务规划.md` | 子代理设计模式、任务规划、并发派遣、安全约束 |
| **第8章** | `08-综合实战.md` | 串联全部知识、理解本项目完整运行流程、动手改造 |

---

## 🎯 学习目标

完成本教程后，你将能够：

1. ✅ 理解 AI Agent 的核心工作原理
2. ✅ 熟练使用 LangChain 的 Chain / Tool / Agent / Memory 四大组件
3. ✅ 读懂本项目的全部源代码
4. ✅ 独立用 LangChain 构建多 Agent 协作系统
5. ✅ 具备架构设计和安全约束意识

---

## 🛠 前置条件

- Python 3.10+
- 基本的 Python 编程能力
- 了解大语言模型（LLM）的基本概念（什么是 API、什么是 token）
- 一个可用的 LLM API Key（本项目使用 DeepSeek）

---

## 🔗 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- 本项目源码：`agent/` 目录 + `agent.py` 入口

---

> **开始学习 👉 [第1章：AI Agent 基础认知](01-AI-Agent基础认知.md)**
