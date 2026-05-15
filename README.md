# DeepSeek Agent 示例

基于 LangChain 实现的多轮对话 AI Agent，带三层记忆系统、自动压缩、可插拔技能、任务规划与子代理派遣。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env       # 填入 DEEPSEEK_API_KEY
python agent.py
```

## 项目结构

```
agent.py                  入口
agent/
├── lc_agent.py           主 Agent 循环（LCAgent）
├── lc_tools.py           工具定义（@tool 函数）
├── subagent_parallel.py  子代理只读工具并发执行器
├── memory.py             三层记忆存储
├── compactor.py          历史压缩 → 情景记忆 + MEMORY.md
├── context.py            system prompt 构建
├── skills.py             技能加载器
├── telemetry.py          token 用量追踪
├── subagents/
│   ├── registry.py       子代理注册表（工具白名单 + max_turns）
│   └── spec.py           子代理规格定义
└── tools/
    └── todo.py           update_todos 实现

templates/                身份/提示词模板
skills/                   可插拔技能包
memory/                   运行期产物（已 gitignore）
```

## 三层记忆

| 层 | 载体 | 何时写 | 何时读 |
|----|------|--------|--------|
| 工作记忆 | `history` 列表（内存） | 每轮追加 | 全量传给 LLM |
| 情景记忆 | `memory/YYYY-MM-DD.md` | 压缩触发时 | 按需 grep |
| 长期记忆 | `memory/MEMORY.md` | 压缩/启动归档时 | 每轮注入 system prompt |

**自动压缩**：当 input_tokens 超过 200K × 50% = 100K 时，将较旧的历史浓缩为情景摘要并更新长期记忆，保留最近 10 轮。

## 内置工具

| 工具 | 说明 |
|------|------|
| `run_command` | 执行 shell 命令 |
| `web_fetch` | 抓取网页 |
| `read_file` / `write_file` / `edit_file` | 文件读写编辑 |
| `glob` / `grep` | 工作区搜索 |
| `load_skill` | 按需加载技能包 |
| `update_todos` | 任务规划 todolist |
| `dispatch_subagent` | 派遣子代理（8 种身份，独立上下文） |

## 子代理

派遣后拥有独立的运行上下文，办完只回传一段总结。**主Agent** 按顺序逐个派遣，但**子代理内部**的连续只读工具（web_fetch、read_file、glob、grep）会通过 `ThreadPoolExecutor` 并发执行。

身份定义在 `templates/subagents/{name}.md`，工具白名单和最大轮数写在 `registry.py`（安全设置不放模板）。

可用子代理：`quick_helper`、`doc_analyzer`、`web_researcher`、`validator`、`engine_executor`、`skill_manager`、`document_processor`、`system_maintainer`。

子代理不能递归派遣子代理，不能修改主 Agent 的 todolist。

## 内置技能

基础：clawhub（技能安装）、ddg-web-search、github、skill-creator、summarize、weather、find-skills  
浏览器：agent-browser（基于 Rust，需 node/npm）  
文档：pdf、word-docx、pptx、xlsx  
设计：ui-ux-pro-max  
知识：ontology、self-improving-agent  
维护：auto-updater

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | API 地址（默认 https://api.deepseek.com） |
| `DEEPSEEK_MODEL` | 使用的模型（默认 deepseek-v4-flash） |

其余环境变量请查看 `.env.example`。