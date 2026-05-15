"""子代理派遣工具：dispatch_subagent。

功能：
- 派遣预设身份的子代理独立处理任务
- 子代理拥有独立的 history、工具注册表和运行器
- 执行完成后只回传一段总结文本，不污染主上下文
- 支持并发派遣（多个 dispatch_subagent 可在同一轮并行执行）
"""
from __future__ import annotations
from threading import Lock

from .base import Tool
from .registry import ToolRegistry
from .schema import StringSchema, tool_parameters_schema


class DispatchSubagentTool(Tool):
    """派遣预设身份的子代理工具。
    
    子代理拥有独立的 history，跑完只回传一段总结文本，
    主 agent 的 history 中只多一条 tool_result。
    
    适用场景：
    - 抓取并阅读多个网页
    - 批量执行命令并整理输出
    - 需要试错的探索性搜索
    - 跨多文件查找
    
    并发特性：
    - 每次派遣都有独立的 history / ToolRegistry / AgentRunner
    - 若主模型在同一帧发出多个 dispatch_subagent，runner 可以并行等待它们完成
    - 结果按原 tool_use 顺序回填
    """

    name = "dispatch_subagent"
    exclusive = False  # 非独占，支持并发

    @property
    def concurrency_safe(self) -> bool:
        """判断是否可以安全地并发执行。
        
        Returns:
            始终返回 True，因为每次派遣都有独立的上下文
        """
        # 每次派遣都有独立 history / ToolRegistry / AgentRunner。若主模型在同一帧
        # 发出多个 dispatch_subagent，runner 可以并行等待它们完成，再按原顺序回填结果。
        return True

    def __init__(self, *, client, model: str,
                 parent_registry: ToolRegistry,
                 subagent_registry,
                 runner_factory):
        """初始化子代理派遣工具。
        
        Args:
            client: LLM 客户端实例
            model: 使用的模型名称
            parent_registry: 父级工具注册表（用于借出白名单工具）
            subagent_registry: 子代理注册表（提供子代理规格）
            runner_factory: 工厂函数，接收 (spec, sub_registry) 返回 AgentRunner 实例
        """
        self._client = client
        self._model = model
        self._parent_registry = parent_registry
        self._subagent_registry = subagent_registry
        self._runner_factory = runner_factory   # 注入: spec, sub_registry -> AgentRunner
        self._counter = 0  # 派遣计数器
        self._counter_lock = Lock()  # 计数器锁

    @property
    def description(self) -> str:
        """工具描述，包含可用子代理类型列表。
        
        Returns:
            详细的工具描述字符串
        """
        return (
            "派遣一个子代理去单独处理任务。子代理有自己独立的上下文，办完只回传"
            "一段文字总结，不污染主上下文。适用于：抓取并阅读多个网页、"
            "批量执行命令并整理输出、需要试错的探索性搜索、跨多文件查找等。"
            "若多件任务互不依赖，可在同一回复中发出多个 dispatch_subagent，"
            "运行时会并发派遣并按原 tool_use 顺序回填结果。\n\n"
            "可用 agent_type:\n"
            f"{self._subagent_registry.describe()}"
        )

    @property
    def parameters(self) -> dict:
        """工具参数 JSON Schema 定义。
        
        Returns:
            包含 agent_type、task、purpose 三个参数的 Schema
        """
        return tool_parameters_schema(
            agent_type=StringSchema(
                "子代理类型，必须是 description 中列出的可用类型之一",
                enum=self._subagent_registry.names(include_aliases=True),
            ),
            task=StringSchema(
                "交代给子代理的任务，写清要做什么、希望返回什么格式的总结"
            ),
            purpose=StringSchema(
                "一句话用途标签，仅用于终端打印",
                nullable=True,
            ),
        )

    def execute(self, *, agent_type: str, task: str, purpose: str | None = None) -> str:
        """执行子代理派遣。
        
        执行流程：
        1. 从子代理注册表获取规格
        2. 创建子代理专属的工具注册表（从父注册表借出白名单工具）
        3. 使用工厂函数创建子代理运行器
        4. 执行子代理对话循环
        5. 返回最终总结
        
        Args:
            agent_type: 子代理类型名称
            task: 任务描述
            purpose: 可选的用途标签（用于终端打印）
            
        Returns:
            子代理的总结文本，如果出错则返回错误消息
        """
        # 获取子代理规格
        spec = self._subagent_registry.get(agent_type)
        if spec is None:
            return (
                f"Error: unknown subagent '{agent_type}'. "
                f"Available: {self._subagent_registry.names(include_aliases=True)}"
            )

        # 子 registry：从父 registry 借出白名单 Tool 实例（Tool 多为无状态，共享指针即可）
        sub_registry = ToolRegistry()
        for tool_name in spec.tool_names:
            tool = self._parent_registry.get(tool_name)
            if tool is not None:
                sub_registry.register(tool)

        # 创建子代理运行器
        runner = self._runner_factory(spec=spec, sub_registry=sub_registry)

        # 递增计数器（线程安全）
        with self._counter_lock:
            self._counter += 1
            counter = self._counter

        # 打印派遣信息
        label = (purpose or task)[:60]
        print(f"\n[派遣子代理 #{counter} · {spec.name}]: {label}")
        print("  ┌── subagent context start ──")

        # 执行子代理对话循环
        history: list = [{"role": "user", "content": task}]
        try:
            final = runner.step(history)
        except Exception as exc:
            print(f"  └── subagent context end (异常: {exc}) ──")
            return f"Error: subagent '{agent_type}' raised: {exc}"

        # 打印完成信息
        print(f"  └── subagent context end (内部 history {len(history)} 条，回传 {len(final)} 字) ──")
        print(f"[子代理汇报]: {final}")
        print(f"[主上下文压缩]: 子代理仅向主 history 追加 {len(final)} 字\n")
        return final