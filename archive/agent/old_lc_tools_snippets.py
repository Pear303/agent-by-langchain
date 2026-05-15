"""
被替换的旧函数实现参考（lc_tools.py）。

这些函数在迁移到纯 LangChain 子 Agent 时被替换，
保留在此以供参考和回退。
"""

# ═══════════════════════════════════════════════════════════════════
# 旧版：set_subagent_context（被 set_subagent_deps 替代）
# ═══════════════════════════════════════════════════════════════════
#
# _subagent_registry: Any = None
# _parent_tool_registry: Any = None
# _client_ref: Any = None
# _model_ref: str = ""
#
# def set_subagent_context(
#     registry: Any, parent_registry: Any,
#     client: Any, model: str,
# ) -> None:
#     global _subagent_registry, _parent_tool_registry, _client_ref, _model_ref
#     _subagent_registry = registry
#     _parent_tool_registry = parent_registry
#     _client_ref = client
#     _model_ref = model


# ═══════════════════════════════════════════════════════════════════
# 旧版：dispatch_subagent（@tool 实现，使用 AgentRunner）
# ═══════════════════════════════════════════════════════════════════
#
# 使用旧版 ToolRegistry + AgentRunner 方式创建子代理执行器。
#
# @tool
# def dispatch_subagent(agent_type: str, task: str) -> str:
#     if _subagent_registry is None:
#         return "Error: Subagent registry not initialized"
#     if _parent_tool_registry is None or _client_ref is None:
#         return "Error: Subagent runner not initialized"
#
#     spec = _subagent_registry.get(agent_type)
#     if spec is None:
#         available = ", ".join(_subagent_registry.names())
#         return f"Error: unknown subagent '{agent_type}'. Available: {available}"
#
#     from .tools.registry import ToolRegistry
#     from .runner import AgentRunner
#
#     sub_registry = ToolRegistry()
#     for tool_name in spec.tool_names:
#         tool = _parent_tool_registry.get(tool_name)
#         if tool is not None:
#             sub_registry.register(tool)
#
#     runner = AgentRunner(
#         client=_client_ref,
#         model=_model_ref,
#         registry=sub_registry,
#         system_prompt=spec.system_prompt,
#         max_tokens=2000,
#         memory_store=None,
#         token_tracker=None,
#         compactor=None,
#         max_turns=spec.max_turns,
#     )
#
#     print(f"\n[派遣子代理 · {agent_type}]: {task[:80]}")
#     history: list = [{"role": "user", "content": task}]
#     try:
#         final = runner.step(history)
#     except Exception as exc:
#         return f"Error: subagent '{agent_type}' raised: {exc}"
#
#     print(f"[子代理汇报]: {final[:200]}")
#     return final


# ═══════════════════════════════════════════════════════════════════
# 旧版：lc_agent.py 中的 old_registry 创建（被移除）
# ═══════════════════════════════════════════════════════════════════
#
# old_registry = ToolRegistry()
# from .tools.shell import RunCommand
# from .tools.web import WebFetch
# from .tools.skills import LoadSkill
# from .tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool
# from .tools.search import GlobTool, GrepTool
#
# old_registry.register(RunCommand())
# old_registry.register(WebFetch())
# old_registry.register(LoadSkill(skills))
# old_registry.register(ReadFileTool(workspace))
# old_registry.register(WriteFileTool(workspace))
# old_registry.register(EditFileTool(workspace))
# old_registry.register(GlobTool(workspace))
# old_registry.register(GrepTool(workspace))
# set_subagent_context(
#     registry=sub_reg,
#     parent_registry=old_registry,
#     client=openai_client,
#     model=model,
# )
