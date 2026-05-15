"""工具注册表：管理所有可用工具的注册、查询和执行。

核心功能：
- 注册和查询工具实例
- 生成 OpenAI 兼容的工具定义（用于 API 调用）
- 参数验证和类型转换
- 执行工具并处理错误
"""
from __future__ import annotations
from typing import Any
from .base import Tool


class ToolRegistry:
    """工具注册表。
    
    管理所有已注册的工具，提供统一的接口来：
    - 获取工具定义（供 LLM API 使用）
    - 验证和转换参数
    - 执行工具调用
    
    错误处理策略：所有错误消息末尾附加提示语，引导 LLM 分析错误并尝试其他方法。
    """
    
    _HINT = "[Analyze the error above and try a different approach.]"

    def __init__(self):
        """初始化工具注册表。"""
        self._tools: dict[str, Tool] = {}  # 工具名称到实例的映射
        self._defs_cache: list[dict] | None = None  # 工具定义缓存

    def register(self, tool: Tool) -> None:
        """注册一个工具实例。
        
        Args:
            tool: Tool 子类实例
        """
        self._tools[tool.name] = tool
        self._defs_cache = None  # 清除缓存，下次获取定义时重新生成

    def get(self, name: str) -> Tool | None:
        """根据名称获取工具实例。
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，如果不存在则返回 None
        """
        return self._tools.get(name)

    def names(self) -> list[str]:
        """获取所有已注册工具的名称列表（排序）。
        
        Returns:
            排序后的工具名称列表
        """
        return sorted(self._tools.keys())

    def get_definitions(self) -> list[dict]:
        """获取所有工具的 OpenAI 兼容定义列表。
        
        定义格式：
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
        
        注意：MCP 工具（名称以 mcp_ 开头）排在最后。
        
        Returns:
            工具定义列表（带缓存）
        """
        if self._defs_cache is not None:
            return self._defs_cache
            
        builtin, mcp = [], []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            entry = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            # MCP 工具和非 MCP 工具分开存储
            (mcp if name.startswith("mcp_") else builtin).append(entry)
            
        self._defs_cache = builtin + mcp
        return self._defs_cache

    def prepare_call(self, name: str, params: Any):
        """准备工具调用：验证参数并转换为正确类型。
        
        Args:
            name: 工具名称
            params: 原始参数字典
            
        Returns:
            (tool_instance, cast_params, error_message) 三元组
            - 如果成功：error_message 为 None
            - 如果失败：tool_instance 和 cast_params 为 None
        """
        # 检查参数是否为字典类型
        if not isinstance(params, dict):
            return None, None, (
                f"Error: tool '{name}' received non-object params: "
                f"{type(params).__name__}"
            )
            
        # 检查工具是否存在
        tool = self._tools.get(name)
        if tool is None:
            return None, None, (
                f"Error: Unknown tool '{name}'. Available: {', '.join(self.names())}"
            )
            
        # 类型转换和验证
        try:
            cast = tool.cast_params(params)
            tool.validate_params(cast)
        except (ValueError, TypeError) as e:
            return tool, None, f"Error: invalid params for '{name}': {e}"
            
        return tool, cast, None

    def execute(self, name: str, params: Any) -> str:
        """执行工具调用。
        
        执行流程：
        1. 准备调用（验证和转换参数）
        2. 如果有错误，返回错误消息 + 提示
        3. 执行工具
        4. 如果结果以 "Error" 开头，附加提示
        5. 如果抛出异常，捕获并返回错误消息 + 提示
        
        Args:
            name: 工具名称
            params: 原始参数字典
            
        Returns:
            执行结果的字符串表示
        """
        tool, cast, err = self.prepare_call(name, params)
        if err:
            return f"{err}\n{self._HINT}"
            
        try:
            result = tool.execute(**cast)
            # 如果工具返回的错误消息，附加提示
            if isinstance(result, str) and result.startswith("Error"):
                return f"{result}\n{self._HINT}"
            return result
        except Exception as e:
            return f"Error executing {name}: {e}\n{self._HINT}"