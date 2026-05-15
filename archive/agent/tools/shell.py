"""Shell 命令执行工具：run_command。

功能：
- 在终端执行 shell 命令
- 捕获标准输出和标准错误
- 独占执行（exclusive=True，不可并发）
"""
from __future__ import annotations
import subprocess

from .base import Tool, tool_parameters
from .schema import StringSchema, tool_parameters_schema


@tool_parameters(tool_parameters_schema(
    command=StringSchema("要执行的 shell 命令"),
))
class RunCommand(Tool):
    """Shell 命令执行工具。
    
    在终端执行一条 shell 命令并返回输出。
    由于命令执行具有副作用，标记为 exclusive（独占），不可并发执行。
    """
    
    name = "run_command"
    description = "在终端执行一条 shell 命令并返回输出"
    exclusive = True  # 独占执行，不可并发

    def execute(self, command: str) -> str:
        """执行 shell 命令。
        
        Args:
            command: 要执行的命令字符串
            
        Returns:
            命令的标准输出或标准错误（优先返回 stdout）
        """
        print(f"[执行命令]: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout or result.stderr
        print(f"[命令输出]: {output}")
        return output