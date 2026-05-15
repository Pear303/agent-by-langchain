"""技能加载工具：load_skill。

功能：
- 按需加载技能的详细知识内容到上下文
- 只读操作（read_only=True，可并发执行）
- 在回答相关问题前调用，避免占用初始上下文窗口
"""
from __future__ import annotations

from .base import Tool
from .schema import StringSchema, tool_parameters_schema


class LoadSkill(Tool):
    """技能加载工具。
    
    加载指定技能的详细知识内容，在回答相关问题前调用。
    由于是只读操作，标记为 read_only（可并发执行）。
    """
    
    name = "load_skill"
    description = "加载指定技能的详细知识内容，在回答相关问题前调用"
    read_only = True  # 只读操作，可并发执行

    def __init__(self, skills_loader):
        """初始化工具。
        
        Args:
            skills_loader: SkillsLoader 实例
        """
        self._loader = skills_loader

    @property
    def parameters(self) -> dict:
        """工具参数 JSON Schema 定义。
        
        Returns:
            包含 skill_name 参数的 Schema
        """
        return tool_parameters_schema(
            skill_name=StringSchema(
                "技能名称，必须是系统提示中列出的可用技能之一"
            ),
        )

    def execute(self, skill_name: str) -> str:
        """加载技能内容。
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能的完整内容（包装在 XML 标签中）
        """
        print(f"[加载技能]: {skill_name}")
        return self._loader.get_content(skill_name)