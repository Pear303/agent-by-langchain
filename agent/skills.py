"""技能加载器：负责从 skills 目录加载和管理可插拔技能包。

每个技能包是一个包含 SKILL.md 文件的目录，SKILL.md 使用 YAML frontmatter 
描述元数据（名称、描述、触发条件等），Markdown 正文包含技能的具体知识和指令。
"""
from __future__ import annotations
import re
from pathlib import Path

import yaml


class SkillsLoader:
    """技能加载器类。
    
    负责扫描 skills 目录，解析所有 SKILL.md 文件，提供技能的查询、
    加载和摘要生成功能。支持 always 标记的技能自动激活。
    """
    
    def __init__(self, skills_dir: Path):
        """初始化技能加载器。
        
        Args:
            skills_dir: 技能包根目录路径
        """
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}  # 存储所有已加载的技能 {name: {meta, body, path}}
        self._load_all()  # 启动时加载所有技能

    def _load_all(self) -> None:
        """扫描 skills 目录并加载所有 SKILL.md 文件。
        
        递归查找所有名为 SKILL.md 的文件，解析其 frontmatter 和正文，
        以技能名称为键存储到 self.skills 字典中。
        """
        if not self.skills_dir.exists():
            return
            
        # 递归查找所有 SKILL.md 文件并按路径排序
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            # 优先使用 frontmatter 中的 name，否则用目录名
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """解析 Markdown 文件的 YAML frontmatter。
        
        Frontmatter 格式：
        ```
        ---
        name: skill-name
        description: 技能描述
        tags: tag1, tag2
        always: true
        ---
        Markdown 正文内容...
        ```
        
        Args:
            text: 完整的 Markdown 文件内容
            
        Returns:
            (meta_dict, body_string) 元组，如果解析失败则返回空字典和原文
        """
        # 匹配 YAML frontmatter 块（--- 包裹的部分）
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
            
        # 解析 YAML 元数据
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
            
        return meta, match.group(2).strip()

    def get_content(self, name: str) -> str:
        """获取指定技能的完整内容（包装在 XML 标签中）。
        
        【调用方】lc_tools.py, skills.py (内部调用), tests/tools/test_skills.py
        
        Args:
            name: 技能名称
            
        Returns:
            格式化的技能内容字符串，如果技能不存在则返回错误提示和可用技能列表
        """
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'

    def get_always_skills(self) -> list[str]:
        """获取所有标记为 always 的技能名称列表。
        
        【调用方】context.py
        
        always 标记的技能会在每次构建系统提示词时自动加载，
        无需用户显式调用 load_skill 工具。
        
        Returns:
            需要始终激活的技能名称列表
        """
        always_skills = []
        for name, skill in self.skills.items():
            if skill["meta"].get("always", False):
                always_skills.append(name)
        return always_skills

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """批量加载指定技能的内容，用于注入到系统提示词中。
        
        【调用方】context.py
        
        Args:
            skill_names: 要加载的技能名称列表
            
        Returns:
            所有技能内容的拼接字符串，用双换行分隔；如果没有有效技能则返回空字符串
        """
        parts = []
        for name in skill_names:
            content = self.get_content(name)
            # 只添加非错误的技能内容
            if not content.startswith("Error:"):
                parts.append(content)
        return "\n\n".join(parts) if parts else ""

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        """生成所有可用技能的摘要列表（用于展示给用户或 LLM）。
        
        【调用方】context.py, subagents/registry.py
        
        摘要格式：
        - **skill-name**: 技能描述 [tags]
        
        Args:
            exclude: 要排除的技能名称集合（例如已加载的 always 技能）
            
        Returns:
            格式化的技能摘要字符串，每行一个技能；如果没有技能则返回空字符串
        """
        exclude = exclude or set()
        if not self.skills:
            return ""
            
        lines = []
        for name, skill in self.skills.items():
            # 跳过被排除的技能
            if name in exclude:
                continue
                
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            
            # 构建摘要行：名称 + 描述 + 可选标签
            line = f"- **{name}**: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
            
        return "\n".join(lines) if lines else ""