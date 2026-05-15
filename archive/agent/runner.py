"""Agent 运行器：负责执行对话循环和工具调用。

核心职责：
- 管理对话历史的多轮交互
- 将 Anthropic 格式消息转换为 OpenAI 兼容格式
- 执行工具调用（支持并发安全工具的并行执行）
- 验证 tool_calls 的完整性
- 触发历史压缩
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

from .tools.registry import ToolRegistry


def _block_field(block, field: str, default: str = ""):
    """兼容 dict 和对象两种格式的 block 字段读取。
    
    Args:
        block: 内容块（可以是字典或对象）
        field: 要读取的字段名
        default: 默认值
        
    Returns:
        字段值，如果不存在则返回默认值
    """
    if isinstance(block, dict):
        return block.get(field, default)
    return getattr(block, field, default)


def _is_tool_use_block(block) -> bool:
    """判断内容块是否为工具调用块。
    
    Args:
        block: 内容块
        
    Returns:
        如果是 tool_use 类型则返回 True
    """
    return _block_field(block, "type") == "tool_use"


def _is_tool_result_block(block) -> bool:
    """判断内容块是否为工具结果块。
    
    Args:
        block: 内容块
        
    Returns:
        如果是 tool_result 类型则返回 True
    """
    return _block_field(block, "type") == "tool_result"


class AgentRunner:
    """Agent 运行器，负责执行对话循环和工具调用。
    
    工作流程：
    1. 接收用户输入和历史对话
    2. 转换消息格式为 OpenAI 兼容格式
    3. 调用 LLM API 获取回复
    4. 如果有工具调用，执行工具并继续下一轮
    5. 如果没有工具调用，返回最终文本回复
    6. 根据需要触发历史压缩
    """
    
    def __init__(
        self,
        client,
        model: str,
        registry: ToolRegistry,
        system_prompt: str,
        max_tokens: int = 200000,
        memory_store=None,
        token_tracker=None,
        compactor=None,
        max_context: int = 200_000,
        compact_threshold: float = 0.5,
        max_turns: int | None = None,
    ):
        """初始化 Agent 运行器。
        
        Args:
            client: LLM 客户端实例（OpenAI 兼容接口）
            model: 使用的模型名称
            registry: 工具注册表
            system_prompt: 系统提示词
            max_tokens: LLM 响应的最大 token 数
            memory_store: 可选的记忆存储管理器
            token_tracker: 可选的 Token 追踪器
            compactor: 可选的历史压缩器
            max_context: 上下文窗口的最大 token 数
            compact_threshold: 触发压缩的阈值比例（0.5 表示 50%）
            max_turns: 单步任务的最大迭代轮数限制
        """
        self.client = client
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.memory_store = memory_store
        self.token_tracker = token_tracker
        self.compactor = compactor
        self.max_context = max_context
        self.compact_threshold = compact_threshold
        self.max_turns = max_turns

    def step(self, history: list) -> str:
        """执行一轮完整对话（用户→...→最终文本），原地修改 history。
        
        这是 Agent 的核心循环，处理以下流程：
        1. 将历史消息转换为 OpenAI 兼容格式
        2. 调用 LLM API 获取回复
        3. 如果回复包含工具调用，执行工具并将结果加入历史
        4. 重复上述过程直到获得纯文本回复
        5. 记录 token 使用量并检查是否需要压缩
        
        Args:
            history: 对话历史列表，会被原地修改
            
        Returns:
            最终的文本回复字符串
        """
        import json
        
        turns = 0
        while True:
            # 检查是否达到最大轮数限制
            if self.max_turns is not None and turns >= self.max_turns:
                return f"（达到 max_turns={self.max_turns} 上限, 未办妥；history 中已有部分进展）"
            turns += 1
            
            # ── 步骤 1：转换消息格式为 OpenAI 兼容格式 ──
            openai_messages = []
            if self.system_prompt:
                openai_messages.append({"role": "system", "content": self.system_prompt})
            
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # 处理多模态内容块列表
                if isinstance(content, list):
                    # 检查这个消息包含什么类型的块
                    has_tool_use = any(_is_tool_use_block(b) for b in content)
                    has_tool_result = any(_is_tool_result_block(b) for b in content)
                    
                    # 情况 A：同时包含 tool_use 和 tool_result（混合消息）
                    if has_tool_use and has_tool_result:
                        text_parts = []
                        tool_calls = []
                        tool_results = []
                        
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype == "text":
                                    text_parts.append(block.get("text", ""))
                                elif btype == "tool_use":
                                    tool_calls.append({
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": str(block.get("input", {}))
                                        }
                                    })
                                elif btype == "tool_result":
                                    tool_results.append({
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": str(block.get("content", ""))
                                    })
                            else:
                                btype = getattr(block, "type", "")
                                if btype == "text":
                                    text_parts.append(getattr(block, "text", ""))
                                elif btype == "tool_use":
                                    tool_calls.append({
                                        "id": getattr(block, "id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": getattr(block, "name", ""),
                                            "arguments": str(getattr(block, "input", {}))
                                        }
                                    })
                                elif btype == "tool_result":
                                    tool_results.append({
                                        "role": "tool",
                                        "tool_call_id": getattr(block, "tool_use_id", ""),
                                        "content": str(getattr(block, "content", ""))
                                    })
                        
                        # 先添加 assistant 消息（包含文本和工具调用）
                        if text_parts or tool_calls:
                            assistant_msg = {"role": "assistant"}
                            if text_parts:
                                assistant_msg["content"] = "\n".join(text_parts)
                            if tool_calls:
                                assistant_msg["tool_calls"] = tool_calls
                            if msg.get("reasoning_content"):
                                assistant_msg["reasoning_content"] = msg["reasoning_content"]
                            openai_messages.append(assistant_msg)
                        
                        # 再添加工具结果消息（独立的 tool 角色）
                        openai_messages.extend(tool_results)
                    
                    # 情况 B：只包含 tool_result（user 角色的工具结果消息）
                    elif has_tool_result and not has_tool_use:
                        # 将所有 tool_result 转换为独立的 tool 消息
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype == "tool_result":
                                    openai_messages.append({
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": str(block.get("content", ""))
                                    })
                            else:
                                btype = getattr(block, "type", "")
                                if btype == "tool_result":
                                    openai_messages.append({
                                        "role": "tool",
                                        "tool_call_id": getattr(block, "tool_use_id", ""),
                                        "content": str(getattr(block, "content", ""))
                                    })
                    
                    # 情况 C：只包含 tool_use 或文本（assistant 消息）
                    else:
                        text_parts = []
                        tool_calls = []
                        
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype == "text":
                                    text_parts.append(block.get("text", ""))
                                elif btype == "tool_use":
                                    tool_calls.append({
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": str(block.get("input", {}))
                                        }
                                    })
                            else:
                                btype = getattr(block, "type", "")
                                if btype == "text":
                                    text_parts.append(getattr(block, "text", ""))
                                elif btype == "tool_use":
                                    tool_calls.append({
                                        "id": getattr(block, "id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": getattr(block, "name", ""),
                                            "arguments": str(getattr(block, "input", {}))
                                        }
                                    })
                        
                        # 添加 assistant 消息
                        if text_parts or tool_calls:
                            assistant_msg = {"role": "assistant"}
                            if text_parts:
                                assistant_msg["content"] = "\n".join(text_parts)
                            if tool_calls:
                                assistant_msg["tool_calls"] = tool_calls
                            if msg.get("reasoning_content"):
                                assistant_msg["reasoning_content"] = msg["reasoning_content"]
                            openai_messages.append(assistant_msg)
                else:
                    # 普通文本消息，直接添加
                    openai_msg = {"role": role, "content": content}
                    # 如果有 reasoning_content，添加到消息中
                    if msg.get("reasoning_content"):
                        openai_msg["reasoning_content"] = msg["reasoning_content"]
                    openai_messages.append(openai_msg)
            
            # ── 步骤 2：验证并清理消息 ──
            validated_messages = self._validate_tool_calls(openai_messages)

            # ── 步骤 3：调用 API 获取回复 ──
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=self.registry.get_definitions(),
                    messages=validated_messages,
                )
            except Exception:
                print("\n[FATAL] API 调用失败，转储消息结构：")
                for i, m in enumerate(validated_messages):
                    role = m.get("role", "?")
                    tc_ids = [tc["id"] for tc in m.get("tool_calls", [])] if "tool_calls" in m else []
                    tc_id = m.get("tool_call_id", "")
                    content_len = len(str(m.get("content", "")))
                    print(f"  [{i}] role={role} tc_ids={tc_ids} tc_id={tc_id} content_len={content_len}")
                raise
            
            # ── 步骤 4：记录 token 使用量 ──
            if self.token_tracker:
                self.token_tracker.record(self.model, response.usage)
            
            # ── 步骤 5：提取助手回复并转换为 Anthropic 格式 ──
            choice = response.choices[0]
            assistant_message = choice.message
            
            # 构建 Anthropic 兼容的内容格式
            content_blocks = []
            tool_calls = assistant_message.tool_calls or []
            
            if assistant_message.content:
                content_blocks.append({"type": "text", "text": assistant_message.content})
            
            for tc in tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments) if tc.function.arguments else {}
                })
            
            # 保存助手消息，包含 reasoning_content（如果存在）
            assistant_msg = {"role": "assistant", "content": content_blocks}
            if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
                assistant_msg["reasoning_content"] = assistant_message.reasoning_content
            history.append(assistant_msg)

            # ── 步骤 6：如果不是工具调用，直接返回文本回复 ──
            if not tool_calls:
                reply = assistant_message.content or ""
                if self.memory_store:
                    self.memory_store.append_history("assistant", reply)
                self._maybe_compact(history)
                return reply

            # ── 步骤 7：执行工具调用，将结果加入 history 后继续下一轮 ──
            # 过滤出纯 tool_use 块（content_blocks 混有 text 块，会导致空 ID 的 tool_result）
            tool_use_blocks = [b for b in content_blocks if _is_tool_use_block(b)]
            tool_results = self._execute_tool_blocks(tool_use_blocks)

            history.append({"role": "user", "content": tool_results})

    def _validate_tool_calls(self, messages: list[dict]) -> list[dict]:
        """验证并清理消息，确保所有 tool_calls 都有对应的响应。
        
        OpenAI API 要求：如果 assistant 消息包含 tool_calls，
        后面必须紧跟对应的 tool 消息响应每个 tool_call_id。
        如果有未响应的 tool_calls，移除对应的 assistant 消息及其 tool 响应。
        
        Args:
            messages: 待验证的消息列表
            
        Returns:
            验证后的消息列表，移除了不完整的工具调用对
        """
        # 第一遍：收集所有 tool 响应的 tool_call_id
        responded_ids = set()
        for msg in messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    responded_ids.add(tool_call_id)
        
        # 第二遍：过滤掉有未响应 tool_calls 的 assistant 消息及其 tool 响应
        validated = []
        skip_tool_ids = set()  # 来自被跳过 assistant 的 tool_call_id，需一并跳过
        
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_call_ids = [tc["id"] for tc in msg["tool_calls"]]
                all_responded = all(tid in responded_ids for tid in tool_call_ids)
                
                if all_responded:
                    validated.append(msg)
                else:
                    unresponded = [tid for tid in tool_call_ids if tid not in responded_ids]
                    print(f"[警告] 跳过包含 {len(unresponded)} 个未响应 tool_calls 的 assistant 消息")
                    # 将该 assistant 的所有 tool_call_id 加入跳过集合，
                    # 确保后续对应的 tool 响应也被跳过，避免产生孤儿 tool 消息
                    skip_tool_ids.update(tool_call_ids)
            
            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if not tool_call_id or tool_call_id in skip_tool_ids:
                    # 空 tool_call_id 或属于已跳过 assistant 的 tool 响应，一并跳过
                    continue
                validated.append(msg)
            else:
                validated.append(msg)
        
        return validated

    def _execute_tool_blocks(self, tool_blocks: list) -> list[dict]:
        """执行工具调用块，并行化连续的安全工具组。
        
        执行策略：
        - 如果工具标记为 concurrency_safe，收集连续的并发安全工具组并用线程池并行执行
        - 非并发安全工具顺序执行
        - 按原始顺序返回结果，保持工具调用的时序性
        
        Args:
            tool_blocks: 工具调用块列表
            
        Returns:
            工具执行结果列表，每个元素是 tool_result 格式的字典
        """
        results_by_id: dict[str, str] = {}
        i = 0
        while i < len(tool_blocks):
            block = tool_blocks[i]
            tool_name = _block_field(block, "name")
            tool_id = _block_field(block, "id")
            tool_input = _block_field(block, "input", default={})

            tool = self.registry.get(tool_name)

            # 如果工具支持并发，收集连续的并发安全工具组
            if tool is not None and tool.concurrency_safe:
                group = []
                while i < len(tool_blocks):
                    candidate_name = _block_field(tool_blocks[i], "name")
                    candidate_tool = self.registry.get(candidate_name)
                    if candidate_tool is None or not candidate_tool.concurrency_safe:
                        break
                    group.append(tool_blocks[i])
                    i += 1

                if len(group) > 1:
                    names = ", ".join(_block_field(b, "name") for b in group)
                    print(f"\n[并发执行 {len(group)} 个工具]: {names}\n")
                    with ThreadPoolExecutor(max_workers=len(group)) as pool:
                        contents = list(pool.map(
                            lambda b: self.registry.execute(
                                _block_field(b, "name"),
                                _block_field(b, "input", default={}),
                            ),
                            group,
                        ))
                    for b, content in zip(group, contents):
                        results_by_id[_block_field(b, "id")] = content
                else:
                    b = group[0]
                    results_by_id[_block_field(b, "id")] = self.registry.execute(
                        _block_field(b, "name"),
                        _block_field(b, "input", default={}),
                    )
                continue

            # 单个工具或非并发安全工具，顺序执行
            results_by_id[tool_id] = self.registry.execute(tool_name, tool_input)
            i += 1

        # 按原始顺序返回结果
        return [
            {
                "type": "tool_result",
                "tool_use_id": _block_field(block, "id"),
                "content": results_by_id[_block_field(block, "id")],
            }
            for block in tool_blocks
        ]

    def _maybe_compact(self, history: list) -> None:
        """根据需要压缩历史对话。
        
        当 token 使用量超过阈值时，调用压缩器将旧的历史归档到记忆文件，
        只保留最近的 K 条消息在内存中。
        
        Args:
            history: 对话历史列表，会被原地修改
        """
        if not (self.compactor and self.token_tracker):
            return
        if not self.token_tracker.should_compact(self.max_context, self.compact_threshold):
            return
        history[:] = self.compactor.compact(history)