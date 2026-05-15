"""工具基类：定义所有工具的抽象接口和参数验证机制。

核心功能：
- 定义 Tool 抽象基类，规范工具的 name、description、parameters、execute
- 提供 JSON Schema 格式的参数类型转换和验证
- 支持并发安全判断（只读且非独占的工具可并发执行）
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


# JSON Schema 类型到 Python 类型的映射
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

# 布尔值字符串表示的真值集合
_BOOL_TRUE = {"true", "1", "yes", "y", "on"}
# 布尔值字符串表示的假值集合
_BOOL_FALSE = {"false", "0", "no", "n", "off"}


def _cast_one(value: Any, schema: dict) -> Any:
    """根据 JSON Schema 递归转换单个值的类型。
    
    支持的类型转换：
    - integer: 字符串转整数，浮点数转整数（如果是整数值）
    - number: 字符串转浮点数
    - boolean: 字符串转布尔值（支持多种表示）
    - array: 递归转换数组元素
    - object: 递归转换对象属性
    
    Args:
        value: 待转换的值
        schema: JSON Schema 定义
        
    Returns:
        转换后的值，如果无法转换则返回原值
    """
    if value is None:
        return None
        
    t = schema.get("type")
    # type 可以是列表（nullable 类型）— 选择第一个非 null 类型
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else None

    # 整数类型转换
    if t == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    # 数字类型转换
    if t == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    # 布尔类型转换
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _BOOL_TRUE:
                return True
            if v in _BOOL_FALSE:
                return False
        return value

    # 数组类型转换（递归处理元素）
    if t == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [_cast_one(v, item_schema) for v in value]

    # 对象类型转换（递归处理属性）
    if t == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: (_cast_one(v, props[k]) if k in props else v) for k, v in value.items()}

    return value


def _validate_one(value: Any, schema: dict, path: str = "") -> None:
    """根据 JSON Schema 递归验证单个值的合法性。
    
    验证规则：
    - 类型检查（包括 nullable 类型）
    - 字符串：枚举值、最小/最大长度
    - 数字：最小/最大值
    - 数组：最小/最大项数、元素类型
    - 对象：必填字段、属性类型
    
    Args:
        value: 待验证的值
        schema: JSON Schema 定义
        path: 当前字段路径（用于错误提示）
        
    Raises:
        ValueError: 如果验证失败
    """
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if value is None and "null" in t:
            return
        t = non_null[0] if non_null else None

    # 空值检查
    if value is None and t is not None:
        raise ValueError(f"{path or 'value'}: must not be null")

    # 类型检查
    expected = _TYPE_MAP.get(t) if t else None
    if expected:
        # 布尔值不是整数或数字
        if t == "integer" and isinstance(value, bool):
            raise ValueError(f"{path or 'value'}: expected integer, got bool")
        if t == "number" and isinstance(value, bool):
            raise ValueError(f"{path or 'value'}: expected number, got bool")
        if not isinstance(value, expected):
            raise ValueError(
                f"{path or 'value'}: expected {t}, got {type(value).__name__}"
            )

    # 字符串约束验证
    if t == "string":
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path or 'value'}: must be one of {schema['enum']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path or 'value'}: length >= {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path or 'value'}: length <= {schema['maxLength']}")
    # 数字约束验证
    elif t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path or 'value'}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path or 'value'}: must be <= {schema['maximum']}")
    # 数组约束验证
    elif t == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path or 'value'}: must have >= {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path or 'value'}: must have <= {schema['maxItems']} items")
        item_schema = schema.get("items", {})
        for i, v in enumerate(value):
            _validate_one(v, item_schema, f"{path}[{i}]")
    # 对象约束验证
    elif t == "object":
        for k in schema.get("required", []):
            if k not in value:
                raise ValueError(f"{path or 'value'}: missing required field '{k}'")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                _validate_one(v, props[k], f"{path}.{k}" if path else k)


class Tool(ABC):
    """工具抽象基类。
    
    所有具体工具必须继承此类并实现以下抽象方法：
    - name: 工具名称
    - description: 工具描述
    - parameters: 参数 JSON Schema
    - execute: 执行逻辑
    
    属性：
    - read_only: 是否为只读操作（默认 False）
    - exclusive: 是否为独占操作（默认 False）
    - concurrency_safe: 是否可并发执行（只读且非独占的工具）
    """
    read_only: bool = False  # 是否为只读操作
    exclusive: bool = False  # 是否为独占操作

    @property
    def concurrency_safe(self) -> bool:
        """判断工具是否可以安全地并发执行。
        
        Returns:
            如果是只读且非独占操作则返回 True
        """
        return self.read_only and not self.exclusive

    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一标识名称。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具的功能描述，用于 LLM 理解何时调用此工具。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具的参数 JSON Schema 定义。"""
        ...

    def cast_params(self, params: dict) -> dict:
        """根据 JSON Schema 转换参数字典的类型。
        
        Args:
            params: 原始参数字典
            
        Returns:
            类型转换后的参数字典
        """
        return _cast_one(params, self.parameters)

    def validate_params(self, params: dict) -> None:
        """根据 JSON Schema 验证参数字典的合法性。
        
        Args:
            params: 待验证的参数字典
            
        Raises:
            ValueError: 如果参数验证失败
        """
        _validate_one(params, self.parameters)

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具的核心逻辑。
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果的字符串表示
        """
        ...


def tool_parameters(schema: dict):
    """类装饰器：将 JSON Schema 字典冻结到 Tool 子类并暴露为 parameters 属性。
    
    作用：减少样板代码，无需在每个 Tool 子类中手动定义 parameters 属性。
    
    使用示例：
        @tool_parameters(tool_parameters_schema(command=StringSchema(...)))
        class RunCommand(Tool):
            @property
            def name(self) -> str:
                return "run_command"
            # ... 其他方法
        
    Args:
        schema: JSON Schema 字典
        
    Returns:
        装饰器函数
    """
    def wrap(cls):
        cls._parameters_schema = schema
        cls.parameters = property(lambda self: type(self)._parameters_schema)
        # 从抽象方法集合中移除 parameters（因为已通过装饰器提供）
        if "parameters" in getattr(cls, "__abstractmethods__", set()):
            cls.__abstractmethods__ = frozenset(
                m for m in cls.__abstractmethods__ if m != "parameters"
            )
        return cls
    return wrap