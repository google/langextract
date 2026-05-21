# Schema 设计

LangExtract 的 schema 设计采用 **example-driven** 模式——schema 从用户提供的 `examples` 中自动推断，而非显式定义 Pydantic 模型。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Prompt 组装](prompt.md)**
- **→ [输出解析与实体对齐](alignment.md)**
- **→ [长文档分块](chunking.md)**

---

## 目录

- [支持的字段类型](#支持的字段类型)
- [实体定义方式](#实体定义方式)
- [Schema 的 JSON 表达与 Python 类对应关系](#schema-的-json-表达与-python-类对应关系)
- [Schema 约束如何应用到 LLM](#schema-约束如何应用到-llm)

---

## 支持的字段类型

| 类型 | 说明 | 示例值 |
|------|------|--------|
| `string` | 字符串 (主要类型) | `"John Smith"`, `"2024-01-15"` |
| `number` | 数值 (自动转为 string) | `42`, `3.14` |
| `dict` | 属性字典 (通过后缀识别) | `{"dosage": "10mg", "route": "oral"}` |
| `list` | 列表 (在 attributes 中) | `["symptom1", "symptom2"]` |

---

## 实体定义方式

实体通过 `Extraction` 类定义，每个实体包含三个核心属性：

```python
# langextract/core/data.py:64-118
@dataclasses.dataclass
class Extraction:
    extraction_class: str      # 实体类型 (如 "person", "medication")
    extraction_text: str       # 实体文本 (从原文提取的内容)
    char_interval: CharInterval | None = None  # 原文中的字符偏移
    alignment_status: AlignmentStatus | None = None  # 对齐状态
    extraction_index: int | None = None  # 排序索引
    group_index: int | None = None       # 分组索引
    attributes: dict[str, str | list[str]] | None = None  # 附加属性
```

---

## Schema 的 JSON 表达与 Python 类对应关系

LangExtract 的 schema 系统有两层抽象：

### 1. 数据层 (Data Layer)

用户通过 `ExampleData` 和 `Extraction` 定义示例：

```python
# 来自 README.md 的示例
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks?",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",      # 对应 JSON 键
                extraction_text="ROMEO",            # 对应 JSON 值
                attributes={"emotional_state": "wonder"}  # 附加属性
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
        ]
    )
]
```

对应的 JSON 表达 (在 prompt 中)：

```json
{
  "extractions": [
    {
      "character": "ROMEO",
      "character_attributes": {
        "emotional_state": "wonder"
      }
    },
    {
      "emotion": "But soft!",
      "emotion_attributes": {
        "feeling": "gentle awe"
      }
    }
  ]
}
```

### 2. Schema 层 (Schema Layer)

`BaseSchema` 是抽象基类，定义了从 examples 生成 provider 配置的接口：

```python
# langextract/core/schema.py:38-91
class BaseSchema(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> BaseSchema:
        """从示例数据构建 schema 实例"""

    @abc.abstractmethod
    def to_provider_config(self) -> dict[str, Any]:
        """转换为 provider 特定的配置 (如 Gemini 的 response_schema)"""

    @property
    @abc.abstractmethod
    def requires_raw_output(self) -> bool:
        """是否输出原始 JSON/YAML (无围栏标记)"""
```

### 3. FormatModeSchema: 通用格式约束

`FormatModeSchema` 是当前主要使用的 schema 实现，它不强制字段级结构，只保证输出格式：

```python
# langextract/core/schema.py:93-139
class FormatModeSchema(BaseSchema):
    def __init__(self, format_type: types.FormatType = types.FormatType.JSON):
        self.format_type = format_type
        self._format = "json" if format_type == types.FormatType.JSON else "yaml"

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[data.ExampleData],
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    ) -> FormatModeSchema:
        """从 examples 构建 schema (当前默认使用 JSON 格式)"""
        return cls(format_type=types.FormatType.JSON)

    def to_provider_config(self) -> dict[str, Any]:
        """返回 provider 配置"""
        return {"format": self._format}

    @property
    def requires_raw_output(self) -> bool:
        """JSON 格式输出原始 JSON (无围栏)，YAML 则需要围栏"""
        return self._format == "json"
```

---

## Schema 约束如何应用到 LLM

在 `extract()` 函数中，schema 被传递给 model factory：

```python
# langextract/extraction.py:298-303
language_model = factory.create_model(
    config=config,
    examples=prompt_template.examples if use_schema_constraints else None,
    use_schema_constraints=use_schema_constraints,
    fence_output=fence_output,
)
```

不同 provider 对 schema 的支持程度不同：

| Provider | Schema 支持 | 实现方式 |
|----------|-------------|----------|
| Gemini | 完整支持 | `response_schema` + 结构化输出 |
| OpenAI | 格式支持 | JSON mode |
| Ollama | 格式支持 | `format: "json"` 参数 |

---

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19
