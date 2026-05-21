# LangExtract Schema & Prompt 架构设计

本文档系统梳理 LangExtract 的核心内部机制，包括 schema 设计方式、prompt 组装逻辑、实体对齐机制、输出解析策略和长文档分块方法。

---

## Table of Contents

- [Overview: 抽取流程一页纸](#overview-抽取流程一页纸)
- [Schema 设计](#schema-设计)
- [Prompt 组装](#prompt-组装)
- [输出解析](#输出解析)
- [实体对齐 (Alignment)](#实体对齐-alignment)
- [长文档分块](#长文档分块)
- [已知限制与 FAQ](#已知限制与-faq)
- [文档 TODO](#文档-todo)

---

## Overview: 抽取流程一页纸

LangExtract 的信息抽取流程是一个典型的 **schema-driven** 流水线，从用户输入到最终返回结构化数据，经过以下阶段：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LangExtract 抽取流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户输入 (User Input)                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  text_or_documents │  │ prompt_description │  │    examples     │            │
│  │   (待抽取文本)    │  │   (抽取指令)     │  │  (少量示例)    │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Schema 推断 (Schema Inference)                  │   │
│  │  - 从 examples 中提取 extraction_class (实体类型)                      │   │
│  │  - 分析 extraction_text 的值类型 (string/number/dict/list)           │   │
│  │  - 构建 BaseSchema 或 FormatModeSchema 实例                           │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Prompt 组装 (Prompt Assembly)                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  1. System Instruction: prompt_description                     │    │   │
│  │  │  2. Few-shot Examples: 格式化 examples 为 JSON/YAML           │    │   │
│  │  │  3. Question: 当前 chunk 文本 (Q: ...)                        │    │   │
│  │  │  4. Answer Prefix: 引导模型输出 (A: )                          │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  关键类: QAPromptGenerator, ContextAwarePromptBuilder               │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LLM 推理 (Inference)                              │   │
│  │  - 支持 Gemini, OpenAI, Ollama 等多种 provider                        │   │
│  │  - 部分模型支持 schema constraints (结构化输出约束)                     │   │
│  │  - 批量处理 (batch) 提高吞吐量                                         │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      输出解析 (Output Parsing)                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  1. 围栏提取: 从 ```json / ```yaml 中提取内容                │    │   │
│  │  │  2. 格式解析: JSON.parse / yaml.safe_load                    │    │   │
│  │  │  3. 容错处理: <think> 标签过滤、宽松解析模式                   │    │   │
│  │  │  4. 结构转换: 转为 Extraction 对象序列                        │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  关键类: FormatHandler, Resolver                                     │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      实体对齐 (Entity Alignment)                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  1. 精确匹配: difflib.SequenceMatcher 逐 token 匹配          │    │   │
│  │  │  2. 模糊匹配: LCS (最长公共子序列) 算法                       │    │   │
│  │  │  3. 归一化: 小写 + 轻量词干化 (去除 s 后缀)                   │    │   │
│  │  │  4. 状态标记: MATCH_EXACT / MATCH_FUZZY / MATCH_LESSER      │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  关键类: WordAligner                                                 │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      结果返回 (Result Return)                          │   │
│  │  - AnnotatedDocument: 包含 extractions 列表                          │   │
│  │  - 每个 Extraction 包含: char_interval, alignment_status, attributes │   │
│  │  - 支持 visualization 可视化                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心数据流向伪代码

```python
# langextract/extraction.py:37-377 (extract 函数)
def extract(text_or_documents, prompt_description, examples, ...):
    # 1. 验证 examples 的对齐质量 (可选但推荐)
    if prompt_validation_level != OFF:
        alignment_report = validate_prompt_alignment(examples)
    
    # 2. 创建 Prompt 模板
    prompt_template = PromptTemplateStructured(
        description=prompt_description,
        examples=examples
    )
    
    # 3. 初始化 LLM (自动选择 provider)
    language_model = factory.create_model(
        model_id=model_id,
        examples=examples if use_schema_constraints else None
    )
    
    # 4. 创建 Annotator
    annotator = Annotator(
        language_model=language_model,
        prompt_template=prompt_template,
        format_handler=format_handler
    )
    
    # 5. 执行抽取 (内部包含分块、推理、解析、对齐)
    result = annotator.annotate_text(
        text=text_or_documents,
        resolver=Resolver(...),
        max_char_buffer=max_char_buffer,
        ...
    )
    
    return result
```

---

## Schema 设计

LangExtract 的 schema 设计采用 **example-driven** 模式——schema 从用户提供的 `examples` 中自动推断，而非显式定义 Pydantic 模型。

### 支持的字段类型

| 类型 | 说明 | 示例值 |
|------|------|--------|
| `string` | 字符串 (主要类型) | `"John Smith"`, `"2024-01-15"` |
| `number` | 数值 (自动转为 string) | `42`, `3.14` |
| `dict` | 属性字典 (通过后缀识别) | `{"dosage": "10mg", "route": "oral"}` |
| `list` | 列表 (在 attributes 中) | `["symptom1", "symptom2"]` |

### 实体定义方式

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

### Schema 的 JSON 表达与 Python 类对应关系

LangExtract 的 schema 系统有两层抽象：

#### 1. 数据层 (Data Layer)

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

#### 2. Schema 层 (Schema Layer)

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

#### 3. FormatModeSchema: 通用格式约束

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

### Schema 约束如何应用到 LLM

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

## Prompt 组装

Prompt 组装是 LangExtract 的核心环节，它将用户的 `prompt_description` 和 `examples` 转换为 LLM 可理解的指令格式。

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Prompt 组装流程                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入                                                                        │
│  ┌─────────────────┐  ┌─────────────────────────────────────────────┐      │
│  │  prompt_description │  │              examples                    │      │
│  │   "Extract persons..." │  │  [ExampleData(text=..., extractions=...)] │      │
│  └────────┬────────┘  └─────────────────────┬───────────────────────┘      │
│           │                                 │                               │
│           └────────────────┬────────────────┘                               │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               PromptTemplateStructured (数据容器)                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  description: "Extract persons and medications from text..."   │  │   │
│  │  │  examples: [ExampleData, ExampleData, ...]                      │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              FormatHandler (示例格式化)                                │   │
│  │                                                                      │   │
│  │  每个 ExampleData.extractions 被格式化为:                            │   │
│  │  {                                                                    │   │
│  │    "extractions": [                                                   │   │
│  │      {"person": "John", "person_attributes": {"age": "30"}},        │   │
│  │      {"medication": "Aspirin", "medication_attributes": {...}}      │   │
│  │    ]                                                                  │   │
│  │  }                                                                    │   │
│  │                                                                      │   │
│  │  输出格式: JSON 或 YAML，带或不带 ``` 围栏                            │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              QAPromptGenerator (最终组装)                              │   │
│  │                                                                      │   │
│  │  render(question=chunk_text) 生成:                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  [description]                                                 │  │   │
│  │  │                                                                 │  │   │
│  │  │  Examples                                                       │  │   │
│  │  │  Q: [example_1.text]                                           │  │   │
│  │  │  A: [formatted_extractions_1]                                   │  │   │
│  │  │                                                                 │  │   │
│  │  │  Q: [example_2.text]                                           │  │   │
│  │  │  A: [formatted_extractions_2]                                   │  │   │
│  │  │                                                                 │  │   │
│  │  │  Q: [current_chunk_text]                                        │  │   │
│  │  │  A:                                                             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           ContextAwarePromptBuilder (跨 chunk 上下文)                  │   │
│  │                                                                      │   │
│  │  可选功能: 注入前一个 chunk 的尾部文本作为上下文                         │   │
│  │                                                                      │   │
│  │  [Previous text]: ...the patient was prescribed                    │  │   │
│  │  [additional_context]                                                │  │   │
│  │                                                                      │   │
│  │  帮助解决指代消解问题: "She" → "Dr. Sarah Johnson"                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键代码路径

#### 1. 示例格式化 (`format_extraction_example`)

```python
# langextract/core/format_handler.py:116-149
def format_extraction_example(
    self, extractions: list[data.Extraction]
) -> str:
    """将 extractions 格式化为 prompt 中的示例"""
    items = [
        {
            ext.extraction_class: ext.extraction_text,
            f"{ext.extraction_class}{self.attribute_suffix}": (
                ext.attributes or {}
            ),
        }
        for ext in extractions
    ]

    if self.use_wrapper and self.wrapper_key:
        payload = {self.wrapper_key: items}  # {"extractions": [...]}
    else:
        payload = items

    if self.format_type == data.FormatType.YAML:
        formatted = yaml.safe_dump(payload, ...)
    else:
        formatted = json.dumps(payload, indent=2, ensure_ascii=False)

    return self._add_fences(formatted) if self.use_fences else formatted
```

#### 2. Prompt 渲染 (`QAPromptGenerator.render`)

```python
# langextract/prompting.py:115-138
def render(self, question: str, additional_context: str | None = None) -> str:
    """生成完整的 prompt 文本"""
    prompt_lines: list[str] = [f"{self.template.description}\n"]

    if additional_context:
        prompt_lines.append(f"{additional_context}\n")

    if self.template.examples:
        prompt_lines.append(self.examples_heading)  # "Examples"
        for ex in self.template.examples:
            prompt_lines.append(self.format_example_as_text(ex))

    prompt_lines.append(f"{self.question_prefix}{question}")  # "Q: ..."
    prompt_lines.append(self.answer_prefix)  # "A: "
    return "\n".join(prompt_lines)
```

#### 3. 示例格式化 (`format_example_as_text`)

```python
# langextract/prompting.py:98-113
def format_example_as_text(self, example: data.ExampleData) -> str:
    """将单个 example 格式化为 Q:A 对"""
    question = example.text
    answer = self.format_handler.format_extraction_example(example.extractions)

    return "\n".join([
        f"{self.question_prefix}{question}",
        f"{self.answer_prefix}{answer}\n",
    ])
```

#### 4. 跨 chunk 上下文 (`ContextAwarePromptBuilder`)

```python
# langextract/prompting.py:242-266
def _build_effective_context(
    self,
    document_id: str,
    additional_context: str | None,
) -> str | None:
    """组合前一个 chunk 的上下文和额外上下文"""
    context_parts: list[str] = []

    if self._context_window_chars and document_id in self._prev_chunk_by_doc_id:
        prev_text = self._prev_chunk_by_doc_id[document_id]
        window = prev_text[-self._context_window_chars :]
        context_parts.append(f"{self._CONTEXT_PREFIX}{window}")
        # 例如: "[Previous text]: ...the patient visited the clinic"

    if additional_context:
        context_parts.append(additional_context)

    return "\n\n".join(context_parts) if context_parts else None
```

### 完整 Prompt 示例

假设用户提供：

```python
prompt_description = """Extract all persons and their roles from the text.
Use exact text from the source. Do not paraphrase."""

examples = [
    lx.data.ExampleData(
        text="Dr. Smith, the chief surgeon, operated on patient Johnson.",
        extractions=[
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="Dr. Smith",
                attributes={"role": "chief surgeon"}
            ),
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="Johnson",
                attributes={"role": "patient"}
            ),
        ]
    )
]
```

生成的 prompt 将是：

```
Extract all persons and their roles from the text.
Use exact text from the source. Do not paraphrase.

Examples
Q: Dr. Smith, the chief surgeon, operated on patient Johnson.
A: ```json
{
  "extractions": [
    {
      "person": "Dr. Smith",
      "person_attributes": {
        "role": "chief surgeon"
      }
    },
    {
      "person": "Johnson",
      "person_attributes": {
        "role": "patient"
      }
    }
  ]
}
```

Q: [当前待处理的 chunk 文本]
A: 
```

---

## 输出解析

LLM 返回的原始文本需要经过解析才能转换为结构化的 `Extraction` 对象。解析过程由 `FormatHandler` 和 `Resolver` 协同完成。

### 解析流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         输出解析流程                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM 原始输出 (Raw Output)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  某些推理模型可能先输出思考过程:                                        │   │
│  │  <think>Let me analyze this text. I see Dr. Smith mentioned...</think> │   │
│  │                                                                      │   │
│  │  然后是结构化输出:                                                     │   │
│  │  ```json                                                              │   │
│  │  {                                                                    │   │
│  │    "extractions": [                                                   │   │
│  │      {"person": "Dr. Smith"},                                         │   │
│  │      {"medication": "Aspirin"}                                        │   │
│  │    ]                                                                  │   │
│  │  }                                                                    │   │
│  │  ```                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: <think> 标签过滤 (可选)                                      │   │
│  │                                                                      │   │
│  │  正则: <think>[\s\S]*?</think>\s*                                   │   │
│  │  位置: langextract/core/format_handler.py:46                        │   │
│  │                                                                      │   │
│  │  原因: DeepSeek-R1, QwQ 等推理模型会先输出思考过程                  │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 围栏提取 (Fence Extraction)                                  │   │
│  │                                                                      │   │
│  │  正则: ```(?P<lang>[A-Za-z0-9_+-]+)?\s*\n(?P<body>[\s\S]*?)```    │   │
│  │  位置: langextract/core/format_handler.py:41-44                     │   │
│  │                                                                      │   │
│  │  规则:                                                                │   │
│  │  - strict_fences=True: 必须恰好一个 ```json 或 ```yaml 块           │   │
│  │  - strict_fences=False: 宽松模式，支持无语言标签或无围栏             │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 格式解析 (Format Parsing)                                    │   │
│  │                                                                      │   │
│  │  JSON: json.loads(content)                                           │   │
│  │  YAML: yaml.safe_load(content)                                       │   │
│  │                                                                      │   │
│  │  容错: 如果第一次解析失败且有 <think> 标签，尝试去除后再解析          │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 结构提取 (Structure Extraction)                               │   │
│  │                                                                      │   │
│  │  期望结构 (wrapper 模式):                                             │   │
│  │  {"extractions": [{"key1": "value1"}, {"key2": "value2"}]}         │   │
│  │                                                                      │   │
│  │  兼容结构 (非 wrapper 模式):                                          │   │
│  │  [{"key1": "value1"}, {"key2": "value2"}]                           │   │
│  │                                                                      │   │
│  │  位置: langextract/core/format_handler.py:151-245 (parse_output)   │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 5: 转换为 Extraction 对象                                       │   │
│  │                                                                      │   │
│  │  每个字典项:                                                          │   │
│  │  {"person": "John", "person_attributes": {"age": "30"}}             │   │
│  │    ↓                                                                  │   │
│  │  Extraction(                                                          │   │
│  │    extraction_class="person",                                        │   │
│  │    extraction_text="John",                                           │   │
│  │    attributes={"age": "30"}                                          │   │
│  │  )                                                                    │   │
│  │                                                                      │   │
│  │  位置: langextract/resolver.py:424-523 (extract_ordered_extractions)│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键代码解析

#### 1. 围栏提取 (`_extract_content`)

```python
# langextract/core/format_handler.py:278-333
def _extract_content(self, text: str) -> str:
    """从文本中提取内容，处理围栏"""
    if not self.use_fences:
        return text.strip()  # 无围栏模式，直接返回

    matches = list(_FENCE_RE.finditer(text))
    
    # 验证语言标签 (json/yaml/yml)
    valid_tags = {
        data.FormatType.YAML: {"yaml", "yml"},
        data.FormatType.JSON: {"json"},
    }
    candidates = [m for m in matches if self._is_valid_language_tag(...)]

    if self.strict_fences:
        # 严格模式: 必须恰好一个有效围栏块
        if len(candidates) != 1:
            raise exceptions.FormatParseError("...")
        return candidates[0].group("body").strip()

    # 宽松模式
    if len(candidates) == 1:
        return candidates[0].group("body").strip()
    elif len(candidates) > 1:
        raise exceptions.FormatParseError("Multiple fenced blocks found")
    
    # 最后尝试: 任意围栏或无围栏
    if matches and len(matches) == 1:
        return matches[0].group("body").strip()
    
    return text.strip()  # 无围栏，直接使用
```

#### 2. 解析输出 (`parse_output`)

```python
# langextract/core/format_handler.py:151-245
def parse_output(
    self, text: str, *, strict: bool | None = None
) -> Sequence[Mapping[str, ExtractionValueType]]:
    """解析模型输出为提取数据"""
    if not text:
        raise exceptions.FormatParseError("Empty or invalid input string.")

    # Step 1: 提取内容 (围栏处理)
    content = self._extract_content(text)

    # Step 2: 解析 JSON/YAML (含 <think> 标签容错)
    try:
        parsed = self._parse_with_fallback(content, strict)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise exceptions.FormatParseError(...) from e

    # Step 3: 提取 extractions 列表
    require_wrapper = self.wrapper_key is not None and (
        self.use_wrapper or bool(strict)
    )

    if isinstance(parsed, dict):
        # Wrapper 模式: {"extractions": [...]}
        if require_wrapper:
            if self.wrapper_key not in parsed:
                raise exceptions.FormatParseError(
                    f"Content must contain an '{self.wrapper_key}' key."
                )
            items = parsed[self.wrapper_key]
        else:
            # 兼容: 尝试已知的 wrapper key
            if data.EXTRACTIONS_KEY in parsed:
                items = parsed[data.EXTRACTIONS_KEY]
            elif self.wrapper_key and self.wrapper_key in parsed:
                items = parsed[self.wrapper_key]
            else:
                items = [parsed]  # 单个对象作为单元素列表
    elif isinstance(parsed, list):
        # 非 wrapper 模式: [...]
        if require_wrapper and (strict or not self.allow_top_level_list):
            raise exceptions.FormatParseError(...)
        items = parsed
    else:
        raise exceptions.FormatParseError(
            f"Expected list or dict, got {type(parsed)}"
        )

    # Step 4: 验证每个 item 是字典
    for item in items:
        if not isinstance(item, dict):
            raise exceptions.FormatParseError(
                "Each item in the sequence must be a mapping."
            )
    
    return items
```

#### 3. <think> 标签容错 (`_parse_with_fallback`)

```python
# langextract/core/format_handler.py:261-276
def _parse_with_fallback(self, content: str, strict: bool):
    """解析内容，失败时尝试去除 <think> 标签"""
    try:
        if self.format_type == data.FormatType.YAML:
            return yaml.safe_load(content)
        return json.loads(content)
    except (yaml.YAMLError, json.JSONDecodeError):
        if strict:
            raise
        # 推理模型 (DeepSeek-R1, QwQ) 会在 JSON 前输出 <think>
        if _THINK_TAG_RE.search(content):
            stripped = _THINK_TAG_RE.sub("", content).strip()
            if self.format_type == data.FormatType.YAML:
                return yaml.safe_load(stripped)
            return json.loads(stripped)
        raise
```

#### 4. 转换为 Extraction 对象 (`extract_ordered_extractions`)

```python
# langextract/resolver.py:424-523
def extract_ordered_extractions(
    self,
    extraction_data: Sequence[Mapping[str, fh.ExtractionValueType]],
) -> Sequence[data.Extraction]:
    """将解析后的数据转换为 Extraction 对象列表"""
    processed_extractions = []
    extraction_index = 0
    index_suffix = self.extraction_index_suffix  # 可选: "_index"
    attributes_suffix = self.format_handler.attribute_suffix  # "_attributes"

    for group_index, group in enumerate(extraction_data):
        for extraction_class, extraction_value in group.items():
            # 跳过索引字段 (如果使用 index_suffix)
            if index_suffix and extraction_class.endswith(index_suffix):
                continue
            
            # 跳过属性字段 (单独处理)
            if attributes_suffix and extraction_class.endswith(attributes_suffix):
                continue

            # 值类型验证: 必须是 str/int/float
            if not isinstance(extraction_value, (str, int, float)):
                raise ValueError(
                    "Extraction text must be a string, integer, or float."
                )
            
            # 统一转为字符串
            if not isinstance(extraction_value, str):
                extraction_value = str(extraction_value)

            # 查找对应的索引 (如果有)
            if index_suffix:
                index_key = extraction_class + index_suffix
                extraction_index = group.get(index_key, None)
                if extraction_index is None:
                    continue  # 无索引则跳过
            else:
                extraction_index += 1

            # 查找对应的属性
            attributes = None
            if attributes_suffix:
                attributes_key = extraction_class + attributes_suffix
                attributes = group.get(attributes_key, None)

            # 创建 Extraction 对象
            processed_extractions.append(
                data.Extraction(
                    extraction_class=extraction_class,
                    extraction_text=extraction_value,
                    extraction_index=extraction_index,
                    group_index=group_index,
                    attributes=attributes,
                )
            )

    # 按索引排序 (如果使用 index_suffix)
    processed_extractions.sort(key=operator.attrgetter("extraction_index"))
    return processed_extractions
```

### 格式错误时的 Fallback 策略

| 场景 | 处理方式 | 控制参数 |
|------|----------|----------|
| 解析失败 (JSON/YAML 语法错误) | `suppress_parse_errors=True` 时返回空列表，否则抛异常 | `resolver_params={"suppress_parse_errors": True}` |
| 多个围栏块 | 严格模式抛异常，宽松模式取第一个 | `strict_fences` |
| 无围栏标签 | 宽松模式尝试直接解析整段文本 | `strict_fences=False` |
| 包含 `<think>` 标签 | 自动去除后重试解析 | 内置 (非 strict 模式) |
| 缺少 `extractions` wrapper | 宽松模式接受顶级列表 | `use_wrapper=False` 或 `allow_top_level_list=True` |

**注意**: `suppress_parse_errors` 在 `extract()` 中默认为 `True`，这意味着单个 chunk 的解析失败不会导致整个文档处理失败。

---

## 实体对齐 (Alignment)

实体对齐是 LangExtract 的核心能力之一——它将 LLM 抽取出的文本片段回溯到原文中的精确字符位置。这使得抽取结果可验证、可可视化。

### 对齐流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          实体对齐流程                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  resolved_extractions: [                                              │   │
│  │    Extraction(extraction_text="Dr. Smith"),                          │   │
│  │    Extraction(extraction_text="Aspirin 10mg")                        │   │
│  │  ]                                                                     │   │
│  │                                                                      │   │
│  │  source_text: "Dr. Smith prescribed Aspirin 10mg to the patient."    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 0: Tokenization & 归一化                                        │   │
│  │                                                                      │   │
│  │  原文 token 化:                                                       │   │
│  │  ["dr", "smith", "prescribed", "aspirin", "10mg", "to", ...]       │   │
│  │                                                                      │   │
│  │  提取文本 token 化 + 归一化:                                          │   │
│  │  - 小写: "Dr. Smith" → "dr. smith"                                  │   │
│  │  - 轻量词干化: "patients" → "patient" (去除 s 后缀)                  │   │
│  │                                                                      │   │
│  │  位置: langextract/resolver.py:1034-1069 (_tokenize_with_lowercase) │   │
│  │        langextract/resolver.py:1063-1069 (_normalize_token)         │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 精确匹配 (Exact Match)                                       │   │
│  │                                                                      │   │
│  │  算法: difflib.SequenceMatcher (Python 标准库)                       │   │
│  │  位置: langextract/resolver.py:921-977                              │   │
│  │                                                                      │   │
│  │  策略:                                                                │   │
│  │  1. 将所有 extraction_text 用特殊分隔符连接                          │   │
│  │  2. 与 source_text 进行全局序列匹配                                  │   │
│  │  3. 对每个匹配块，判断是完全匹配还是部分匹配                          │   │
│  │                                                                      │   │
│  │  匹配状态:                                                            │   │
│  │  - MATCH_EXACT: extraction_text 与原文完全一致                       │   │
│  │  - MATCH_LESSER: 匹配的文本比 extraction_text 短                     │   │
│  │                      (extraction 更长，只匹配到一部分)               │   │
│  │  - 不匹配: 进入模糊匹配阶段                                           │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 模糊匹配 (Fuzzy Match) - 仅当精确匹配失败时                  │   │
│  │                                                                      │   │
│  │  有两种算法:                                                          │   │
│  │                                                                      │   │
│  │  A) Legacy 算法 (deprecated)                                         │   │
│  │     - difflib.SequenceMatcher.ratio()                               │   │
│  │     - 滑动窗口遍历所有可能的匹配位置                                  │   │
│  │     - 位置: langextract/resolver.py:578-702 (_fuzzy_align_extraction)│   │
│  │                                                                      │   │
│  │  B) LCS 算法 (默认，推荐)                                             │   │
│  │     - 最长公共子序列 (Longest Common Subsequence)                   │   │
│  │     - 动态规划 O(n*m²) 时间复杂度                                    │   │
│  │     - 双重门控: coverage + density                                   │   │
│  │     - 位置: langextract/resolver.py:704-774 (_lcs_fuzzy_align_extraction)│   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 计算偏移量 & 设置状态                                        │   │
│  │                                                                      │   │
│  │  计算:                                                                │   │
│  │  - token_interval: 在 chunk 内的 token 索引 + token_offset          │   │
│  │  - char_interval: 通过 token 的 char_interval 计算字符偏移           │   │
│  │  - alignment_status: MATCH_EXACT / MATCH_FUZZY / MATCH_LESSER     │   │
│  │                                                                      │   │
│  │  对齐失败:                                                            │   │
│  │  - char_interval = None                                              │   │
│  │  - token_interval = None                                             │   │
│  │  - alignment_status = None                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键代码解析

#### 1. 精确匹配主流程 (`align_extractions`)

```python
# langextract/resolver.py:776-1031
def align_extractions(
    self,
    extraction_groups: Sequence[Sequence[data.Extraction]],
    source_text: str,
    token_offset: int = 0,
    char_offset: int = 0,
    enable_fuzzy_alignment: bool = True,
    fuzzy_alignment_threshold: float = 0.75,
    ...
) -> Sequence[Sequence[data.Extraction]]:
    """将 extractions 对齐到原文"""
    # Step 1: 准备 tokens
    source_tokens = list(_tokenize_with_lowercase(source_text, ...))
    
    # Step 2: 用特殊分隔符连接所有 extraction_text
    # 分隔符: "\u241F" (Unicode 单元分隔符)，确保不会出现在正常文本中
    delim = "\u241F"
    extraction_tokens = list(_tokenize_with_lowercase(
        f" {delim} ".join(
            extraction.extraction_text
            for extraction in itertools.chain(*extraction_groups)
        ),
        tokenizer_impl=tokenizer_impl,
    ))
    
    # Step 3: 精确匹配 (difflib.SequenceMatcher)
    self._set_seqs(source_tokens, extraction_tokens)
    
    # 遍历匹配块
    for i, j, n in self._get_matching_blocks()[:-1]:
        # i: source 中的起始 token 索引
        # j: extraction 中的起始 token 索引
        # n: 匹配的 token 数量
        
        # 查找对应的 extraction
        extraction, _ = index_to_extraction_group.get(j, (None, None))
        
        # 设置 token_interval
        extraction.token_interval = tokenizer_lib.TokenInterval(
            start_index=i + token_offset,
            end_index=i + n + token_offset,
        )
        
        # 通过 token 计算 char_interval
        start_token = tokenized_text.tokens[i]
        end_token = tokenized_text.tokens[i + n - 1]
        extraction.char_interval = data.CharInterval(
            start_pos=char_offset + start_token.char_interval.start_pos,
            end_pos=char_offset + end_token.char_interval.end_pos,
        )
        
        # 判断匹配类型
        extraction_text_len = len(extraction_tokens_for_this_extraction)
        if extraction_text_len == n:
            extraction.alignment_status = data.AlignmentStatus.MATCH_EXACT
            exact_matches += 1
        else:
            # 部分匹配 (extraction 更长，只匹配到一部分)
            if accept_match_lesser:
                extraction.alignment_status = data.AlignmentStatus.MATCH_LESSER
                lesser_matches += 1
            else:
                # 不接受部分匹配，重置
                extraction.token_interval = None
                extraction.char_interval = None
                extraction.alignment_status = None

    # Step 4: 模糊匹配 (对精确匹配失败的 extractions)
    if enable_fuzzy_alignment and unaligned_extractions:
        for extraction in unaligned_extractions:
            if fuzzy_alignment_algorithm == "lcs":
                aligned = self._lcs_fuzzy_align_extraction(...)
            else:
                aligned = self._fuzzy_align_extraction(...)
            
            if aligned:
                aligned_extractions.append(aligned)

    return aligned_extraction_groups
```

#### 2. LCS 模糊匹配算法 (`_lcs_fuzzy_align_extraction`)

```python
# langextract/resolver.py:704-774
def _lcs_fuzzy_align_extraction(
    self,
    extraction: data.Extraction,
    source_tokens_norm: list[str],  # 已归一化的原文 tokens
    tokenized_text: tokenizer_lib.TokenizedText,
    token_offset: int,
    char_offset: int,
    fuzzy_alignment_threshold: float = 0.75,
    fuzzy_alignment_min_density: float = 1/3,
    ...
) -> data.Extraction | None:
    """使用 LCS 算法进行模糊对齐"""
    # Step 1: Tokenize 和归一化 extraction_text
    extraction_tokens = list(_tokenize_with_lowercase(extraction.extraction_text, ...))
    extraction_tokens_norm = [_normalize_token(t) for t in extraction_tokens]
    
    # Step 2: 计算所有可能的 LCS 匹配
    # 返回: {match_count: LcsSpan(matches, start, end)}
    spans = _best_lcs_spans(source_tokens_norm, extraction_tokens_norm)
    
    # Step 3: 按匹配数量从高到低尝试，找到第一个通过双重门控的
    for k in sorted(spans.keys(), reverse=True):
        candidate = spans[k]
        if _accept_lcs_match(
            candidate,
            len(extraction_tokens_norm),
            threshold=fuzzy_alignment_threshold,
            min_density=fuzzy_alignment_min_density,
        ):
            accepted = candidate
            break
    
    if accepted is None:
        return None
    
    # Step 4: 设置 intervals 和状态
    extraction.token_interval = tokenizer_lib.TokenInterval(
        start_index=accepted.start + token_offset,
        end_index=accepted.end + 1 + token_offset,
    )
    
    start_token = tokenized_text.tokens[accepted.start]
    end_token = tokenized_text.tokens[accepted.end]
    extraction.char_interval = data.CharInterval(
        start_pos=char_offset + start_token.char_interval.start_pos,
        end_pos=char_offset + end_token.char_interval.end_pos,
    )
    
    extraction.alignment_status = data.AlignmentStatus.MATCH_FUZZY
    return extraction
```

#### 3. LCS 双重门控 (`_accept_lcs_match`)

```python
# langextract/resolver.py:1165-1192
def _accept_lcs_match(
    span: LcsSpan,
    extraction_len: int,
    threshold: float = 0.75,
    min_density: float = 1/3,
) -> bool:
    """应用覆盖度和密度双重门控"""
    if span.matches == 0 or extraction_len == 0:
        return False
    
    # Coverage Gate (覆盖度): 匹配的 token 数 >= 阈值比例
    # 例如: extraction 有 4 个 tokens，threshold=0.75，需要至少匹配 3 个
    needed = math.ceil(extraction_len * threshold)
    if span.matches < needed:
        return False
    
    # Density Gate (密度): 匹配的 token 数 / 匹配区间长度 >= min_density
    # 防止匹配的 tokens 分散在太长的区间中
    # 例如: 匹配 2 个 tokens，但分散在 10 个 token 的区间中 → 密度 0.2 < 1/3 → 拒绝
    if span.span_len <= 0:
        return False
    density = span.matches / span.span_len
    return density >= min_density
```

#### 4. Token 归一化 (`_normalize_token`)

```python
# langextract/resolver.py:1063-1069
@functools.lru_cache(maxsize=10000)
def _normalize_token(token: str) -> str:
    """小写 + 轻量词干化 (去除复数 s)"""
    token = token.lower()
    # 长度 > 3 且以 s 结尾且不以 ss 结尾 → 去除 s
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token
```

### 对齐状态说明

| 状态 | 值 | 含义 | 示例 |
|------|-----|------|------|
| `MATCH_EXACT` | `"match_exact"` | 精确匹配 | extraction_text="John"，原文中恰好有 "John" |
| `MATCH_LESSER` | `"match_lesser"` | 部分匹配 (匹配文本更短) | extraction_text="John Smith"，只匹配到 "John" |
| `MATCH_FUZZY` | `"match_fuzzy"` | 模糊匹配 | extraction_text="Jon"，匹配到原文的 "John" |
| `None` | - | 对齐失败 | 无法在原文中找到对应片段 |

### 对齐失败时的处理

对齐失败的 extraction 会保留，但 `char_interval` 和 `token_interval` 为 `None`。用户可以通过过滤来只保留成功对齐的结果：

```python
# 只保留成功对齐的 extractions
grounded_extractions = [
    e for e in result.extractions 
    if e.char_interval is not None
]
```

**原因**: LLM 可能从 few-shot examples 中"幻觉"出内容，或者提取的文本与原文表述不完全一致。LangExtract 不会丢弃这些结果，而是让用户决定如何处理。

### 对齐参数配置

对齐参数通过 `resolver_params` 传递给 `extract()`：

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    resolver_params={
        # 模糊匹配开关
        "enable_fuzzy_alignment": True,
        
        # 覆盖度阈值: 至少匹配 75% 的 tokens
        "fuzzy_alignment_threshold": 0.75,
        
        # 密度阈值: 匹配 tokens / 区间长度 >= 1/3
        "fuzzy_alignment_min_density": 1/3,
        
        # 算法选择: "lcs" (默认) 或 "legacy" (deprecated)
        "fuzzy_alignment_algorithm": "lcs",
        
        # 是否接受部分匹配 (MATCH_LESSER)
        "accept_match_lesser": True,
        
        # 解析错误时是否抑制异常
        "suppress_parse_errors": True,
    }
)
```

---

## 长文档分块

当输入文本超过 LLM 的上下文窗口或 `max_char_buffer` 限制时，LangExtract 会将文档分割成多个 chunks 分别处理。

### 分块策略

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          长文档分块策略                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  核心原则                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. 优先按句子边界分割 (保持语义完整性)                                 │   │
│  │  2. 尊重换行符 (诗歌、列表等格式)                                      │   │
│  │  3. 单句过长时按 token 分割                                            │   │
│  │  4. 单个 token 超过 buffer 时单独成块                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  分块场景                                                                    │
│                                                                             │
│  场景 A: 单句超长，需要在句内分割                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  原文 (诗歌):                                                         │   │
│  │  "No man is an island,                                               │   │
│  │   Entire of itself,                                                  │   │
│  │   Every man is a piece of the continent,                             │   │
│  │   A part of the main."                                               │   │
│  │                                                                      │   │
│  │  max_char_buffer=40                                                  │   │
│  │                                                                      │   │
│  │  分块结果:                                                            │   │
│  │  Chunk 1: "No man is an island,\nEntire of itself,"      (38 chars)│   │
│  │  Chunk 2: "Every man is a piece of the continent,"       (38 chars)│   │
│  │  Chunk 3: "A part of the main."                        (19 chars)   │   │
│  │                                                                      │   │
│  │  特点: 尊重换行符，在换行处优先分割                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  场景 B: 单个 token 超长                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  原文: "This is antidisestablishmentarianism."                       │   │
│  │  max_char_buffer=20                                                  │   │
│  │                                                                      │   │
│  │  分块结果:                                                            │   │
│  │  Chunk 1: "This is"                                      (7 chars)  │   │
│  │  Chunk 2: "antidisestablishmentarianism"                (28 chars)  │   │
│  │  Chunk 3: "."                                          (1 char)     │   │
│  │                                                                      │   │
│  │  特点: 超长 token 即使超过 buffer 也单独成块                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  场景 C: 多短句可合并                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  原文: "Roses are red. Violets are blue. Flowers are nice. And so   │   │
│  │         are you."                                                     │   │
│  │  max_char_buffer=60                                                  │   │
│  │                                                                      │   │
│  │  分块结果:                                                            │   │
│  │  Chunk 1: "Roses are red. Violets are blue. Flowers are nice."      │   │
│  │                                                    (50 chars)        │   │
│  │  Chunk 2: "And so are you."                            (15 chars)   │   │
│  │                                                                      │   │
│  │  特点: 多个完整句子可合并到一个 chunk (不超过 buffer)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键代码解析

#### 1. ChunkIterator 主逻辑 (`__next__`)

```python
# langextract/chunking.py:441-506
def __next__(self) -> TextChunk:
    # 获取下一个句子 (或句子的剩余部分)
    sentence = next(self.sentence_iter)
    
    # 策略 1: 如果第一个 token 就超过 buffer，单独成块
    curr_chunk = create_token_interval(
        sentence.start_index, sentence.start_index + 1
    )
    if self._tokens_exceed_buffer(curr_chunk):
        self.sentence_iter = SentenceIterator(
            self.tokenized_text, curr_token_pos=sentence.start_index + 1
        )
        self.broken_sentence = True
        return TextChunk(token_interval=curr_chunk, document=self.document)
    
    # 策略 2: 在句子内追加 tokens，直到接近 buffer
    start_of_new_line = -1
    for token_index in range(curr_chunk.start_index, sentence.end_index):
        # 记录换行位置 (用于优先在换行处分割)
        if self.tokenized_text.tokens[token_index].first_token_after_newline:
            start_of_new_line = token_index
        
        test_chunk = create_token_interval(
            curr_chunk.start_index, token_index + 1
        )
        
        if self._tokens_exceed_buffer(test_chunk):
            # 超过 buffer 了
            # 优先在最近的换行处分割 (如果有)
            if start_of_new_line > 0 and start_of_new_line > curr_chunk.start_index:
                curr_chunk = create_token_interval(
                    curr_chunk.start_index, start_of_new_line
                )
            # 更新句子迭代器，下次从这里继续
            self.sentence_iter = SentenceIterator(
                self.tokenized_text, curr_token_pos=curr_chunk.end_index
            )
            self.broken_sentence = True
            return TextChunk(token_interval=curr_chunk, document=self.document)
        else:
            curr_chunk = test_chunk  # 继续追加
    
    # 策略 3: 整句没超过 buffer，尝试合并更多句子
    if self.broken_sentence:
        self.broken_sentence = False
    else:
        for sentence in self.sentence_iter:
            test_chunk = create_token_interval(
                curr_chunk.start_index, sentence.end_index
            )
            if self._tokens_exceed_buffer(test_chunk):
                self.sentence_iter = SentenceIterator(
                    self.tokenized_text, curr_token_pos=curr_chunk.end_index
                )
                return TextChunk(token_interval=curr_chunk, document=self.document)
            else:
                curr_chunk = test_chunk  # 合并整句
    
    return TextChunk(token_interval=curr_chunk, document=self.document)
```

#### 2. 句子边界检测 (`SentenceIterator`)

```python
# langextract/chunking.py:282-340
class SentenceIterator:
    """迭代 tokenized 文本的句子"""
    
    def __next__(self) -> tokenizer_lib.TokenInterval:
        # 找到包含当前 token 的句子范围
        sentence_range = tokenizer_lib.find_sentence_range(
            self.tokenized_text.text,
            self.tokenized_text.tokens,
            self.curr_token_pos,
        )
        # 从当前位置开始，而不是句子开头
        # (如果我们在句子中间，从这里继续)
        sentence_range = create_token_interval(
            self.curr_token_pos, sentence_range.end_index
        )
        self.curr_token_pos = sentence_range.end_index
        return sentence_range
```

### Overlap 与上下文窗口

LangExtract **没有使用传统的 chunk overlap 机制**，而是提供了 **`context_window_chars`** 参数来解决跨 chunk 的指代消解问题。

| 机制 | 说明 | 示例 |
|------|------|------|
| 传统 overlap | 相邻 chunks 共享部分文本 | Chunk1: [0-100], Chunk2: [80-180] |
| LangExtract context_window | 前一个 chunk 的尾部文本作为 prompt 上下文 | Chunk2 的 prompt 包含 Chunk1 的最后 N 个字符 |

**ContextAwarePromptBuilder 实现**:

```python
# langextract/prompting.py:179-276
class ContextAwarePromptBuilder(PromptBuilder):
    """支持跨 chunk 上下文追踪的 prompt builder"""
    
    _CONTEXT_PREFIX = "[Previous text]: ..."
    
    def __init__(
        self,
        generator: QAPromptGenerator,
        context_window_chars: int | None = None,  # 例如: 100
    ):
        super().__init__(generator)
        self._context_window_chars = context_window_chars
        self._prev_chunk_by_doc_id: dict[str, str] = {}  # 按文档追踪
    
    def build_prompt(
        self,
        chunk_text: str,
        document_id: str,
        additional_context: str | None = None,
    ) -> str:
        # 构建有效上下文 (前一个 chunk + 额外上下文)
        effective_context = self._build_effective_context(
            document_id, additional_context
        )
        
        prompt = self._generator.render(
            question=chunk_text,
            additional_context=effective_context,
        )
        
        # 更新状态: 保存当前 chunk 供下一个使用
        self._update_state(document_id, chunk_text)
        return prompt
    
    def _build_effective_context(
        self, document_id: str, additional_context: str | None
    ) -> str | None:
        context_parts: list[str] = []
        
        # 注入前一个 chunk 的尾部
        if self._context_window_chars and document_id in self._prev_chunk_by_doc_id:
            prev_text = self._prev_chunk_by_doc_id[document_id]
            window = prev_text[-self._context_window_chars :]  # 取尾部
            context_parts.append(f"{self._CONTEXT_PREFIX}{window}")
        
        if additional_context:
            context_parts.append(additional_context)
        
        return "\n\n".join(context_parts) if context_parts else None
```

**使用示例**:

```python
result = lx.extract(
    text_or_documents=long_text,
    prompt_description=prompt,
    examples=examples,
    context_window_chars=100,  # 每个 chunk 包含前一个 chunk 的最后 100 字符
)
```

**效果**:

假设文档被分为两个 chunks：
- Chunk1: "Dr. Sarah Johnson is a cardiologist at the hospital. She"
- Chunk2: " specializes in heart disease and hypertension."

没有 context_window 时，Chunk2 的 "She" 可能无法正确解析。

有 `context_window_chars=50` 时，Chunk2 的 prompt 会包含：
```
[Previous text]: ...cardiologist at the hospital. She

Q: specializes in heart disease and hypertension.
A: 
```

这样 LLM 就能知道 "She" 指的是 "Dr. Sarah Johnson"。

### 跨 Chunk 实体合并与去重

LangExtract 目前 **没有自动的跨 chunk 实体去重机制**。每个 chunk 的处理是独立的，结果累积到 `per_doc` 字典中。

```python
# langextract/annotation.py:307-332 (Annotator._annotate_documents_single_pass)
def _annotate_documents_single_pass(...):
    per_doc: DefaultDict[str, list[data.Extraction]] = collections.defaultdict(list)
    
    for batch in batch_iter:
        # ... 推理、解析、对齐 ...
        
        for text_chunk, scored_outputs in zip(batch, outputs):
            # ...
            
            aligned_extractions = resolver.align(...)
            
            for extraction in aligned_extractions:
                # 直接追加，没有去重
                per_doc[text_chunk.document_id].append(extraction)
```

**用户需要自己处理去重**，可以基于：
1. `char_interval` 重叠检测
2. `extraction_text` + `extraction_class` 相似度

**例外: Sequential Extraction Passes**

当使用 `extraction_passes > 1` 时，多次抽取的结果会进行非重叠合并：

```python
# langextract/annotation.py:46-84
def _merge_non_overlapping_extractions(
    all_extractions: list[Iterable[data.Extraction]],
) -> list[data.Extraction]:
    """合并多次抽取的结果，重叠时保留较早的抽取"""
    if not all_extractions:
        return []
    if len(all_extractions) == 1:
        return list(all_extractions[0])
    
    merged_extractions = list(all_extractions[0])  # 第一次抽取的结果
    
    for pass_extractions in all_extractions[1:]:
        for extraction in pass_extractions:
            # 检查是否与已合并的结果重叠
            overlaps = False
            if extraction.char_interval is not None:
                for existing_extraction in merged_extractions:
                    if existing_extraction.char_interval is not None:
                        if _extractions_overlap(extraction, existing_extraction):
                            overlaps = True
                            break
            
            # 只有不重叠时才添加
            if not overlaps:
                merged_extractions.append(extraction)
    
    return merged_extractions

def _extractions_overlap(
    extraction1: data.Extraction, extraction2: data.Extraction
) -> bool:
    """检查两个 extraction 的字符区间是否重叠"""
    # [start1, end1) 与 [start2, end2) 重叠
    return start1 < end2 and start2 < end1
```

**注意**: 这是同一文档多次抽取的合并策略，不是跨 chunk 去重。

### 分块参数配置

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    # 分块相关参数
    max_char_buffer=1000,      # 每个 chunk 的最大字符数
    batch_length=10,            # 每批处理的 chunk 数量
    max_workers=10,             # 并行 worker 数
    context_window_chars=100,   # 前一个 chunk 的上下文字符数 (可选)
    extraction_passes=1,        # 抽取次数 (可选，多次抽取时合并非重叠结果)
)
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_char_buffer` | 1000 | 每个 chunk 的最大字符数。调小可提高准确率但增加 API 调用。 |
| `batch_length` | 10 | 每批处理的 chunk 数量。与 `max_workers` 共同决定并行度。 |
| `max_workers` | 10 | 最大并行 worker 数。有效并行度受限于 `min(batch_length, max_workers)`。 |
| `context_window_chars` | `None` | 前一个 chunk 的上下文字符数。用于指代消解。 |
| `extraction_passes` | 1 | 抽取次数。> 1 时执行多次抽取并合并非重叠结果。 |

---

## 已知限制与 FAQ

### Q1: 为什么我的实体对齐失败了？

**常见原因**:

1. **LLM 提取的文本与原文不一致**
   - LLM 可能 paraphrase（转述）原文，例如原文是 "John Smith"，但 LLM 返回 "Mr. Smith"
   - 解决方案：在 prompt_description 中强调 "Use exact text from the source. Do not paraphrase."

2. **提取文本跨越 chunk 边界**
   - 如果一个实体被分割在两个 chunks 中，对齐可能失败
   - 解决方案：使用 `context_window_chars` 参数，或调整 `max_char_buffer`

3. **模糊匹配阈值设置过高**
   - 默认 `fuzzy_alignment_threshold=0.75`，如果提取文本与原文差异较大，可能无法匹配
   - 解决方案：调低阈值 `resolver_params={"fuzzy_alignment_threshold": 0.6}`

4. **特殊字符或大小写问题**
   - 虽然有归一化处理，但某些特殊字符可能导致问题
   - 检查 `extraction_text` 中是否有不可见字符

**调试方法**:
```python
# 查看对齐失败的 extractions
failed = [e for e in result.extractions if e.alignment_status is None]
for e in failed:
    print(f"Failed: class={e.extraction_class}, text={e.extraction_text}")
```

---

### Q2: 为什么 schema 里的 Optional 字段没被抽取？

**LangExtract 没有传统的 "Optional" 概念**。

LangExtract 的 schema 是 **example-driven** 的，不是类型驱动的。这意味着：

1. **schema 从 examples 推断**
   - 如果你在 examples 中定义了某个 extraction_class，LLM 会被引导去抽取这类实体
   - 但这不是强制的——LLM 可能抽取也可能不抽取

2. **没有 "必填/可选" 标记**
   - 传统 Pydantic 模型有 `Optional[]` 或 `required=True/False`
   - LangExtract 没有这个机制

3. **如何控制抽取行为**
   - 通过 `prompt_description` 描述应该抽取什么
   - 通过 `examples` 展示抽取模式
   - 如果某些实体经常被遗漏，增加更多相关 examples

**注意**: 如果 LLM 没有抽取某个实体，结果中不会有对应的 Extraction 对象（值为 null 或空字符串也不会被表示）。

---

### Q3: 为什么我的输出解析失败了？

**常见场景**:

1. **LLM 没有返回 JSON/YAML 格式**
   - 某些模型可能忽略格式指令，返回自然语言
   - 解决方案：
     - 确保 examples 格式正确
     - 使用支持 schema constraints 的模型（如 Gemini）
     - 检查 `use_schema_constraints=True`（默认）

2. **多个围栏块冲突**
   - LLM 可能返回多个 ```json 块
   - 检查 `strict_fences` 设置（默认 False，取第一个有效块）

3. **推理模型的 <think> 标签**
   - DeepSeek-R1, QwQ 等模型会在 JSON 前输出思考过程
   - LangExtract 会自动处理（非 strict 模式），但如果格式太复杂可能失败

4. **缺少 `extractions` wrapper**
   - 某些模型可能直接返回 `[...]` 而不是 `{"extractions": [...]}`
   - 默认 `allow_top_level_list=True` 会处理这种情况

**调试方法**:
```python
# 使用 debug=True 查看原始输出
result = lx.extract(
    ...,
    debug=True,  # 启用详细日志
)
```

---

### Q4: 为什么同一个实体会被多次抽取？

**原因**:

1. **跨 chunk 边界**
   - 一个实体可能出现在多个 chunks 中（如果 `context_window_chars` 包含了它）
   - LangExtract 目前没有自动去重

2. **多次抽取 (`extraction_passes > 1`)**
   - 虽然多次抽取会合并非重叠结果，但如果同一个实体在不同位置有相似文本，可能被多次抽取

3. **LLM 自身的不稳定性**
   - 即使是相同的 prompt，LLM 也可能返回略有不同的结果

**解决方案**（用户自行处理）:
```python
# 基于 char_interval 去重
def deduplicate(extractions):
    seen = set()
    result = []
    for e in extractions:
        if e.char_interval is None:
            continue
        key = (e.extraction_class, e.char_interval.start_pos, e.char_interval.end_pos)
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result
```

---

### Q5: `max_char_buffer` 应该设多大？

**考虑因素**:

1. **模型上下文窗口**
   - `max_char_buffer` 应该远小于模型的最大 token 限制
   - 因为 prompt 本身（description + examples）也占用 tokens

2. **抽取精度 vs API 成本**
   - 较小的 `max_char_buffer` → 更多 chunks → 更多 API 调用 → 更高成本，但可能更准确
   - 较大的 `max_char_buffer` → 更少 chunks → 更低成本，但可能遗漏信息

3. **经验建议**
   - 默认 `1000` 是一个平衡值
   - 简单任务（如抽取人名）可以用较大值（如 `2000-3000`）
   - 复杂任务（如关系抽取）建议用较小值（如 `500-1000`）

4. **与 token 数量的关系**
   - `max_char_buffer` 是字符数，不是 token 数
   - 粗略估计：英文 ~1 token = 4 chars，中文 ~1 token = 2 chars

**配置示例**:
```python
# 高精度模式
result = lx.extract(
    ...,
    max_char_buffer=500,      # 较小 chunk
    extraction_passes=3,       # 多次抽取提高召回
)

# 低成本模式
result = lx.extract(
    ...,
    max_char_buffer=2000,     # 较大 chunk
    extraction_passes=1,      # 单次抽取
)
```

---

## 文档 TODO

在编写本文档过程中，发现以下代码注释或文档可能需要改进：

### 1. `langextract/core/schema.py`

- `BaseSchema` 的 `from_examples` 方法缺少详细 docstring，说明如何从 examples 推断 schema
- `FormatModeSchema` 的 `requires_raw_output` 属性的行为在不同 provider 之间的差异需要更清晰的说明

### 2. `langextract/resolver.py`

- `WordAligner.align_extractions` 方法的参数 `delim` 的选择理由（为什么是 `\u241F`）缺少注释
- `_accept_lcs_match` 中的双重门控（coverage + density）的设计 rationale 可以补充说明

### 3. `langextract/chunking.py`

- `ChunkIterator` 中分块策略的设计选择（为什么优先换行 > 句子 > token）缺少高层文档
- `broken_sentence` 标志的使用场景需要更清晰的注释

### 4. `langextract/core/format_handler.py`

- `parse_output` 方法中各种兼容路径（wrapper vs 非 wrapper, strict vs 非 strict）的决策树可以用注释说明
- `_THINK_TAG_RE` 的存在理由（支持哪些模型）可以补充

### 5. Public API docstring

- `lx.extract()` 的 docstring 很详细，但 `resolver_params` 中的各个对齐参数可以增加更详细的说明
- 建议在 docstring 中添加对齐参数的默认值和推荐范围

---

## 附录：核心类关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           核心类关系图                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │ ExampleData  │────▶│  Extraction  │────▶│  CharInterval │              │
│  │  (示例数据)   │     │  (抽取结果)   │     │  (字符区间)   │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
│         │                    │                                              │
│         │                    ▼                                              │
│         │           ┌──────────────┐                                       │
│         │           │TokenInterval │                                       │
│         │           │  (token 区间) │                                       │
│         │           └──────────────┘                                       │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         extract() 入口函数                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │ │
│  │  │  Annotator  │  │  Resolver   │  │FormatHandler│                   │ │
│  │  │  (协调器)   │  │  (解析对齐)  │  │ (格式处理)  │                   │ │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘                   │ │
│  │         │                │                                              │ │
│  │         ▼                ▼                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐                                      │ │
│  │  │ChunkIterator│  │ WordAligner │                                      │ │
│  │  │  (分块器)   │  │  (对齐器)   │                                      │ │
│  │  └─────────────┘  └─────────────┘                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      Prompt 组装层                                     │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐                │ │
│  │  │PromptTemplateStruct │  │  QAPromptGenerator      │                │ │
│  │  │    (模板数据)        │  │     (prompt 生成器)     │                │ │
│  │  └──────────┬──────────┘  └────────────┬────────────┘                │ │
│  │             │                           │                              │ │
│  │             ▼                           ▼                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐         │ │
│  │  │          ContextAwarePromptBuilder                       │         │ │
│  │  │      (支持跨 chunk 上下文的 prompt builder)              │         │ │
│  │  └─────────────────────────────────────────────────────────┘         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      Schema 层                                          │ │
│  │  ┌──────────────┐                                                      │ │
│  │  │  BaseSchema  │ (抽象基类)                                          │ │
│  │  └──────┬───────┘                                                      │ │
│  │         │                                                              │ │
│  │         ▼                                                              │ │
│  │  ┌──────────────────┐                                                  │ │
│  │  │ FormatModeSchema │ (当前主要实现: JSON/YAML 格式约束)              │ │
│  │  └──────────────────┘                                                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19