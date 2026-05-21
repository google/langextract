# Prompt 组装

Prompt 组装是 LangExtract 的核心环节，它将用户的 `prompt_description` 和 `examples` 转换为 LLM 可理解的指令格式。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [输出解析与实体对齐](alignment.md)**
- **→ [长文档分块](chunking.md)**

---

## 目录

- [完整流程](#完整流程)
- [关键代码路径](#关键代码路径)
- [完整 Prompt 示例](#完整-prompt-示例)

---

## 完整流程

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
│  │  帮助解决指代消解问题: "She" → "Dr. Sarah Johnson"                   │  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 关键代码路径

### 1. 示例格式化 (`format_extraction_example`)

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

### 2. Prompt 渲染 (`QAPromptGenerator.render`)

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

### 3. 示例格式化 (`format_example_as_text`)

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

### 4. 跨 chunk 上下文 (`ContextAwarePromptBuilder`)

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

---

## 完整 Prompt 示例

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

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19
