# LangExtract 架构概览

本文档系统梳理 LangExtract 的核心内部机制，帮助开发者理解信息抽取的完整流程。

---

## 快速导航

| 文档 | 内容概述 | 适用对象 |
|------|----------|----------|
| [Schema 设计](schema.md) | 字段类型、实体定义、JSON 与 Python 类对应关系 | 需要理解如何定义抽取任务的开发者 |
| [Prompt 组装](prompt.md) | 从 schema 到最终 prompt 的完整流程、Q:A 格式 | 需要定制 prompt 或理解 LLM 交互的开发者 |
| [输出解析与实体对齐](alignment.md) | LLM 输出解析、容错策略、精确匹配 + LCS 模糊匹配算法 | 需要处理对齐失败或调试抽取结果的开发者 |
| [长文档分块](chunking.md) | Chunk 策略、context_window_chars、跨 chunk 合并 | 需要处理长文档或优化分块参数的开发者 |
| [文档 TODO](TODO.md) | 代码注释改进计划、优先级、Issue 模板 | 贡献者、维护者 |

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [Prompt 组装](prompt.md)**
- **→ [输出解析与实体对齐](alignment.md)**
- **→ [长文档分块](chunking.md)**

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

3. **推理模型的 `<think>` 标签**
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

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_char_buffer` | 1000 | 每个 chunk 的最大字符数。调小可提高准确率但增加 API 调用。 |
| `batch_length` | 10 | 每批处理的 chunk 数量。与 `max_workers` 共同决定并行度。 |
| `max_workers` | 10 | 最大并行 worker 数。有效并行度受限于 `min(batch_length, max_workers)`。 |
| `context_window_chars` | `None` | 前一个 chunk 的上下文字符数。用于指代消解。 |
| `extraction_passes` | 1 | 抽取次数。> 1 时执行多次抽取并合并非重叠结果。 |

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
