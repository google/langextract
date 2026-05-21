# 架构文档 TODO

本文档记录在编写架构文档过程中发现的代码注释或文档改进点。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [Prompt 组装](prompt.md)**
- **→ [输出解析与实体对齐](alignment.md)**
- **→ [长文档分块](chunking.md)**

---

## 目录

- [待办事项列表](#待办事项列表)
- [Issue 模板](#issue-模板)

---

## 待办事项列表

| ID | 优先级 | 状态 | 标题 | 关联文件 |
|----|--------|------|------|----------|
| 1 | P1 | open | `BaseSchema.from_examples()` 方法缺少详细 docstring | `langextract/core/schema.py` |
| 2 | P2 | open | `FormatModeSchema.requires_raw_output` 行为差异说明 | `langextract/core/schema.py` |
| 3 | P1 | open | `WordAligner.align_extractions()` 中 `delim` 选择理由注释 | `langextract/resolver.py` |
| 4 | P2 | open | `_accept_lcs_match` 双重门控设计 rationale | `langextract/resolver.py` |
| 5 | P2 | open | `ChunkIterator` 分块策略设计选择高层文档 | `langextract/chunking.py` |
| 6 | P1 | open | `parse_output` 兼容路径决策树注释 | `langextract/core/format_handler.py` |
| 7 | P2 | open | `_THINK_TAG_RE` 存在理由补充 | `langextract/core/format_handler.py` |
| 8 | P1 | open | `resolver_params` 对齐参数详细说明 | `langextract/extraction.py` |

---

### 详细说明

#### TODO 1: `BaseSchema.from_examples()` 方法缺少详细 docstring

| 字段 | 内容 |
|------|------|
| **优先级** | P1 |
| **状态** | open |
| **关联文件** | `langextract/core/schema.py` |
| **当前问题** | `from_examples` 方法缺少详细 docstring，说明如何从 examples 推断 schema |
| **建议改进** | 补充 docstring，说明：1) 如何提取 extraction_class；2) 如何推断值类型；3) 如何处理 attributes |
| **影响范围** | 新开发者理解 schema 推断机制的入口点 |

**Issue 模板**:

```
### 标题: 补充 BaseSchema.from_examples() 方法的 docstring

### 背景
在编写架构文档时，发现 `langextract/core/schema.py` 中的 `BaseSchema.from_examples()` 抽象方法缺少详细的 docstring。这个方法是理解 LangExtract example-driven schema 推断机制的关键入口点。

### 问题描述
- 当前 `from_examples` 只有一个简单的签名说明
- 没有说明从 examples 推断 schema 的具体逻辑：
  - 如何从 `Extraction.extraction_class` 提取实体类型？
  - 如何分析 `extraction_text` 的值类型？
  - 如何处理 `attributes` 字段？

### 验收标准
- [ ] `BaseSchema.from_examples()` 方法有完整的 docstring
- [ ] docstring 包含参数说明（`examples_data`、`attribute_suffix`）
- [ ] docstring 包含返回值说明（`BaseSchema` 实例）
- [ ] docstring 包含 schema 推断的逻辑说明
```

---

#### TODO 2: `FormatModeSchema.requires_raw_output` 行为差异说明

| 字段 | 内容 |
|------|------|
| **优先级** | P2 |
| **状态** | open |
| **关联文件** | `langextract/core/schema.py` |
| **当前问题** | `requires_raw_output` 属性的行为在不同 provider 之间的差异需要更清晰的说明 |
| **建议改进** | 补充说明：1) JSON 模式 vs YAML 模式的差异；2) 不同 provider 如何处理 `requires_raw_output` |
| **影响范围** | 开发者理解不同 provider 的行为差异 |

**Issue 模板**:

```
### 标题: 补充 FormatModeSchema.requires_raw_output 行为说明

### 背景
`FormatModeSchema.requires_raw_output` 属性决定了输出是否需要围栏标记。但这个行为在 JSON 模式和 YAML 模式下有所不同，且与具体 provider 的实现有关。

### 问题描述
- 当前 `requires_raw_output` 的实现是：`self._format == "json"`
- 这意味着 JSON 格式输出原始 JSON（无围栏），YAML 格式需要围栏
- 但不同 provider（Gemini、OpenAI、Ollama）对这个属性的处理可能不同

### 验收标准
- [ ] 在 `FormatModeSchema` 类的 docstring 中补充 `requires_raw_output` 的行为说明
- [ ] 说明 JSON 模式和 YAML 模式的差异
- [ ] 说明不同 provider 可能的实现差异
```

---

#### TODO 3: `WordAligner.align_extractions()` 中 `delim` 选择理由注释

| 字段 | 内容 |
|------|------|
| **优先级** | P1 |
| **状态** | open |
| **关联文件** | `langextract/resolver.py` |
| **当前问题** | `WordAligner.align_extractions` 方法中使用 `\u241F` 作为分隔符，但缺少为什么选择这个字符的注释 |
| **建议改进** | 补充注释说明：1) 为什么选择 `\u241F`（Unicode 单元分隔符）；2) 为什么不使用其他分隔符 |
| **影响范围** | 理解精确匹配机制的关键设计选择 |

**Issue 模板**:

```
### 标题: 补充 WordAligner.align_extractions 中 delim 选择理由的注释

### 背景
在 `langextract/resolver.py` 的 `WordAligner.align_extractions()` 方法中，使用 `\u241F`（Unicode 单元分隔符）作为分隔符连接多个 extraction_text。这个选择是精确匹配机制中的关键设计。

### 问题描述
- 分隔符选择：`delim = "\u241F"`
- 代码中没有注释说明为什么选择这个字符
- 新开发者可能不理解：
  - 为什么是 `\u241F` 而不是空格或其他字符？
  - 这个字符有什么特殊属性？

### 验收标准
- [ ] 在 `delim = "\u241F"` 语句前添加注释
- [ ] 说明这是 Unicode 单元分隔符（Unit Separator）
- [ ] 说明选择理由：不会出现在正常文本中，确保精确匹配的准确性
```

---

#### TODO 4: `_accept_lcs_match` 双重门控设计 rationale

| 字段 | 内容 |
|------|------|
| **优先级** | P2 |
| **状态** | open |
| **关联文件** | `langextract/resolver.py` |
| **当前问题** | `_accept_lcs_match` 中的双重门控（coverage + density）的设计 rationale 可以补充说明 |
| **建议改进** | 补充注释说明：1) 为什么需要双重门控；2) 每个门控解决什么问题；3) 默认值（0.75, 1/3）的选择理由 |
| **影响范围** | 理解模糊匹配算法的核心逻辑 |

**Issue 模板**:

```
### 标题: 补充 _accept_lcs_match 双重门控的设计 rationale

### 背景
`_accept_lcs_match` 函数实现了 LCS 模糊匹配的双重门控验证：
1. Coverage Gate: 匹配的 token 数 >= 阈值比例
2. Density Gate: 匹配的 token 数 / 匹配区间长度 >= min_density

### 问题描述
- 当前代码只有简单的实现逻辑
- 没有说明为什么需要双重门控
- 没有说明默认值（0.75, 1/3）的选择理由

### 验收标准
- [ ] 在 `_accept_lcs_match` 函数前添加或补充 docstring
- [ ] 说明 Coverage Gate 解决的问题（防止匹配过少 tokens）
- [ ] 说明 Density Gate 解决的问题（防止匹配的 tokens 分散在太长区间）
- [ ] 补充默认值选择的 rationale（如果有）
```

---

#### TODO 5: `ChunkIterator` 分块策略设计选择高层文档

| 字段 | 内容 |
|------|------|
| **优先级** | P2 |
| **状态** | open |
| **关联文件** | `langextract/chunking.py` |
| **当前问题** | `ChunkIterator` 中分块策略的设计选择（为什么优先换行 > 句子 > token）缺少高层文档 |
| **建议改进** | 在模块级或类级 docstring 中补充：1) 分块策略的优先级；2) 每个优先级的设计理由；3) `broken_sentence` 标志的使用场景 |
| **影响范围** | 理解长文档分块机制 |

**Issue 模板**:

```
### 标题: 补充 ChunkIterator 分块策略的高层文档

### 背景
`ChunkIterator` 实现了 LangExtract 的长文档分块逻辑，策略优先级为：
1. 优先按换行符分割（保持格式）
2. 然后按句子边界分割（保持语义）
3. 最后按 token 分割（超长句处理）

### 问题描述
- 当前代码中，分块逻辑分散在 `__next__` 方法中
- 没有统一的高层文档说明策略设计选择
- `broken_sentence` 标志的使用场景不够清晰

### 验收标准
- [ ] 在 `ChunkIterator` 类的 docstring 中补充分块策略说明
- [ ] 说明优先级：换行 > 句子 > token
- [ ] 说明每个优先级的设计理由
- [ ] 说明 `broken_sentence` 标志的作用和使用场景
```

---

#### TODO 6: `parse_output` 兼容路径决策树注释

| 字段 | 内容 |
|------|------|
| **优先级** | P1 |
| **状态** | open |
| **关联文件** | `langextract/core/format_handler.py` |
| **当前问题** | `parse_output` 方法中各种兼容路径（wrapper vs 非 wrapper, strict vs 非 strict）的决策树可以用注释说明 |
| **建议改进** | 用注释或流程图形式说明：1) strict 模式 vs 非 strict 模式的行为差异；2) wrapper 模式 vs 非 wrapper 模式的处理流程 |
| **影响范围** | 理解输出解析的容错机制 |

**Issue 模板**:

```
### 标题: 补充 parse_output 兼容路径的决策树注释

### 背景
`FormatHandler.parse_output()` 方法处理多种兼容情况：
- strict vs 非 strict 模式
- wrapper vs 非 wrapper 模式
- 顶级列表支持

### 问题描述
- 代码逻辑复杂，包含多个条件分支
- 没有统一的决策树说明
- 新开发者难以理解不同参数组合的行为

### 验收标准
- [ ] 在 `parse_output` 方法中添加决策树注释
- [ ] 说明 `strict` 参数的影响
- [ ] 说明 `use_wrapper`、`allow_top_level_list` 的交互
- [ ] 用 ASCII 图或表格形式展示参数组合的行为
```

---

#### TODO 7: `_THINK_TAG_RE` 存在理由补充

| 字段 | 内容 |
|------|------|
| **优先级** | P2 |
| **状态** | open |
| **关联文件** | `langextract/core/format_handler.py` |
| **当前问题** | `_THINK_TAG_RE` 的存在理由（支持哪些模型）可以补充 |
| **建议改进** | 补充注释说明：1) 正则表达式的用途；2) 哪些模型会输出 `<think>` 标签；3) 这个容错机制的设计背景 |
| **影响范围** | 理解推理模型的输出格式处理 |

**Issue 模板**:

```
### 标题: 补充 _THINK_TAG_RE 存在理由的注释

### 背景
`_THINK_TAG_RE` 正则表达式用于处理推理模型（如 DeepSeek-R1、QwQ）的输出，这些模型会在 JSON 输出前先输出思考过程。

### 问题描述
- 当前定义：`_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*")`
- 没有注释说明：
  - 这个正则的用途是什么？
  - 哪些模型会输出 `<think>` 标签？
  - 这个容错机制是何时、为何引入的？

### 验收标准
- [ ] 在 `_THINK_TAG_RE` 定义前添加注释
- [ ] 说明支持的模型（DeepSeek-R1、QwQ 等推理模型）
- [ ] 说明这些模型的输出特点（先思考后输出）
- [ ] 说明 `_parse_with_fallback` 中如何使用这个正则
```

---

#### TODO 8: `resolver_params` 对齐参数详细说明

| 字段 | 内容 |
|------|------|
| **优先级** | P1 |
| **状态** | open |
| **关联文件** | `langextract/extraction.py` |
| **当前问题** | `lx.extract()` 的 docstring 很详细，但 `resolver_params` 中的各个对齐参数可以增加更详细的说明 |
| **建议改进** | 在 docstring 中补充：1) 对齐参数的默认值；2) 推荐范围；3) 调整建议（何时调高调低） |
| **影响范围** | Public API 文档，用户快速上手 |

**Issue 模板**:

```
### 标题: 补充 resolver_params 中对齐参数的详细说明

### 背景
`extract()` 函数的 `resolver_params` 参数允许用户配置对齐行为，包括：
- `enable_fuzzy_alignment`
- `fuzzy_alignment_threshold`
- `fuzzy_alignment_min_density`
- `fuzzy_alignment_algorithm`
- `accept_match_lesser`
- `suppress_parse_errors`

### 问题描述
- 当前 docstring 中缺少这些参数的详细说明
- 用户可能不知道：
  - 默认值是什么？
  - 推荐调整范围是什么？
  - 何时应该调整这些参数？

### 验收标准
- [ ] 在 `extract()` 函数的 docstring 中补充 `resolver_params` 的详细说明
- [ ] 列出所有支持的对齐参数及其默认值
- [ ] 给出推荐范围和调整建议
- [ ] 可以考虑添加一个配置示例表格
```

---

## Issue 模板

为方便提交 Issue，以下是标准模板：

```markdown
### 标题: [简短描述问题]

### 背景
[问题的背景信息，为什么需要改进]

### 问题描述
[具体问题是什么，当前代码的状态]

### 验收标准
- [ ] 可验证的标准 1
- [ ] 可验证的标准 2
- [ ] 可验证的标准 3
```

---

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19
