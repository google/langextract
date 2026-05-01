# 输出解析与实体对齐

LLM 返回的原始文本需要经过解析才能转换为结构化的 `Extraction` 对象。解析过程由 `FormatHandler` 和 `Resolver` 协同完成。

实体对齐是 LangExtract 的核心能力之一——它将 LLM 抽取出的文本片段回溯到原文中的精确字符位置。这使得抽取结果可验证、可可视化。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [Prompt 组装](prompt.md)**
- **→ [长文档分块](chunking.md)**

---

## 目录

- [输出解析](#输出解析)
  - [解析流程](#解析流程)
  - [关键代码解析](#关键代码解析)
  - [格式错误时的 Fallback 策略](#格式错误时的-fallback-策略)
- [实体对齐](#实体对齐)
  - [对齐流程](#对齐流程)
  - [关键代码解析](#关键代码解析-1)
  - [对齐状态说明](#对齐状态说明)
  - [对齐失败时的处理](#对齐失败时的处理)
  - [对齐参数配置](#对齐参数配置)

---

## 输出解析

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

#### 3. `<think>` 标签容错 (`_parse_with_fallback`)

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

## 实体对齐

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

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19
