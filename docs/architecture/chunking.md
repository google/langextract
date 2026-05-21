# 长文档分块

当输入文本超过 LLM 的上下文窗口或 `max_char_buffer` 限制时，LangExtract 会将文档分割成多个 chunks 分别处理。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [Prompt 组装](prompt.md)**
- **→ [输出解析与实体对齐](alignment.md)**

---

## 目录

- [分块策略](#分块策略)
- [关键代码解析](#关键代码解析)
- [Overlap 与上下文窗口](#overlap-与上下文窗口)
- [跨 Chunk 实体合并与去重](#跨-chunk-实体合并与去重)
- [分块参数配置](#分块参数配置)

---

## 分块策略

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

---

## 关键代码解析

### 1. ChunkIterator 主逻辑 (`__next__`)

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

### 2. 句子边界检测 (`SentenceIterator`)

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

---

## Overlap 与上下文窗口

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

---

## 跨 Chunk 实体合并与去重

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

---

## 分块参数配置

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

**本文档基于代码版本**: langextract (docs/schema-design 分支)
**最后更新**: 2026-04-19
