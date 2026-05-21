# Provider 层设计

LangExtract 的 provider 层提供了统一的 LLM 抽象接口，支持多种模型后端（Gemini、OpenAI、Ollama 等），并通过插件机制支持第三方扩展。

---

## 架构导航

- **← [返回概览](overview.md)**
- **→ [Schema 设计](schema.md)**
- **→ [Prompt 组装](prompt.md)**

---

## 目录

- [LLMProvider 契约](#llmprovider-契约)
- [内置 Provider 对比](#内置-provider-对比)
- [ProviderRegistry 与模型解析](#providerregistry-与模型解析)
- [Entry Points 插件机制](#entry-points-插件机制)
- [MockProvider 使用指南](#mockprovider-使用指南)
- [Context Manager 资源管理](#context-manager-资源管理)
- [高层 API 与 Registry 关系](#高层-api-与-registry-关系)

---

## LLMProvider 契约

`LLMProvider` 是所有 LLM provider 必须实现的抽象基类，定义了统一的接口契约。

### 核心接口

```python
# langextract/core/base_model.py:64-193
class LLMProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider 名称标识符，如 "gemini", "openai", "mock"。"""
    
    @property
    @abc.abstractmethod
    def supported_models(self) -> TypingSequence[str]:
        """支持的模型 ID 正则表达式模式列表。"""
    
    @abc.abstractmethod
    def generate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """同步生成文本。"""
    
    @abc.abstractmethod
    async def agenerate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """异步生成文本。"""
    
    @abc.abstractmethod
    def close(self) -> None:
        """清理资源（关闭 HTTP 客户端、连接池等）。"""
```

### 返回类型

**GenerateResult**: 封装生成结果的数据类

```python
# langextract/core/base_model.py:49-62
@dataclasses.dataclass
class GenerateResult:
    text: str              # 生成的文本输出
    usage: Usage | None    # 可选的 token 使用量信息
    raw_response: Any      # 原始 API 响应
```

**Usage**: Token 使用量信息

```python
# langextract/core/base_model.py:34-47
@dataclasses.dataclass
class Usage:
    input_tokens: int | None   # 输入 prompt 的 token 数
    output_tokens: int | None  # 输出的 token 数
    total_tokens: int | None   # 总 token 数
```

### BaseLanguageModel 默认实现

`BaseLanguageModel` 继承 `LLMProvider`，提供了默认实现以保持向后兼容：

| 方法 | 默认实现 | 说明 |
|------|----------|------|
| `name` | 从类名推导 | 去掉 "LanguageModel" 后缀并小写 |
| `supported_models` | 返回空列表 | Provider 应覆盖此属性 |
| `generate` | 包装 `infer()` | 基于旧的 `infer()` API |
| `agenerate` | 线程池包装 | 在 `ThreadPoolExecutor` 中运行 `generate()` |
| `close` | 空操作 | Provider 应覆盖以清理资源 |

---

## 内置 Provider 对比

| Provider | 名称 | 模型模式 | Schema 支持 | 异步支持 | 依赖包 |
|----------|------|----------|-------------|----------|--------|
| Gemini | `gemini` | `^gemini`, `^models/gemini` | 完整支持 (response_schema) | 线程池包装 | `google-genai` |
| OpenAI | `openai` | `^gpt`, `^o1`, `^o3`, `^o4` | 格式支持 (JSON mode) | 线程池包装 | `openai` |
| Ollama | `ollama` | `^ollama:` | 格式支持 | 线程池包装 | 无 (HTTP) |
| Mock | `mock` | `^mock$`, `^mock-` | 无 (测试用) | 原生同步 | 无 |

### Gemini Provider

```python
# langextract/providers/gemini.py
@router.register(
    *patterns.GEMINI_PATTERNS,
    priority=patterns.GEMINI_PRIORITY,
)
class GeminiLanguageModel(BaseLanguageModel):
    def __init__(
        self,
        model_id: str = 'gemini-2.5-flash',
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        temperature: float = 0.0,
        max_workers: int = 10,
        **kwargs,
    ): ...
```

**Key Features**:
- 完整的 `GeminiSchema` 结构化输出支持
- 支持 Vertex AI (Enterprise) 模式
- 内置 Batch API 支持
- `google-genai` 官方 SDK

### OpenAI Provider

```python
# langextract/providers/openai.py
@router.register(
    *patterns.OPENAI_PATTERNS,
    priority=patterns.OPENAI_PRIORITY,
)
class OpenAILanguageModel(BaseLanguageModel):
    def __init__(
        self,
        model_id: str = 'gpt-4o-mini',
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs,
    ): ...
```

**Key Features**:
- JSON mode 格式约束
- 支持 `base_url` 自定义端点（兼容 Azure OpenAI、本地兼容 API）
- `openai` 官方 SDK

---

## ProviderRegistry 与模型解析

`ProviderRegistry` 提供了中心化的 provider 注册和查找机制。

### 基本用法

```python
from langextract.providers import ProviderRegistry

# 获取全局 registry 实例
registry = ProviderRegistry.get_global()

# 按名称获取 provider 类
gemini_cls = registry.get("gemini")
mock_cls = registry.get("mock")

# 按模型 ID 解析 provider
provider_cls = registry.get_for_model("gemini-2.5-flash")

# 注册自定义 provider
registry.register(MyCustomProvider, patterns=[r"^my-model"])

# 清空 registry (用于测试)
registry.clear()
```

### 模型解析机制

模型解析基于正则表达式模式匹配，**高优先级**的 provider 优先匹配：

```
模型 ID: "gemini-2.5-flash"
     │
     ▼
┌───────────────────────────────────────────────┐
│          Router Pattern Matching              │
├───────────────────────────────────────────────┤
│  1. 遍历所有注册的 pattern (按优先级排序)      │
│  2. 检查 `re.match(pattern, model_id)`       │
│  3. 返回第一个匹配的 provider                  │
└───────────────────────────────────────────────┘
     │
     ▼
返回: GeminiLanguageModel
```

### Priority 机制

```python
# 高优先级会覆盖低优先级
@router.register(r"^gemini", priority=0)    # 低优先级
class DefaultGeminiProvider(...): ...

@router.register(r"^gemini", priority=100)  # 高优先级
class CustomGeminiProvider(...): ...

# 解析 "gemini-pro" → 返回 CustomGeminiProvider
```

**默认优先级值**:

| Provider | 默认 Priority |
|----------|---------------|
| 内置 Provider (Gemini/OpenAI) | 0 |
| MockProvider | 100 |
| 第三方插件 (默认) | 20 |

---

## Entry Points 插件机制

LangExtract 使用 Python 的 `entry_points` 机制实现第三方 provider 的自动发现和注册。

### 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│              Entry Points 自动发现流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 用户安装第三方包: pip install langextract-myprovider   │
│                                                             │
│  2. ProviderRegistry 初始化时调用 load_plugins_once()      │
│                                                             │
│  3. 查询 importlib.metadata.entry_points()                 │
│     ── 筛选 group="langextract.providers" ──               │
│                                                             │
│  4. 加载 entry point 指向的类                              │
│     ── 自动调用 @router.register 注册 ──                    │
│                                                             │
│  5. 完成！用户可通过 registry.get("myprovider") 使用       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 插件开发步骤

#### 1. 创建 Provider 类

```python
# my_package/provider.py
from langextract.core import base_model
from langextract.core import types as core_types
from langextract.providers import router

@router.register(
    r"^my-model", r"^myprovider:",
    priority=50,
)
class MyProviderLanguageModel(base_model.BaseLanguageModel):
    model_id: str = "my-model-default"
    
    @property
    def name(self) -> str:
        return "myprovider"
    
    @property
    def supported_models(self) -> list[str]:
        return [r"^my-model", r"^myprovider:"]
    
    @classmethod
    def get_model_patterns(cls) -> list[str]:
        """可选：静态方法返回支持的模型模式"""
        return [r"^my-model", r"^myprovider:"]
    
    def __init__(self, model_id: str, api_key: str | None = None, **kwargs):
        super().__init__()
        self.model_id = model_id
        self._client = self._init_client(api_key)
    
    def _init_client(self, api_key: str | None):
        # 初始化你的 API 客户端
        ...
    
    def infer(self, batch_prompts: Sequence[str], **kwargs):
        # 实现推理逻辑
        for prompt in batch_prompts:
            response = self._call_api(prompt, **kwargs)
            yield [core_types.ScoredOutput(score=1.0, output=response)]
    
    def close(self) -> None:
        # 清理资源
        if hasattr(self, '_client'):
            self._client.close()
```

#### 2. 配置 pyproject.toml

```toml
[project]
name = "langextract-myprovider"
version = "0.1.0"
description = "My custom provider for LangExtract"
requires-python = ">=3.10"
dependencies = [
    "langextract>=1.2.0",
    # 你的其他依赖
]

[project.entry-points."langextract.providers"]
myprovider = "my_package.provider:MyProviderLanguageModel"

# 带 priority 后缀的语法 (可选)
# myprovider = "my_package.provider:MyProviderLanguageModel:priority=100"
```

### Priority 后缀语法

Entry point value 支持 `:priority=N` 后缀来指定注册优先级：

```toml
[project.entry-points."langextract.providers"]
# 基本语法
basic_provider = "my_pkg:BasicProvider"

# 带优先级的语法
high_priority_provider = "my_pkg:HighPriorityProvider:priority=100"
low_priority_provider = "my_pkg:LowPriorityProvider:priority=0"
```

**优先级解析规则**:
1. 如果 entry point value 有 `:priority=N` 后缀，使用该值
2. 否则使用类的 `pattern_priority` 属性（默认 20）
3. 最后使用默认插件优先级 20

### 禁用插件加载

```python
# 环境变量禁用
export LANGEXTRACT_DISABLE_PLUGINS=1

# 或在代码中设置
import os
os.environ["LANGEXTRACT_DISABLE_PLUGINS"] = "1"
```

---

## MockProvider 使用指南

`MockProvider` 是专为测试设计的 provider，不需要真实 API 调用。

### 基本用法

```python
from langextract.providers import MockProvider, ProviderRegistry

# 方式 1: 直接实例化
mock = MockProvider(fixed_response='{"result": "test"}')
result = mock.generate("Hello")
print(result.text)  # '{"result": "test"}'

# 方式 2: 通过 registry 获取 (推荐)
registry = ProviderRegistry()
mock_cls = registry.get("mock")
mock = mock_cls(fixed_response="test response")

# 方式 3: 使用响应函数
def custom_response(prompt: str, **kwargs) -> str:
    if "name" in prompt:
        return '{"name": "Alice"}'
    elif "value" in prompt:
        return '{"value": 42}'
    return '{"default": true}'

mock = MockProvider(response_fn=custom_response)
mock.generate("What is your name?")   # '{"name": "Alice"}'
mock.generate("What is the value?")  # '{"value": 42}'
```

### 与 Usage 信息

```python
from langextract.providers import MockProvider, Usage

mock = MockProvider(
    fixed_response='{"result": "ok"}',
    usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
)

result = mock.generate("Test")
assert result.usage.input_tokens == 100
assert result.usage.output_tokens == 50
```

### 作为 pytest fixture

```python
import pytest
from langextract.providers import MockProvider, ProviderRegistry

@pytest.fixture
def mock_provider():
    """Fixture 提供已配置的 MockProvider"""
    return MockProvider(fixed_response='{"test": true}')

@pytest.fixture
def registry_mock():
    """Fixture 提供带有 MockProvider 的 registry"""
    registry = ProviderRegistry()
    registry.clear()  # 重置状态
    # MockProvider 会自动注册
    return registry

def test_with_mock(mock_provider):
    result = mock_provider.generate("Hello")
    assert '"test": true' in result.text

def test_via_registry(registry_mock):
    mock_cls = registry_mock.get("mock")
    mock = mock_cls(fixed_response="custom")
    result = mock.generate("test")
    assert result.text == "custom"
```

---

## Context Manager 资源管理

`LLMProvider` 支持同步和异步两种上下文管理器，确保资源自动释放。

### 同步用法

```python
from langextract.providers import ProviderRegistry

# 方式 1: 手动管理 (容易忘记 close)
provider = ProviderRegistry.get("gemini")(api_key="...")
try:
    result = provider.generate("Hello")
finally:
    provider.close()

# 方式 2: with 语句 (推荐)
with ProviderRegistry.get("gemini")(api_key="...") as provider:
    result = provider.generate("Hello")
# provider.close() 已自动调用
```

### 异步用法

```python
import asyncio
from langextract.providers import MockProvider

async def async_usage():
    # async with 语句
    async with MockProvider(fixed_response="async test") as mock:
        result = await mock.agenerate("Hello")
        print(result.text)
    
    # mock.close() 已自动调用

asyncio.run(async_usage())
```

### 测试 close() 调用

`MockProvider.close_called` 计数器可用于验证资源清理：

```python
from langextract.providers import MockProvider

def test_context_manager_calls_close():
    mock = MockProvider()
    assert mock.close_called == 0
    
    with mock:
        result = mock.generate("Test")
        assert mock.close_called == 0  # 还在上下文中
    
    assert mock.close_called == 1  # 已退出上下文

def test_async_context_manager():
    import asyncio
    
    async def test():
        mock = MockProvider()
        async with mock:
            result = await mock.agenerate("Test")
            assert mock.close_called == 0
        assert mock.close_called == 1
    
    asyncio.run(test())
```

### 异常时的资源清理

上下文管理器保证即使发生异常，`close()` 也会被调用：

```python
from langextract.providers import MockProvider

mock = MockProvider()

try:
    with mock:
        raise ValueError("Something went wrong")
except ValueError:
    pass

assert mock.close_called == 1  # close() 仍然被调用
```

---

## 高层 API 与 Registry 关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    高层 API → Registry → Provider 调用链                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    用户调用: lx.extract()                            │   │
│  │  (langextract/extraction.py)                                         │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    factory.create_model()                             │   │
│  │  (langextract/factory.py)                                             │   │
│  │  - 解析 model_id                                                       │   │
│  │  - 查找对应的 provider 类                                              │   │
│  │  - 实例化 provider                                                     │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ProviderRegistry / router                         │   │
│  │  (langextract/providers/registry.py, router.py)                     │   │
│  │  - 管理已注册的 provider 模式                                          │   │
│  │  - 按优先级匹配模型 ID                                                  │   │
│  │  - 自动加载 entry points 插件                                          │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LLMProvider 实例                                   │   │
│  │  (GeminiLanguageModel, OpenAILanguageModel, MockProvider, ...)      │   │
│  │  - 封装真实 API 调用                                                   │   │
│  │  - 管理 HTTP 客户端资源                                                │   │
│  │  - 支持 generate() / agenerate()                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 完整调用流程示例

```python
import langextract as lx

# 用户调用高层 API
result = lx.extract(
    text_or_documents="ROMEO. But soft! What light...",
    prompt_description="Extract character names",
    examples=[...],
    config=lx.factory.ModelConfig(
        model_id="gemini-2.5-flash",  # 决定使用哪个 provider
        provider_kwargs={"api_key": "your-api-key"},
    ),
)

# 内部流程:
# 1. extract() → factory.create_model()
# 2. create_model() → registry.get_for_model("gemini-2.5-flash")
# 3. registry 匹配 pattern "^gemini" → GeminiLanguageModel
# 4. 实例化 GeminiLanguageModel(api_key="...")
# 5. 调用 provider.generate() / infer() 执行推理
```

### 直接使用 Provider

有时你可能想直接使用 provider 而不经过完整的 extract 流程：

```python
from langextract.providers import ProviderRegistry

# 获取 provider 类
gemini_cls = ProviderRegistry.get("gemini")

# 实例化
with gemini_cls(
    model_id="gemini-2.5-flash",
    api_key="your-api-key",
    temperature=0.0,
) as provider:
    # 直接调用 generate
    result = provider.generate(
        prompt="Translate 'Hello' to French",
        model="gemini-2.5-flash",
    )
    print(result.text)  # "Bonjour"
```

---

## 附录: 关键文件位置

| 文件 | 说明 |
|------|------|
| `langextract/core/base_model.py` | `LLMProvider`, `BaseLanguageModel`, `GenerateResult`, `Usage` |
| `langextract/providers/registry.py` | `ProviderRegistry`, `ProviderInfo`, `MockProvider` |
| `langextract/providers/router.py` | 底层模式匹配与注册机制 |
| `langextract/providers/__init__.py` | 导出入口、插件加载 `load_plugins_once()` |
| `langextract/providers/gemini.py` | Gemini provider 实现 |
| `langextract/providers/openai.py` | OpenAI provider 实现 |
| `tests/provider_abstraction_test.py` | Provider 层单元测试 |

---

**本文档基于代码版本**: langextract (main)
**最后更新**: 2026-06-18
