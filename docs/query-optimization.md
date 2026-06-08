# 检索前查询侧优化 — 修改总结

在用户原始查询与向量检索之间插入一个可配置的查询处理管道，通过清洗、改写、分解、扩展和 HyDE 五种策略提升检索命中率和回答质量。所有策略默认关闭，完全向后兼容。

---

## 一、架构概览

```
用户输入 (前端)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  🆕 backend/services/query_service.py   ← 核心新建   │
│  查询清洗 / 改写 / 分解 / 扩展 / HyDE                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ✏️ backend/services/search_service.py  ← 核心改造   │
│  新增 multi_search() 并行检索 + RRF 结果融合          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ✏️ backend/main.py                    ← 接口改造     │
│  /qa, /search 端点接入查询处理层                      │
└─────────────────────────────────────────────────────┘
    │
    ├── ✏️ backend/schemas.py             ← 数据模型    │
    ├── ✏️ backend/services/generation_service.py       │
    ├── ✏️ backend/services/prompt_service.py           │
    ├── ✏️ frontend/src/pages/QA.jsx      ← 前端界面    │
    └── ✏️ frontend/src/pages/Search.jsx                │
```

### 向后兼容设计

- 所有新增参数均为可选，默认值等价于"不启用"
- 不传 `query_strategies` 或传 `[]` → 100% 走原路径
- 已有端点 `/import-to-course`、`/embed`、`/index` 等零影响

---

## 二、新建 `backend/services/query_service.py`

查询处理核心模块，提供 5 种策略及统一的编排入口。

### 5 种策略

| 策略 | 方法 | LLM | 功能 |
|------|------|:---:|------|
| **clean** | `clean(query)` | 否 | 规则去噪：全角转半角 → 去礼貌前缀（"请问""我想问"）→ 去句末语气词（"啊""呢""吗"）→ 合并空白 |
| **rewrite** | `rewrite(query, ...)` | 是 | LLM 将口语化问题改写为接近文档书面风格的精确检索语句 |
| **decompose** | `decompose(query, ...)` | 是 | LLM 将复杂问题拆解为 2-5 个独立可检索的子问题 |
| **expand** | `expand(query, ...)` | 是 | LLM 生成 3-5 个语义相近但表述不同的查询变体 |
| **hyde** | `hyde_document(query, ...)` | 是 | LLM 生成 100-300 字假设文档片段，用文档片段替代原始 query 做向量检索 |

### 数据结构

```python
@dataclass
class ProcessedQuery:
    original: str                  # 原始用户输入
    queries: List[str]             # 最终用于检索的查询列表（去重后）
    strategies_applied: List[str]  # 实际应用的策略
    intermediates: dict            # 中间结果（用于调试/日志）
```

### 策略编排 (`process()`)

- 按 `strategies` 列表顺序依次执行（管道模式）
- `clean → rewrite → decompose → expand → hyde` 逐步处理后统一去重
- 需要 LLM 的策略共用 `rewrite_model_provider` / `rewrite_model_name` / `api_key`
- `GenerationService` 通过 `TYPE_CHECKING` + property 懒加载避免循环导入

---

## 三、修改 `backend/services/prompt_service.py`

在原有 6 种生成任务类型之外，新增 4 种查询优化专用 TaskType：

| TaskType | 用途 | temperature | max_tokens |
|----------|------|:----------:|:----------:|
| `QUERY_REWRITE` | 查询改写 | 0.1 | 256 |
| `QUERY_DECOMPOSE` | 问题分解 | 0.2 | 512 |
| `QUERY_EXPAND` | 查询扩展 | 0.6 | 512 |
| `HYDE` | 假设文档生成 | 0.5 | 512 |

- 不参与 `detect_task_type()` 自动检测，仅由 `QueryService` 显式调用
- 每种模板包含标准版和 small-model 版两套 prompt

---

## 四、修改 `backend/services/generation_service.py`

新增 `generate_raw()` 方法 — 通用 LLM 调用接口：

- 接收 `system_msg` + `user_msg` + `generation_params`，直接调用 provider 内部方法
- 与 `generate()` 不同：不构建 RAG 上下文、不拼接 `[Source N]`、不保存文件
- 支持 huggingface / openai / aliyun / deepseek 全部 4 个 provider

---

## 五、修改 `backend/services/search_service.py`

新增 `multi_search()` 方法 — 多查询并行检索 + RRF 融合：

- 对 `queries` 中的每个查询并行执行 `search()`（`asyncio.gather`）
- RRF (Reciprocal Rank Fusion)：`score = sum(1 / (k + rank_i))`，k 默认 60
- 以 text 的 MD5 哈希去重，相同文档的 RRF 分数累加
- 单查询（`len(queries) == 1`）直接走原 `search()`，零额外开销
- 未知融合策略自动回退到 RRF

新增 `import asyncio`。

---

## 六、修改 `backend/schemas.py`

`QARequest` 和 `SearchRequest` 各新增 3 个可选字段：

```python
query_strategies: List[str] = Field(default=[])          # 策略列表，空=不启用
rewrite_model_provider: str = Field(default="deepseek")   # 查询优化 LLM
rewrite_model_name: str = Field(default="deepseek-v3")    # 查询优化模型
```

---

## 七、修改 `backend/main.py`

- **`/qa` 端点**：`if data.query_strategies:` → `QueryService.process()` → 多查询走 `multi_search()`，单查询走原 `search()`；生成回答仍用原始 query
- **`/search` 端点**：同样接入 `QueryService.process()` + `multi_search()`
- **新增 `GET /query-strategies`**：返回 5 种策略元数据（value / label / description / requires_llm）

---

## 八、修改前端

### `frontend/src/pages/QA.jsx`

- 新增可折叠的 **🔍 查询优化** 面板，位于"任务类型"和"提问"之间
- 从 `/query-strategies` 动态获取策略列表，渲染为复选框
- 每种策略显示 LLM/无开销 标记和描述文字
- 选择需要 LLM 的策略时自动展开提供商和模型选择器
- 折叠状态显示已勾选策略数量

### `frontend/src/pages/Search.jsx`

- 检索控件区新增简洁版查询优化面板（折叠按钮 + 复选框）
- 同步传递 `query_strategies` 到 `/search` 接口

---

## 九、Python 3.14 兼容性懒加载修复

**问题**：Python 3.14 缺少 `llama-index`（ONNX Runtime）、`unstructured`、`pymilvus`（gRPC cygrpc）的原生扩展预编译 wheel，导致模块导入时 DLL 崩溃或长时间阻塞。

**修复策略**：将不必要的模块级顶层导入改为方法内懒加载，确保服务器启动路径（`/qa`、`/search`）不触发这些模块。

### `backend/services/chunking_service.py`

- 移除顶层 `from llama_index.core.node_parser import SemanticSplitterNodeParser`
- 移除顶层 `from llama_index.core.schema import Document as LlamaDocument`
- 移除顶层 `from llama_index.embeddings.huggingface import HuggingFaceEmbedding`
- `_get_embed_model()` 内懒加载 `HuggingFaceEmbedding`（仅 semantic chunking 触发）
- `_semantic_chunks()` 内懒加载 `SemanticSplitterNodeParser` + `LlamaDocument`
- `from langchain.text_splitter` → `from langchain_text_splitters`（适配新版 langchain）

### `backend/services/loading_service.py`

- 移除顶层 `from unstructured.partition.pdf import partition_pdf`
- `_load_with_unstructured()` 内懒加载（仅 unstructured 加载方式触发）

### `backend/services/vector_store_service.py`

- 移除顶层全部 `pymilvus` 导入
- `_index_to_milvus()` 内懒加载
- `list_collections()` / `delete_collection()` / `get_collection_info()` 在 Milvus 分支内懒加载

### `backend/services/search_service.py`

- 移除顶层全部 `pymilvus` 导入
- `search()` 的 `finally` 块中 `connections.disconnect()` 改为 try/except 包裹的懒加载

### `backend/requirements_win.txt`

- `nltk==3.9.1` → `nltk>=3.9.3`
- `pydantic==2.8.2` → `pydantic>=2.8.2`

---

## 十、文件改动清单

| # | 文件 | 改动类型 |
|---|------|---------|
| 1 | `backend/services/query_service.py` | 🆕 新建 |
| 2 | `backend/services/prompt_service.py` | ✏️ 新增 4 个 TaskType + 模板 |
| 3 | `backend/services/generation_service.py` | ✏️ 新增 `generate_raw()` |
| 4 | `backend/services/search_service.py` | ✏️ 新增 `multi_search()` + 懒加载 |
| 5 | `backend/schemas.py` | ✏️ 新增可选查询策略字段 |
| 6 | `backend/main.py` | ✏️ /qa、/search 接入优化 + /query-strategies |
| 7 | `frontend/src/pages/QA.jsx` | ✏️ 查询优化面板 |
| 8 | `frontend/src/pages/Search.jsx` | ✏️ 查询优化选项 |
| 9 | `backend/services/chunking_service.py` | 🔧 懒加载修复 |
| 10 | `backend/services/loading_service.py` | 🔧 懒加载修复 |
| 11 | `backend/services/vector_store_service.py` | 🔧 懒加载修复 |
| 12 | `backend/requirements_win.txt` | 🔧 版本兼容修复 |
