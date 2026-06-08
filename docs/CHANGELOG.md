# RAG 项目功能扩展总结

> 基于基础版本（`ed48598 init`）之后的全部改动，涵盖 5 大功能方向和若干技术修复。

---

## 目录

1. [Prompt 工程与任务路由](#一prompt-工程与任务路由)
2. [多格式文件解析](#二多格式文件解析)
3. [数据侧分块优化](#三数据侧分块优化)
4. [检索前查询优化](#四检索前查询优化)
5. [检索后后处理优化](#五检索后后处理优化)
6. [技术修复](#六技术修复)

---

## 一、Prompt 工程与任务路由

**新建** `backend/services/prompt_service.py`，实现可配置的多任务 Prompt 模板引擎。

### 核心功能

- **6 种生成任务类型**：`qa`（问答）、`summarize`（摘要）、`analyze`（分析）、`compare`（对比）、`explain`（解释）、`creative`（创意），每种任务配有独立的角色提示词模板和生成参数。

| 任务类型  | temperature | max_tokens | 用途           |
|-----------|-------------|------------|----------------|
| qa        | 0.3         | 1024       | 基于参考资料回答 |
| summarize | 0.2         | 512        | 精炼摘要        |
| analyze   | 0.4         | 1024       | 深度分析        |
| compare   | 0.3         | 1024       | 多维度对比      |
| explain   | 0.4         | 1024       | 概念解释        |
| creative  | 0.8         | 1536       | 创意内容        |

- **任务自动检测**：`detect_task_type(query)` 基于关键词匹配从用户问题中推断任务类型。
- **小模型适配**：`is_small_model()` 检测 ≤3B 参数模型，自动使用更简短的 prompt 避免超出上下文窗口。

### 关联改动

- `generation_service.py`：集成 `PromptService`，4 个 provider（huggingface / openai / aliyun / deepseek）统一调用，生成结果保存 `detected_task` 字段用于追溯。
- `backend/main.py`：新增 `GET /task-types` 端点，供前端动态渲染下拉选择框。
- `schemas.py`：`GenerateRequest` / `QARequest` 新增 `task_type` 字段。
- `frontend/src/pages/QA.jsx` 和 `Generation.jsx`：新增任务类型下拉选择，动态从接口获取列表。

---

## 二、多格式文件解析

将原本以 PDF 为中心的输入流程扩展为通用文档输入流程。

### 支持格式

| 格式 | 处理方式 |
|------|----------|
| PDF  | 复用 PyMuPDF / PyPDF / PDFPlumber / Unstructured |
| JSON / JSONL | 自动提取结构化文本，兼容 chunks 和问答格式 |
| Markdown | 按标题分段读取 |
| TXT | 直接读取纯文本 |
| CSV | 按行提取字段内容 |
| DOCX | 提取段落、标题和表格文本（python-docx） |
| PNG / JPG 等图片 | OCR 识别图片文字（Tesseract） |

### 核心改动

- `loading_service.py`：新增 `load_document()` 通用入口，按扩展名自动分发，输出统一的 `page_map` 结构，下游分块、embedding、入库无需改动。
- `parsing_service.py`：新增 `parse_document()`，保留 `parse_pdf()` 兼容包装。
- `main.py`：`/process`、`/parse`、`/load`、`/import-to-course` 均接入通用加载逻辑。
- 前端 `LoadFile.jsx`、`ParseFile.jsx`、`ImportCourse.jsx`：文件选择框扩展为多格式，增加格式提示。
- `requirements_win.txt`：补充 `python-docx`、`pillow`、`pytesseract`、`huggingface-hub` 等依赖。

---

## 三、数据侧分块优化

在原有分块体系上，新增父子分块与 Markdown 结构感知分块两种高级策略，解决"切块太大搜不准，切块太小答不全"的矛盾。

### 两种高级策略

| 策略 | 参数值 | 原理 |
|------|--------|------|
| **父子分块** | `parent_child` | 大段落（父块）切成短句（子块）用于检索；命中子块时自动替换为父块完整上下文送入生成模型（Small-to-Big） |
| **结构化分块** | `markdown_structure` | 按 H1/H2/H3 标题层级切分，在块开头拼接 `【文档结构：第一章 > 第三节】` 前缀，保留层级信息 |

### 核心改动

- `chunking_service.py`：新增 `_parent_child_chunks()` 和 `_markdown_structure_chunks()` 方法，生成含 `parent_id`、`parent_content`、`heading_hierarchy` 的元数据。
- `search_service.py`：检索结果后处理中，检测 `parent_id` 字段并自动将子块替换为父块，维护 `seen_parent_ids` 集合去重，避免多个子块重复引入同一父块。
- `vector_store_service.py`：`_index_to_chroma()` 写入时持久化 `parent_id`、`parent_content`、`heading_hierarchy` 字段。
- `schemas.py`：`ChunkRequest` 新增 `parent_chunk_size`、`child_chunk_size` 参数；`ChunkMetadata` 和 `SearchResultMeta` 新增对应可选字段。

---

## 四、检索前查询优化

在用户原始查询与向量检索之间插入可配置的查询处理管道，提升检索命中率。

### 5 种查询策略

| 策略 | 需要 LLM | 功能 |
|------|:--------:|------|
| `clean` | 否 | 规则去噪：全角转半角、去礼貌前缀（"请问""我想问"）、去语气词、合并空白 |
| `rewrite` | 是 | 将口语化问题改写为贴近文档书面风格的精确检索语句 |
| `decompose` | 是 | 将复杂问题拆解为 2-5 个可独立检索的子问题 |
| `expand` | 是 | 生成 3-5 个语义相近但表述不同的查询变体 |
| `hyde` | 是 | 生成 100-300 字假设文档片段，替代原始 query 做向量检索 |

### 多查询融合

`search_service.py` 新增 `multi_search()` 方法：并行执行多查询后用 **RRF（Reciprocal Rank Fusion）**融合排序，`score = Σ 1/(k + rank_i)`，以文本 MD5 去重，相同片段的分数累加。

### 核心改动

- **新建** `backend/services/query_service.py`：策略管道编排，输出 `ProcessedQuery`（含原始 query、最终查询列表、策略链和中间结果）。
- `prompt_service.py`：新增 4 种查询优化专用 TaskType（`QUERY_REWRITE`、`QUERY_DECOMPOSE`、`QUERY_EXPAND`、`HYDE`），每种配有标准版和小模型版两套模板。
- `generation_service.py`：新增 `generate_raw()` 通用 LLM 调用接口，不构建 RAG 上下文，供 QueryService 调用。
- `main.py`：`/qa` 和 `/search` 接入 QueryService；新增 `GET /query-strategies` 端点返回策略元数据。
- `schemas.py`：`QARequest` / `SearchRequest` 新增 `query_strategies`、`rewrite_model_provider`、`rewrite_model_name` 三个可选字段（默认空，零影响旧调用）。
- 前端 `QA.jsx`：新增可折叠的"查询优化"面板，策略列表动态从接口获取，勾选需要 LLM 的策略时自动展开 provider 和模型选择器。
- 前端 `Search.jsx`：检索控件区新增简洁版查询优化面板。

---

## 五、检索后后处理优化

向量检索完成后、生成模型读取上下文前，对候选片段进行可配置的后处理链路。

### 后处理策略

| 策略 | 功能 |
|------|------|
| `deduplicate` | 去除重复或高度重叠的候选片段 |
| `rerank` | 重排序，支持本地 CrossEncoder、Ollama LLM 重排，失败时降级到词面相关性评分 |
| `compress` | 压缩长片段：抽取式（快速、可控）和 LLM 压缩（语义密度高的文本），含多层校验和失败回退保留原文 |
| `diversify` | 相关性与来源多样性折中的 MMR 筛选 |
| `context_pack` | 按字符数 + token 预算打包，超出预算按句子边界截断，确保上下文窗口可控 |

### 设计特点

- **保守可靠**：策略失败时降级或保留原结果，不中断请求。LLM 压缩默认不删除无关片段，除非显式允许。
- **候选扩展**：启用重排或多样性筛选时，先召回比最终返回数更多的候选，再后处理过滤，让召回阶段和精选阶段各司其职。
- **Trace 可观察**：每次请求返回 `postprocess_trace`，记录每步输入数量、输出数量、耗时和关键配置，方便定位质量问题。

### 前端体验

- Search / QA / Generation 三个页面统一支持后处理配置，提供三种预设：质量优先、速度优先、本地离线。
- 可手动调整候选召回数、重排方式、压缩方式、LLM provider 和 token 预算等参数。
- Generation 页面支持对已保存的搜索结果重新执行后处理。
- 新增 `PostprocessTrace.jsx` 组件，展示每步处理统计（包括 LLM 压缩的成功 / 跳过 / 拒绝 / 失败数）。

### 核心改动

- **新建** `backend/services/post_retrieval_service.py`：后处理链路核心实现。
- **新建** `backend/scripts/evaluate_postprocess.py`：对比不同后处理配置效果的评估脚本。
- `search_service.py`：接入后处理管道，返回值增加 `postprocess_trace`。
- `main.py`：`/search`、`/qa`、`/generate` 均接入后处理参数。
- `schemas.py`：新增后处理相关请求和响应字段。

---

## 六、技术修复

### bge-small-zh-v1.5 模型路径修复

- 删除空目录 `backend/models/bge-small-zh-v1.5/`。
- `chunking_service.py`：将硬编码路径改为调用 `get_huggingface_model_path()` 自动解析 HuggingFace Hub 缓存，本地无缓存时回退远程下载。

### HuggingFace 模型自动加载修复

- `/qa` 端点原硬编码 `load_model=False`，HuggingFace provider 首次调用报 `'str' object has no attribute 'invoke'`。
- `generation_service.py`：改为 `if load_model or not isinstance(self.model, ChatHuggingFace)` 自动判断是否需要加载。
- `main.py`：`/qa` 端点改为 `load_model=(data.provider == "huggingface")`。

### ChromaDB SQLite 兼容性迁移

- ChromaDB 从 0.5.3 升级到 1.5.9 后，`seq_id` 列类型从 BLOB 变为 INTEGER，报 `mismatched types` 错误。
- 手动 SQLite 迁移：备份数据库 → ALTER TABLE 添加新 INTEGER 列 → BLOB 按大端序解码转 INTEGER → 删除旧列 → 重命名 → 恢复全部 15 个 collection。

### Python 3.14 懒加载兼容修复

- `llama-index`、`unstructured`、`pymilvus` 在 Python 3.14 下无预编译 wheel，模块级导入导致启动崩溃或阻塞。
- `chunking_service.py`、`loading_service.py`、`vector_store_service.py`、`search_service.py` 均将顶层导入改为方法内懒加载，服务器启动路径（`/qa`、`/search`）不再触发这些模块。
- `requirements_win.txt`：`nltk`、`pydantic` 放宽版本约束为 `>=`。

---

## 文件改动汇总

| 模块 | 新建 | 主要修改 |
|------|------|----------|
| `backend/services/prompt_service.py` | ✅ | — |
| `backend/services/query_service.py` | ✅ | — |
| `backend/services/post_retrieval_service.py` | ✅ | — |
| `backend/scripts/evaluate_postprocess.py` | ✅ | — |
| `backend/services/generation_service.py` | — | 任务路由、generate_raw、HuggingFace 修复 |
| `backend/services/loading_service.py` | — | 多格式 load_document、懒加载修复 |
| `backend/services/chunking_service.py` | — | 父子分块、结构化分块、懒加载修复 |
| `backend/services/search_service.py` | — | Small-to-Big、multi_search、RRF、后处理接入 |
| `backend/services/vector_store_service.py` | — | 元数据扩展、懒加载修复 |
| `backend/services/parsing_service.py` | — | 通用 parse_document |
| `backend/main.py` | — | 多个端点接入新能力，新增 5 个端点 |
| `backend/schemas.py` | — | 多处新增字段 |
| `frontend/src/pages/QA.jsx` | — | 任务类型、查询优化、后处理配置 |
| `frontend/src/pages/Search.jsx` | — | 查询优化、后处理配置 |
| `frontend/src/pages/Generation.jsx` | — | 任务类型、后处理配置 |
| `frontend/src/pages/LoadFile.jsx` | — | 多格式支持 |
| `frontend/src/pages/ParseFile.jsx` | — | 多格式支持 |
| `frontend/src/pages/ImportCourse.jsx` | — | 多格式支持 |
| `frontend/src/components/PostprocessTrace.jsx` | ✅ | — |
| `requirements_win.txt` | — | 新增多格式依赖，放宽版本约束 |
