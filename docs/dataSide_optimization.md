# 检索前数据侧优化

在现有的基础分块体系上，引入“父子分块”和“Markdown 结构感知分块”两种高级策略，以解决传统 RAG 系统中“切块太大搜不准，切块太小答不全”的痛点，并充分利用文档的原本结构。所有新策略默认可选，完全向后兼容现有的基础分块逻辑。

---

## 一、架构概览

```
上传文档 (PDF / JSON)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│   backend/services/chunking_service.py   核心改造    │
│  新增 parent_child / markdown_structure 两种切分策略  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│   backend/services/vector_store_service.py  扩充    │
│  元数据增强(持久化 parent_id、层级等结构化信息)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  backend/services/search_service.py     核心改造     │
│  命中子块后，自动拦截并替换为父块大上下文，且自带去重    │
└─────────────────────────────────────────────────────┘
    │
    ├── ✏️ backend/schemas.py                ← 数据模型    │
    └── ✏️ backend/main.py                   ← 接口改造    │
```

### 向后兼容设计

- 默认 `chunking_option` 依然支持原有的 `by_pages`, `fixed_size`, `by_paragraphs` 等。
- 只有明确传入 `parent_child` 或 `markdown_structure` 时，才会触发生效。
- 没有 `parent_id` 元数据的旧文档，检索时平滑回退到原始搜索逻辑，零负面影响。

---

## 二、修改 `backend/services/chunking_service.py`

在切块核心服务中新增两种高级分块分支，并在类中补充相应的具体实现方法。

### 2 种高级分块策略

| 策略 | 对应参数 (`method`) | 核心组件 | 功能原理 |
|------|--------------------|:--------:|------|
| **父子分块** | `parent_child` | `RecursiveCharacterTextSplitter` | **小精度检索+大上下文生成**：大段落（父块）切分为短句（子块）。小块保留文本内容，同时在 metadata 中挂载父块内容与唯一 `parent_id`。 |
| **结构化分块** | `markdown_structure` | `MarkdownHeaderTextSplitter` | **充分利用文档的结构化**：按 `H1, H2, H3` 切分，在生成的 chunk 开头拼接 `【文档结构：第一章 > 第三节】` 的前缀，并保留层级树。 |

### 新增核心方法
- `_parent_child_chunks(text, parent_size, child_size)`: 先按大 size 切分生成 UUID 作为 `parent_id`，再遍历按小 size 切分子块。
- `_markdown_structure_chunks(text)`: 利用 LangChain 提供的 Markdown 拆分器提取 `metadata.items()` 拼装树形结构。

---

## 三、修改 `backend/services/search_service.py`

重写 `search()` 方法中对向量数据库召回结果（`results`）的处理循环，实现 **Small-to-Big** 动态替换。

### 检索拦截与扩展逻辑
- **拦截检查**：命中 chunk 后，检查 metadata 是否存在 `parent_id` 和 `parent_content`。
- **动态替换**：如果存在，说明命中的是“子块”（短句），系统直接提取 `parent_content`（大段落全文），替换掉原本要返回的短句子。
- **自动去重**：维护一个 `seen_parent_ids = set()`。如果多个相关性高的短句同属于一个“大段落”，大段落只会被送入大模型一次，有效节省 Token 并防止重复干扰。

---

## 四、修改 `backend/services/vector_store_service.py`

增强写入 Chroma 时的 `metadata` 组装逻辑。

- 在 `_index_to_chroma()` 和 `add_to_named_collection()` 遍历 `embeddings_data["embeddings"]` 时：
  - 提取并写入新增字段：`parent_id`
  - 提取并写入新增字段：`parent_content`
  - 提取并写入新增字段：`heading_hierarchy`
- 确保 Chroma 向量库完整留存这些附加信息以支撑后续的检索替换。

---

## 五、修改 `backend/schemas.py`

扩充相关的 Pydantic 数据模型，以支持新字段的流通：

```python
class ChunkRequest(BaseModel):
    # chunking_option 枚举值新增 parent_child | markdown_structure
    ...
    # 新增父子分块专用控制参数
    parent_chunk_size: int = Field(1000, description="父块的字符数")
    child_chunk_size: int = Field(200, description="子块的字符数")

class ChunkMetadata(BaseModel):
    # 基础字段保持不变...
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    heading_hierarchy: Optional[str] = None

class SearchResultMeta(BaseModel):
    # 基础字段保持不变...
    parent_id: Optional[str] = None
    heading_hierarchy: Optional[str] = None
```

---

## 六、修改 `backend/main.py`

暴露前端控制参数，使现有路由端点支持高级分块能力：

- **`/import-to-course` 端点**：
  - 表单参数 `Form()` 新增 `parent_chunk_size: int = Form(1000)` 和 `child_chunk_size: int = Form(200)`。
  - 在调用 `chunking_service.chunk_text()` 时将这两个参数向下传递。
- **`/chunk` 端点**：
  - 接收 JSON Body 中的新参数并传递至切块服务，支持在“七步流程”界面下单独测试父子分块效果。

---

## 七、文件改动清单

| # | 文件 | 改动类型 |
|---|------|---------|
| 1 | `backend/services/chunking_service.py` | ✏️ 核心切块：新增父子分块与 Markdown 分块算法 |
| 2 | `backend/services/search_service.py` | ✏️ 检索增强：拦截子块并实现动态父块替换与去重 |
| 3 | `backend/services/vector_store_service.py` | ✏️ 向量入库：扩展 Chroma 的元数据持久化支持 |
| 4 | `backend/schemas.py` | ✏️ 数据模型：新增父块大小、子块大小及关联 ID 等字段 |
| 5 | `backend/main.py` | ✏️ 接口对接：更新导入端点以接受新的分块配置参数 |