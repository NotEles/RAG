# Works — RAG 项目修改记录

## 一、Prompt 工程与任务路由系统

### 新建 `backend/services/prompt_service.py`

Prompt 模板引擎，核心功能：

- **6 种任务类型**：qa（知识问答）、summarize（摘要）、analyze（分析）、compare（对比）、explain（解释）、creative（创意），每种任务配有独立的角色提示词模板
- **任务自动检测**：`detect_task_type(query)` 基于关键词匹配，自动从用户问题中推断任务类型
- **小模型适配**：`is_small_model(model_name)` 检测 ≤3B 参数模型，自动使用更简短清晰的 prompt（避免长模板超出上下文窗口）
- **任务专属生成参数**：每种任务类型配置独立的 `generation_params`，例如 creative 温度 0.8、summarize 温度 0.2
- **前端接口**：`get_task_types()` 返回任务列表，供前端下拉框渲染

| 任务类型  | temperature | top_p | max_tokens | 用途                 |
| --------- | ----------- | ----- | ---------- | -------------------- |
| qa        | 0.3         | 0.9   | 1024       | 基于参考资料回答问题 |
| summarize | 0.2         | 0.85  | 512        | 精炼摘要             |
| analyze   | 0.4         | 0.9   | 1024       | 深度分析             |
| compare   | 0.3         | 0.9   | 1024       | 多维度对比           |
| explain   | 0.4         | 0.9   | 1024       | 概念解释             |
| creative  | 0.8         | 0.95  | 1536       | 创意内容             |

### 修改 `backend/services/generation_service.py`

- 集成 `PromptService`，在 `generate()` 中统一调用 `get_prompt()` 获取 system_msg / user_msg / generation_params
- 4 个 provider 方法（huggingface / openai / aliyun / deepseek）均接收 system_msg、user_msg、generation_params 参数，不再自行构建 prompt
- HuggingFace provider 应用 generation_params 到 pipeline（temperature / top_p / max_new_tokens）
- OpenAI / Aliyun / DeepSeek 通过 API 参数传递 temperature / top_p / max_tokens
- 结果 JSON 保存 detected_task 和 generation_params 用于追溯
- 模型列表精简为仅 `Qwen3-0.6B`

### 修改 `backend/schemas.py`

- `GenerateRequest` 新增 `task_type` 字段
- `GenerateResponse` 新增 `detected_task` 字段
- `QARequest` 新增 `task_type` 字段，provider 描述补充 `huggingface`

### 修改 `backend/main.py`

- `POST /generate` 和 `POST /qa` 传递 `task_type` 到 generation_service
- 新增 `GET /task-types` 端点，返回任务类型列表

### 修改 `frontend/src/pages/QA.jsx`

- 侧边栏新增"任务类型"下拉选择（auto + 6 种类型，附带描述文字）
- 本地模型提供商添加 `huggingface` 选项，模型列表添加 `Qwen3-0.6B`
- 请求体传递 `task_type`

### 修改 `frontend/src/pages/Generation.jsx`

- 新增任务类型下拉选择
- 从 `/task-types` 接口动态获取任务列表
- 状态消息展示 `detected_task`

---

## 二、bge-small-zh-v1.5 模型路径修复

**问题**：`backend/models/bge-small-zh-v1.5/` 目录为空（无任何模型文件），embedding/chunking 路径尝试加载时报 `Unrecognized model` 错误。

**修复**：

- 删除空目录 `backend/models/bge-small-zh-v1.5/`
- 修改 `backend/services/chunking_service.py`：将硬编码路径 `_BGE_MODEL_PATH` 改为调用 `get_huggingface_model_path("BAAI/bge-small-zh-v1.5")`，自动解析 HuggingFace Hub 缓存路径（`models--BAAI--bge-small-zh-v1.5/snapshots/`），本地无缓存时回退到远程下载

---

## 三、HuggingFace 模型自动加载修复

**问题**：`/qa` 端点硬编码 `load_model=False`，HuggingFace provider 首次调用时 `self.model` 仍为初始空字符串 `""`，调用 `.invoke()` 报错 `'str' object has no attribute 'invoke'`。

**修复**：

- `generation_service.py`：将 `if bool(load_model) == True` 改为 `if load_model or not isinstance(self.model, ChatHuggingFace)`，模型未加载时自动加载，已加载则跳过
- `main.py`：`/qa` 端点改为 `load_model=(data.provider == "huggingface")`，HuggingFace 自动触发模型加载

---

## 四、ChromaDB SQLite 兼容性迁移

**问题**：ChromaDB 从 0.5.3 升级到 1.5.9 后，`embeddings.seq_id` 和 `max_seq_id.seq_id` 列仍为 BLOB 类型，新版 Rust 代码期望 INTEGER 类型，报错 `mismatched types; Rust type u64 (as SQL type INTEGER) is not compatible with SQL type BLOB`。

**修复**（手动 SQLite 迁移）：

1. 备份 `chroma.sqlite3` 为 `chroma.sqlite3.bak`
2. ALTER TABLE 添加新 INTEGER 列
3. 将 BLOB 数据按大端序解码转为 INTEGER（与 ChromaDB migration 00005 逻辑一致）
4. 删除旧 BLOB 列，重命名新列
5. 恢复全部 15 个 collection

---
