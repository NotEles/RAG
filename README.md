# RAG 知识库问答系统

基于 [黄佳《手工制作一个RAG框架》](https://u.geekbang.org/subject/airag/1009927) 项目改进。原项目提供了完整的 RAG 七步教学流程，本版本在此基础上增加了面向实际使用的**一键导入**与**问答**功能，并对前端交互和后端接口做了系统性改进。

---

## 快速部署（Windows）

### 1. 后端

```powershell
# 创建并激活 conda 环境
conda create -n rag python=3.11
conda activate rag

# 进入后端目录，安装依赖
cd backend
pip install -r ../requirements_win.txt

# 启动后端（端口 8001）
uvicorn main:app --host 127.0.0.1 --port 8001
```

> 向量数据库默认使用 **Chroma**（本地持久化，无需额外安装）。

### 本地模型配置（可选）

若需使用本地小模型（如 Qwen3-0.6B），需先下载模型并配置路径：

```powershell
# 设置 HuggingFace 本地缓存目录（模型会下载到此目录）
$env:HF_MODEL_PATH = "D:\Hugging-face\hub"

# 模型会在首次调用时自动下载，或手动下载：
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir D:\Hugging-face\hub\models--Qwen--Qwen3-0.6B
```

> 本地模型需要 `accelerate` 包（已在 requirements_win.txt 中）。

### 2. 前端

```powershell
# 进入前端目录，安装依赖
cd frontend
npm install

# 启动前端（端口 5174）
npm run dev
```

前端默认访问地址：`http://localhost:5174`  
后端地址配置在 `frontend/src/config/config.js`，本地开发无需修改（默认 `http://localhost:8001`）。

---

## 本次改进内容

### 1. 侧边栏重构

原有的七个步骤页面收纳进可折叠的 **📁 步骤学习** 文件夹，新增两个顶级入口：

| 页面 | 路由 | 说明 |
|------|------|------|
| 步骤学习 | `/load-file` 等 | 原七步流程，折叠为一个目录 |
| 文件导入 | `/import` | 一键导入到 `course` 知识库 |
| 问答 | `/qa` | 检索 + RAG 生成 |

---

### 2. 文件导入与解析优化（`/import`）

在原有 PDF 智能解析能力基础上，新增了多格式文件解析入口，可将 PDF、JSON、Markdown、DOCX、图片等文件直接导入到名为 `course` 的 Chroma 向量数据库，无需经历七步流程。系统会根据文件扩展名自动选择解析方式，并统一转换为 RAG 后续可用的文本块。

**支持的选项：**

- **文件类型**：PDF、JSON/JSONL、Markdown、DOCX、TXT、CSV、PNG/JPG 等图片格式（自动识别格式）
- **加载方式**（PDF）：PyMuPDF、PyPDF、PDFPlumber、Unstructured
- **分块方式**：按页、固定大小、按段落、按句子、语义分块
- **编码方式**：HuggingFace（本地）或 OpenAI（需填 API Key）

**多格式解析说明：**

| 文件格式 | 处理方式 |
|------|---------|
| PDF | 保留原有 PyMuPDF、PyPDF、PDFPlumber、Unstructured 多策略解析 |
| JSON / JSONL | 自动提取结构化文本，兼容标准 chunks 与问答数据格式 |
| Markdown / TXT | 直接读取文本内容，Markdown 会按标题结构切分 |
| DOCX | 提取段落、标题和表格文本，需要 `python-docx` |
| CSV | 按行提取字段内容并转换为文本 |
| PNG / JPG 等图片 | 通过 OCR 提取图片文字，需要 `pillow`、`pytesseract` 和系统 Tesseract OCR |

**JSON 文件支持三种格式：**

| 格式 | 处理方式 |
|------|---------|
| 已有 `chunks` 字段的标准格式 | 直接使用 |
| 问答格式（含 `question` + `answers` 字段） | 每个 (问题, 答案) 对生成独立 chunk |
| 其他任意 JSON | 自动提取数组条目，按选定分块方式处理 |

**问答 JSON 格式说明：**

```json
{
  "课程主题名": [
    {
      "id": 1,
      "question": "什么是自然语言处理？",
      "answers": [
        { "answer_quality": 0, "answer": "..." },
        { "answer_quality": 9, "answer": "..." }
      ]
    }
  ]
}
```

选择此格式时，界面会显示**答案质量过滤**滑块（0–9），设为 0 保留全部答案，设为 5 则只保留高质量答案。每条答案生成一个独立向量，提升检索覆盖率。

---

### 3. 问答（`/qa`）

从 `course`（或其他 collection）检索相关知识，结合 RAG 生成回答。

**模型支持：**

| 提供商 | 模型 |
|--------|------|
| DeepSeek | DeepSeek V3、DeepSeek R1（含推理链） |
| OpenAI | GPT-4o、GPT-4o mini、GPT-3.5 Turbo |
| 阿里云百炼 | Qwen Turbo、Qwen Plus |
| 本地模型 | Qwen3-0.6B（HuggingFace，需下载到 `D:\Hugging-face\hub`） |

**任务类型与 Prompt 工程：**

系统内置 6 种任务类型的专用 Prompt 模板，通过**任务路由**自动匹配或手动选择：

| 任务类型 | 触发关键词 | temperature | 说明 |
|----------|-----------|-------------|------|
| 知识问答 | 默认 | 0.3 | 结构化回答，引用来源编号 |
| 摘要总结 | 总结/摘要/概括 | 0.2 | 低温度忠实原文，分层输出 |
| 深度分析 | 分析/为什么/原因 | 0.4 | 思维链推理，分步输出 |
| 对比比较 | 对比/比较/区别 | 0.3 | 表格化对比异同和优劣 |
| 概念解释 | 解释/什么是/定义 | 0.4 | 由浅入深，举例说明 |
| 创意写作 | 写/创作/设计/方案 | 0.8 | 高温度鼓励发散，多角度方案 |

- **自动路由**：默认根据问题关键词自动识别任务类型，选择对应 prompt 和生成参数
- **小模型适配**：自动识别 ≤3B 参数模型，使用更简短明确的 prompt 变体
- **每任务参数独立**：不同任务的 temperature、top_p、max_tokens 自动调整，无需手动设置

**检索参数：**
- 数据库：默认 `course`，可切换到任意已有 collection
- Top K：返回最相关的 K 条文档（1–20）
- 相似度阈值：过滤低相关结果
- 最少字数：过滤内容过短的 chunk

结果分两个标签页展示：**生成回答**（Markdown 渲染）和**检索来源**（含相似度百分比）。

---

### 4. 后端接口规范化（`backend/schemas.py`）

新增 Pydantic 模型文件，为所有 POST 端点定义明确的请求/响应结构，替换原来的裸 `dict`。启动后端后访问 `http://localhost:8001/docs` 可查看完整的交互式接口文档。

**主要接口一览：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/import-to-course` | POST | 上传文件，自动处理并写入 `course` |
| `/qa` | POST | 检索 + 生成，一体化问答 |
| `/course/info` | GET | 查询 `course` collection 状态 |
| `/search` | POST | 向量相似度检索 |
| `/generate` | POST | 基于检索结果生成回答（支持 task_type 参数） |
| `/task-types` | GET | 获取可用任务类型列表 |
| `/embed` | POST | 生成文档向量嵌入 |
| `/index` | POST | 将嵌入写入向量数据库 |

---

## 技术栈

- **前端**：React 18 + Vite + Tailwind CSS
- **后端**：Python FastAPI + Pydantic
- **向量数据库**：Chroma（本地持久化）
- **嵌入模型**：HuggingFace `BAAI/bge-small-zh-v1.5`（本地）/ OpenAI API
- **生成模型**：DeepSeek / OpenAI / 阿里云百炼（API）/ Qwen3-0.6B（本地）
- **Prompt 工程**：`prompt_service.py` — 6 种任务模板 + 自动路由 + 小模型适配
