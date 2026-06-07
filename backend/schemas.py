"""
schemas.py — 后端所有端点的 Pydantic 请求/响应模型

命名约定：
  - XxxRequest  : POST 请求体
  - XxxResponse : 端点响应体
  - XxxItem     : 列表中的单条记录

FastAPI 会根据这些模型自动生成 /docs 的交互式文档。
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

PostprocessStrategy = Literal["deduplicate", "rerank", "compress", "diversify", "context_pack"]
RerankMethod = Literal["cross_encoder", "llm"]
CompressMethod = Literal["none", "extractive", "llm"]
PostprocessLlmProvider = Literal["ollama", "deepseek", "openai", "aliyun", "huggingface"]


# ─────────────────────────────────────────────
# 公共子结构
# ─────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    chunk_id: int
    page_number: Union[int, str]
    page_range: str
    word_count: int
    # --- 新增：为父子分块和结构化分块准备的字段 ---
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    heading_hierarchy: Optional[str] = None

class Chunk(BaseModel):
    content: str
    metadata: ChunkMetadata


# ─────────────────────────────────────────────
# /save  保存分块文档
# ─────────────────────────────────────────────

class SaveChunksRequest(BaseModel):
    docName: str = Field(..., description="文档名称（不含扩展名）")
    chunks: List[Chunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SaveChunksResponse(BaseModel):
    status: str
    message: str
    filepath: str


# ─────────────────────────────────────────────
# /chunk  对已加载文档进行分块
# ─────────────────────────────────────────────

class ChunkRequest(BaseModel):
    doc_id: str = Field(..., description="01-loaded-docs 中的文件名，含 .json 后缀")
    chunking_option: str = Field(
        ...,
        description="分块方式：by_pages | fixed_size | by_paragraphs | by_sentences | semantic",
    )
    chunk_size: int = Field(1000, ge=50, description="固定大小分块时每块的字符数")
    # --- 新增：父子分块专用参数 ---
    parent_chunk_size: int = Field(1000, description="父块的字符数")
    child_chunk_size: int = Field(200, description="子块的字符数")



# ─────────────────────────────────────────────
# /embed  生成向量嵌入
# ─────────────────────────────────────────────

class EmbedRequest(BaseModel):
    documentId: str = Field(..., description="01-loaded-docs 或 01-chunked-docs 中的文件名")
    provider: str = Field(..., description="嵌入提供商：openai | huggingface | bedrock")
    model: str = Field(..., description="嵌入模型名称")

class EmbedResponse(BaseModel):
    status: str
    message: str
    filepath: str


# ─────────────────────────────────────────────
# /index  将嵌入写入向量数据库
# ─────────────────────────────────────────────

class IndexRequest(BaseModel):
    fileId: str = Field(..., description="02-embedded-docs 中的文件名，含 .json 后缀")
    vectorDb: str = Field(..., description="向量数据库：chroma | milvus | faiss 等")
    indexMode: str = Field(..., description="索引模式：hnsw | standard | flat 等（随数据库而异）")

class IndexResponse(BaseModel):
    database: str
    index_mode: str
    total_vectors: int
    index_size: Any
    processing_time: float
    collection_name: str


# ─────────────────────────────────────────────
# /search  向量相似度检索
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    collection_id: str = Field(..., description="Chroma collection 名称")
    top_k: int = Field(3, ge=1, le=50, description="返回结果数量")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值，低于此值的结果被过滤")
    word_count_threshold: int = Field(100, ge=0, description="最少字数，低于此值的结果被过滤")
    save_results: bool = Field(False, description="是否保存搜索结果到文件")
    # ── 查询优化（可选，默认空 = 不启用）──
    query_strategies: List[str] = Field(
        default=[],
        description="查询优化策略列表：clean | rewrite | decompose | expand | hyde，空列表不启用"
    )
    rewrite_model_provider: str = Field(
        default="deepseek",
        description="查询优化所用的 LLM 提供商（仅当 query_strategies 非空时生效）"
    )
    rewrite_model_name: str = Field(
        default="deepseek-v3",
        description="查询优化所用的 LLM 模型名称"
    )
    # ── 检索后优化（可选）──
    postprocess_enabled: bool = Field(False, description="是否启用检索后优化")
    postprocess_strategies: List[PostprocessStrategy] = Field(
        default_factory=list,
        description="检索后优化策略：deduplicate | rerank | compress | diversify | context_pack"
    )
    postprocess_fetch_k: int = Field(20, ge=1, le=50, description="启用重排时先召回的候选数量")
    rerank_method: RerankMethod = Field("cross_encoder", description="重排方式：cross_encoder | llm")
    rerank_model: str = Field("BAAI/bge-reranker-base", description="本地 CrossEncoder 重排模型")
    rerank_top_k: int = Field(5, ge=1, le=20, description="重排后保留的结果数量")
    compress_method: CompressMethod = Field("extractive", description="压缩方式：none | extractive | llm")
    max_context_chars: int = Field(8000, ge=1000, le=50000, description="最终上下文字符预算")
    max_context_tokens: int = Field(3000, ge=256, le=32000, description="最终上下文 token 预算")
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="MMR 相关性权重，越高越重视相关性")
    postprocess_llm_provider: PostprocessLlmProvider = Field("ollama", description="LLM 重排/压缩提供商：ollama | deepseek | openai | aliyun | huggingface")
    postprocess_llm_model: str = Field("qwen2.5:3b", description="LLM 重排/压缩模型")
    postprocess_api_key: Optional[str] = Field(None, description="LLM 重排/压缩 API Key，未传则读取环境变量")

class SearchResultMeta(BaseModel):
    source: Optional[str] = None
    page: Optional[str] = None
    chunk: Optional[str] = None
    total_chunks: Optional[int] = None
    page_range: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_timestamp: Optional[str] = None
    # --- 新增 ---
    parent_id: Optional[str] = None
    heading_hierarchy: Optional[str] = None
    original_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_fallback_from: Optional[str] = None
    final_score: Optional[float] = None
    compressed: Optional[bool] = None
    original_length: Optional[int] = None
    compressed_length: Optional[int] = None
    llm_compress_skipped_length: Optional[int] = None
    llm_compress_rejected_length: Optional[int] = None
    llm_compress_rejected_reason: Optional[str] = None
    postprocess_reason: Optional[str] = None
    mmr_score: Optional[float] = None
    packed_tokens: Optional[int] = None

class SearchResultItem(BaseModel):
    text: str
    score: float = Field(..., description="余弦相似度（0~1，越高越相关）")
    metadata: SearchResultMeta

class SearchResponse(BaseModel):
    results: List[SearchResultItem]


# ─────────────────────────────────────────────
# /save-search  保存检索结果
# ─────────────────────────────────────────────

class SaveSearchRequest(BaseModel):
    query: str
    collection_id: str
    results: List[Dict[str, Any]]

class SaveSearchResponse(BaseModel):
    saved_filepath: str


# ─────────────────────────────────────────────
# /generate  RAG 生成回答
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str
    provider: str = Field(..., description="生成模型提供商：openai | deepseek | aliyun | huggingface")
    model_name: str
    search_results: List[Dict[str, Any]] = Field(..., description="来自 /search 的结果列表")
    load_model: bool = Field(False, description="是否重新加载本地 HuggingFace 模型")
    api_key: Optional[str] = None
    show_reasoning: bool = Field(False, description="DeepSeek R1 是否显示推理过程")
    task_type: Optional[str] = Field(None, description="任务类型：auto | qa | summarize | analyze | compare | explain | creative，None 则自动检测")
    # ── 检索后优化（可选，用于对传入的 search_results 再加工）──
    postprocess_enabled: bool = Field(False, description="是否启用检索后优化")
    postprocess_strategies: List[PostprocessStrategy] = Field(
        default_factory=list,
        description="检索后优化策略：deduplicate | rerank | compress | diversify | context_pack"
    )
    rerank_method: RerankMethod = Field("cross_encoder", description="重排方式：cross_encoder | llm")
    rerank_model: str = Field("BAAI/bge-reranker-base", description="本地 CrossEncoder 重排模型")
    rerank_top_k: int = Field(5, ge=1, le=20, description="重排后保留的结果数量")
    compress_method: CompressMethod = Field("extractive", description="压缩方式：none | extractive | llm")
    max_context_chars: int = Field(8000, ge=1000, le=50000, description="最终上下文字符预算")
    max_context_tokens: int = Field(3000, ge=256, le=32000, description="最终上下文 token 预算")
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="MMR 相关性权重，越高越重视相关性")
    postprocess_llm_provider: PostprocessLlmProvider = Field("ollama", description="LLM 重排/压缩提供商：ollama | deepseek | openai | aliyun | huggingface")
    postprocess_llm_model: str = Field("qwen2.5:3b", description="LLM 重排/压缩模型")
    postprocess_api_key: Optional[str] = Field(None, description="LLM 重排/压缩 API Key，未传则读取对应提供商环境变量")

class GenerateResponse(BaseModel):
    response: str
    saved_filepath: str
    detected_task: Optional[str] = None
    search_results: Optional[List[SearchResultItem]] = None


# ─────────────────────────────────────────────
# /import-to-course  一键导入到 course 知识库
# ─────────────────────────────────────────────

class ImportResponse(BaseModel):
    status: str
    message: str
    chunks_imported: int
    collection: str
    filename: str


# ─────────────────────────────────────────────
# /qa  检索 + 生成一体化问答
# ─────────────────────────────────────────────

class QARequest(BaseModel):
    query: str = Field(..., description="用户问题")
    api_key: Optional[str] = Field(None, description="LLM API Key")
    provider: str = Field(..., description="生成模型提供商：openai | deepseek | aliyun | huggingface")
    model_name: str
    collection: str = Field("course", description="要检索的 Chroma collection 名称")
    top_k: int = Field(3, ge=1, le=20)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    word_count_threshold: int = Field(20, ge=0)
    show_reasoning: bool = False
    task_type: Optional[str] = Field(None, description="任务类型：auto | qa | summarize | analyze | compare | explain | creative，None 则自动检测")
    # ── 查询优化（可选，默认空 = 不启用）──
    query_strategies: List[str] = Field(
        default=[],
        description="查询优化策略列表：clean | rewrite | decompose | expand | hyde，空列表不启用"
    )
    rewrite_model_provider: str = Field(
        default="deepseek",
        description="查询优化所用的 LLM 提供商（仅当 query_strategies 非空时生效）"
    )
    rewrite_model_name: str = Field(
        default="deepseek-v3",
        description="查询优化所用的 LLM 模型名称"
    )
    # ── 检索后优化（可选）──
    postprocess_enabled: bool = Field(False, description="是否启用检索后优化")
    postprocess_strategies: List[PostprocessStrategy] = Field(
        default_factory=list,
        description="检索后优化策略：deduplicate | rerank | compress | diversify | context_pack"
    )
    postprocess_fetch_k: int = Field(20, ge=1, le=50, description="启用重排时先召回的候选数量")
    rerank_method: RerankMethod = Field("cross_encoder", description="重排方式：cross_encoder | llm")
    rerank_model: str = Field("BAAI/bge-reranker-base", description="本地 CrossEncoder 重排模型")
    rerank_top_k: int = Field(5, ge=1, le=20, description="重排后保留的结果数量")
    compress_method: CompressMethod = Field("extractive", description="压缩方式：none | extractive | llm")
    max_context_chars: int = Field(8000, ge=1000, le=50000, description="最终上下文字符预算")
    max_context_tokens: int = Field(3000, ge=256, le=32000, description="最终上下文 token 预算")
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="MMR 相关性权重，越高越重视相关性")
    postprocess_llm_provider: PostprocessLlmProvider = Field("ollama", description="LLM 重排/压缩提供商：ollama | deepseek | openai | aliyun | huggingface")
    postprocess_llm_model: str = Field("qwen2.5:3b", description="LLM 重排/压缩模型")
    postprocess_api_key: Optional[str] = Field(None, description="LLM 重排/压缩 API Key，未传则读取对应提供商环境变量")

class QAResponse(BaseModel):
    query: str
    search_results: List[SearchResultItem]
    response: str


# ─────────────────────────────────────────────
# Collection / Document 相关
# ─────────────────────────────────────────────

class CollectionItem(BaseModel):
    id: str
    name: str
    count: Optional[int] = None

class CollectionInfoResponse(BaseModel):
    name: str
    num_entities: int
    schema: Dict[str, Any] = Field(default_factory=dict)

class DocumentItem(BaseModel):
    id: str
    name: str
    type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentsResponse(BaseModel):
    documents: List[DocumentItem]
