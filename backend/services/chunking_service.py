from datetime import datetime
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.model_utils import get_huggingface_model_path

logger = logging.getLogger(__name__)

class ChunkingService:
    """
    文本分块服务，提供多种文本分块策略

    该服务支持以下分块方法：
    - by_pages: 按页面分块，每页作为一个块
    - fixed_size: 按固定大小分块
    - by_paragraphs: 按段落分块
    - by_sentences: 按句子分块
    - semantic: 语义分块，基于嵌入相似度自动识别语义边界
    """

    _embed_model = None  # 类级别懒加载，所有实例共享

    @classmethod
    def _get_embed_model(cls):
        if cls._embed_model is None:
            model_path = get_huggingface_model_path("BAAI/bge-small-zh-v1.5")
            cls._embed_model = HuggingFaceEmbedding(model_name=model_path)
        return cls._embed_model
    
    def chunk_text(self, text: str, method: str, metadata: dict, page_map: list = None, chunk_size: int = 1000) -> dict:
        """
        将文本按指定方法分块
        
        Args:
            text: 原始文本内容
            method: 分块方法，支持 'by_pages', 'fixed_size', 'by_paragraphs', 'by_sentences'
            metadata: 文档元数据
            page_map: 页面映射列表，每个元素包含页码和页面文本
            chunk_size: 固定大小分块时的块大小
            
        Returns:
            包含分块结果的文档数据结构
        
        Raises:
            ValueError: 当分块方法不支持或页面映射为空时
        """
        try:
            if not page_map:
                raise ValueError("Page map is required for chunking.")
            
            chunks = []
            total_pages = len(page_map)
            
            if method == "by_pages":
                # 直接使用 page_map 中的每页作为一个 chunk
                for page_data in page_map:
                    chunk_metadata = {
                        "chunk_id": len(chunks) + 1,
                        "page_number": page_data['page'],
                        "page_range": str(page_data['page']),
                        "word_count": len(page_data['text'].split())
                    }
                    chunks.append({
                        "content": page_data['text'],
                        "metadata": chunk_metadata
                    })
            
            elif method == "fixed_size":
                # 对每页内容进行固定大小分块
                for page_data in page_map:
                    page_chunks = self._fixed_size_chunks(page_data['text'], chunk_size)
                    for idx, chunk in enumerate(page_chunks, 1):
                        chunk_metadata = {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split())
                        }
                        chunks.append({
                            "content": chunk["text"],
                            "metadata": chunk_metadata
                        })
            
            elif method == "semantic":
                for page_data in page_map:
                    page_chunks = self._semantic_chunks(page_data['text'])
                    for chunk in page_chunks:
                        chunk_metadata = {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split())
                        }
                        chunks.append({
                            "content": chunk["text"],
                            "metadata": chunk_metadata
                        })

            elif method in ["by_paragraphs", "by_sentences"]:
                # 对每页内容进行段落或句子分块
                splitter_method = self._paragraph_chunks if method == "by_paragraphs" else self._sentence_chunks
                for page_data in page_map:
                    page_chunks = splitter_method(page_data['text'])
                    for chunk in page_chunks:
                        chunk_metadata = {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data['page'],
                            "page_range": str(page_data['page']),
                            "word_count": len(chunk["text"].split())
                        }
                        chunks.append({
                            "content": chunk["text"],
                            "metadata": chunk_metadata
                        })
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

            # 创建标准化的文档数据结构
            document_data = {
                "filename": metadata.get("filename", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", ""),
                "chunking_method": method,
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks
            }
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def _fixed_size_chunks(self, text: str, chunk_size: int) -> list[dict]:
        """
        将文本按固定大小分块
        
        Args:
            text: 要分块的文本
            chunk_size: 每块的最大字符数
            
        Returns:
            分块后的文本列表
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_length > 0 else 0)
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append({"text": " ".join(current_chunk)})
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += word_length
            
        if current_chunk:
            chunks.append({"text": " ".join(current_chunk)})
            
        return chunks

    def _paragraph_chunks(self, text: str) -> list[dict]:
        """
        将文本按段落分块
        
        Args:
            text: 要分块的文本
            
        Returns:
            分块后的段落列表
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [{"text": para} for para in paragraphs]

    def _semantic_chunks(
        self,
        text: str,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 85
    ) -> list[dict]:
        """
        使用语义相似度对文本进行分块

        Args:
            text: 要分块的文本
            buffer_size: 计算相似度时合并的相邻句子数
            breakpoint_percentile_threshold: 判定语义断点的百分位阈值（越高块越少）

        Returns:
            分块后的文本列表
        """
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self._get_embed_model()
        )
        doc = LlamaDocument(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        return [{"text": node.text} for node in nodes if node.text.strip()]

    def _sentence_chunks(self, text: str) -> list[dict]:
        """
        将文本按句子分块

        Args:
            text: 要分块的文本

        Returns:
            分块后的句子列表
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[".", "!", "?", "\n", " "]
        )
        texts = splitter.split_text(text)
        return [{"text": t} for t in texts]

    def chunk_from_json_qa(
        self,
        data: dict,
        source_filename: str = "course_qa.json",
        min_quality: int = 0,
    ) -> dict:
        """
        将课程问答 JSON 转换为标准 chunk 格式。
        每个 (问题, 答案) 对生成一个独立 chunk，保留所有质量 >= min_quality 的答案。

        期望的 JSON 结构：
        {
            "课程主题名": [
                {
                    "id": 1,
                    "question": "问题文本",
                    "answers": [
                        {"answer_quality": 0, "answer": "..."},
                        {"answer_quality": 9, "answer": "..."},
                        ...
                    ]
                },
                ...
            ]
        }

        Args:
            data: 已解析的 JSON 数据字典
            source_filename: 源文件名
            min_quality: 过滤低质量答案的最低分（0 = 保留全部）

        Returns:
            与 chunk_text 格式一致的标准文档数据结构
        """
        try:
            chunks = []

            for topic, qa_list in data.items():
                if not isinstance(qa_list, list):
                    logger.warning(f"Skipping non-list value for topic: {topic}")
                    continue

                for qa_item in qa_list:
                    question = str(qa_item.get("question", "")).strip()
                    question_id = qa_item.get("id", 0)
                    answers = qa_item.get("answers", [])

                    if not question or not answers:
                        continue

                    for ans in answers:
                        quality = ans.get("answer_quality", 0) if isinstance(ans, dict) else 0
                        if quality < min_quality:
                            continue
                        answer_text = str(ans.get("answer", "") if isinstance(ans, dict) else ans).strip()
                        if not answer_text:
                            continue

                        content = f"问题：{question}\n回答：{answer_text}"
                        chunks.append({
                            "content": content,
                            "metadata": {
                                "chunk_id": len(chunks) + 1,
                                "page_number": str(question_id),
                                "page_range": str(question_id),
                                "word_count": len(content.split()),
                                "topic": topic,
                                "question_id": question_id,
                                "question": question,
                                "answer_quality": quality,
                            }
                        })

            return {
                "filename": source_filename,
                "total_chunks": len(chunks),
                "total_pages": len(data),
                "loading_method": "json",
                "chunking_method": "by_qa",
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks,
            }

        except Exception as e:
            logger.error(f"Error in chunk_from_json_qa: {str(e)}")
            raise
