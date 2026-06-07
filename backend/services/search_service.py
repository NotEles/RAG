from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG
import os
import json
import asyncio
import re
import chromadb

chromadb_path = "./03-vector-store/chromadb"

logger = logging.getLogger(__name__)


class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """

    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置Milvus连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)
        self.client=chromadb.PersistentClient(chromadb_path)

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表

        Returns:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """
        return [
            #     {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"}
            {"id": VectorDBProvider.CHROMA.value, "name": "chroma"}
        ]

    def list_collections(self, provider: str = VectorDBProvider.CHROMA.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合

        Args:
            provider (str): 向量数据库提供商，默认为Milvus

        Returns:
            List[Dict[str, Any]]: 集合信息列表，包含id、名称和实体数量

        Raises:
            Exception: 连接或查询集合时发生错误
        """
        try:
            # client = MilvusClient(
            #     uri="http://localhost:19530",
            #     token="root:Milvus",
            #     db_name=self.milvus_uri
            # )
            logger.info(f"into list collection")

            collections = []
            collection_names = self.client.list_collections()
            print(collection_names)

            for sample in collection_names:
                name=sample.name
                try:
                    #collection = self.client.get_or_create_collection(name)
                    collection = self.client.get_or_create_collection(name)
                    collections.append({
                        "id": name,
                        "name": name,
                        "count": 1      #collection.num_entities
                    })
                except Exception as e:
                    logger.error(f"Error getting info for collection {name}: {str(e)}")

            return collections

        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
        # finally:
        #     connections.disconnect("default")

    def save_search_results(
        self,
        query: str,
        collection_id: str,
        results: List[Dict[str, Any]],
        postprocess_trace: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        保存搜索结果到JSON文件

        Args:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表

        Returns:
            str: 保存文件的路径

        Raises:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)

            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }
            if postprocess_trace is not None:
                search_data["postprocess_trace"] = postprocess_trace

            logger.info(f"Saving search results to: {filepath}")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise

    async def search(self,
                     query: str,
                     collection_id: str,
                     top_k: int = 3,
                     threshold: float = 0.7,
                     word_count_threshold: int = 20,
                     save_results: bool = False) -> Dict[str, Any]:
        """
        执行向量搜索

        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量，默认为3
            threshold (float): 相似度阈值，低于此值的结果将被过滤，默认为0.7
            word_count_threshold (int): 文本字数阈值，低于此值的结果将被过滤，默认为20
            save_results (bool): 是否保存搜索结果，默认为False

        Returns:
            Dict[str, Any]: 包含搜索结果的字典，如果保存结果则包含保存路径

        Raises:
            Exception: 搜索过程中发生错误
        """
        try:
            # 添加参数日志
            logger.info(f"Search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results} (type: {type(save_results)})")

            logger.info(
                f"Starting search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}")

            # 连接到 Chroma
            # 获取collection
            logger.info(f"Loading collection: {collection_id}")

            collection = self.client.get_collection(collection_id)
            num_entities = collection.count()
            logger.info(f"Collection info - Entities: {num_entities}")

            if num_entities == 0:
                raise ValueError(f"Collection '{collection_id}' is empty, please re-index the document.")

            # 取一条记录的 metadata 来确定嵌入模型，避免触发 ChromaDB 默认模型下载
            sample_data = collection.get(limit=1)
            sample_meta = sample_data['metadatas'][0] if sample_data['metadatas'] else {}

            logger.info(f"query: {query}")

            # 使用collection中存储的配置创建查询向量
            logger.info("Creating query embedding")
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=sample_meta.get('embedding_provider'),
                model=sample_meta.get('embedding_model')
            )
            logger.info(f"Query embedding created with dimension: {len(query_embedding)}")

            actual_k = min(top_k, num_entities)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
            )

            logger.info(f"Sample query results: {results.get('documents')[0][0]}")

            # 处理结果
            processed_results = []
            results_count=len(results['ids'][0])
            logger.info(f"Raw search results count: {results_count}")

            # === 新增：用于记录已经扩展过的 parent_id，防止给 LLM 喂入重复的父块 ===
            seen_parent_ids = set()

            for hit in range(results_count):
                hit_score=1-results['distances'][0][hit]
                logger.info(f"Processing hit - Score: {hit_score}")
                
                if hit_score >= threshold:
                    meta = results['metadatas'][0][hit]
                    original_text = results.get('documents')[0][hit]
                    stored_word_count = int(meta.get('word_count') or 0)
                    
                    # === 核心逻辑：如果是父子分块，实施 "Small-to-Big" 扩展 ===
                    parent_id = meta.get('parent_id')
                    parent_content = meta.get('parent_content')
                    
                    if parent_id and parent_content:
                        if parent_id in seen_parent_ids:
                            logger.info(f"Skipping duplicate parent context for parent_id: {parent_id}")
                            continue  # 同一个大段落已经被命中过，去重跳过！
                            
                        seen_parent_ids.add(parent_id)
                        # 用大段落(父块) 替换 检索到的小短句(子块)
                        display_text = f"【来源扩展段落】\n{parent_content}"
                    else:
                        display_text = original_text

                    effective_word_count = max(stored_word_count, self._count_words(display_text or ""))
                    if effective_word_count < word_count_threshold:
                        logger.info(
                            "Skipping hit below word_count_threshold: %s < %s",
                            effective_word_count,
                            word_count_threshold,
                        )
                        continue
                    # ===========================================================
                    processed_results.append({
                        "text": display_text,
                        "score": float(hit_score),
                        "metadata": {
                            "source": results['metadatas'][0][hit].get('document_name'),
                            "page": results['metadatas'][0][hit].get('page_number'),
                            "chunk": results.get('ids')[0][hit],
                            "total_chunks": results['metadatas'][0][hit].get('total_chunks'),
                            "page_range": results['metadatas'][0][hit].get('page_range'),
                            "word_count": effective_word_count,
                            "embedding_provider": results['metadatas'][0][hit].get('embedding_provider'),
                            "embedding_model": results['metadatas'][0][hit].get('embedding_model'),
                            "embedding_timestamp": results['metadatas'][0][hit].get('embedding_timestamp'),
                            "parent_id": parent_id,
                            "heading_hierarchy": results['metadatas'][0][hit].get('heading_hierarchy'),
                            "retrieval_score": float(hit_score),
                            "original_score": float(hit_score),
                            "final_score": float(hit_score),
                            "candidate_rank": hit + 1,
                            "expanded_from_child": bool(parent_id and parent_content),
                        }
                    })



            # 连接到 Milvus
            #logger.info(f"Connecting to Milvus at {self.milvus_uri}")
            #connections.connect(
            #    alias="default",
            #    uri=self.milvus_uri
            #)



            # 获取collection
            # logger.info(f"Loading collection: {collection_id}")
            #collection = Collection(collection_id)
            #collection.load()

            # 记录collection的基本信息
            # logger.info(f"Collection info - Entities: {collection.num_entities}")

            # 执行搜索
            # logger.info("Querying sample entity")
            # sample_entity = collection.query(
            #    expr="id >= 0",
            #    output_fields=["embedding_provider", "embedding_model"],
            #    limit=1
            # )

            #
            # if not sample_entity:
            #     logger.error(f"Collection {collection_id} is empty")
            #     raise ValueError(f"Collection {collection_id} is empty")
            #
            # logger.info(f"Sample entity configuration: {sample_entity[0]}")
            #
            # # 使用collection中存储的配置创建查询向量
            # logger.info("Creating query embedding")
            # query_embedding = self.embedding_service.create_single_embedding(
            #     query,
            #     provider=sample_entity[0]["embedding_provider"],
            #     model=sample_entity[0]["embedding_model"]
            # )
            # logger.info(f"Query embedding created with dimension: {len(query_embedding)}")
            #
            # # 执行搜索
            # search_params = {
            #     "metric_type": "COSINE",
            #     "params": {"nprobe": 10}
            # }
            # logger.info(f"Executing search with params: {search_params}")
            # logger.info(f"Word count threshold filter: word_count >= {word_count_threshold}")
            #
            # results = collection.search(
            #     data=[query_embedding],
            #     anns_field="vector",
            #     param=search_params,
            #     limit=top_k,
            #     expr=f"word_count >= {word_count_threshold}",
            #     output_fields=[
            #         "content",
            #         "document_name",
            #         "chunk_id",
            #         "total_chunks",
            #         "word_count",
            #         "page_number",
            #         "page_range",
            #         "embedding_provider",
            #         "embedding_model",
            #         "embedding_timestamp"
            #     ]
            # )

            # 处理结果
            # processed_results = []
            # logger.info(f"Raw search results count: {len(results[0])}")
            #
            # for hits in results:
            #     for hit in hits:
            #         logger.info(f"Processing hit - Score: {hit.score}, Word Count: {hit.entity.get('word_count')}")
            #         if hit.score >= threshold:
            #             processed_results.append({
            #                 "text": hit.entity.content,
            #                 "score": float(hit.score),
            #                 "metadata": {
            #                     "source": hit.entity.document_name,
            #                     "page": hit.entity.page_number,
            #                     "chunk": hit.entity.chunk_id,
            #                     "total_chunks": hit.entity.total_chunks,
            #                     "page_range": hit.entity.page_range,
            #                     "embedding_provider": hit.entity.embedding_provider,
            #                     "embedding_model": hit.entity.embedding_model,
            #                     "embedding_timestamp": hit.entity.embedding_timestamp
            #                 }
            #             })

            response_data = {"results": processed_results}

            # 添加详细的保存逻辑日志
            logger.info(f"Preparing to handle save_results (flag: {save_results})")
            if save_results:
                logger.info("Save results is True, attempting to save...")
                if processed_results:
                    try:
                        filepath = self.save_search_results(query, collection_id, processed_results)
                        logger.info(f"Successfully saved results to: {filepath}")
                        response_data["saved_filepath"] = filepath
                    except Exception as e:
                        logger.error(f"Error saving results: {str(e)}")
                        response_data["save_error"] = str(e)
                        raise  # 添加这行来查看完整的错误堆栈
                else:
                    logger.info("No results to save")
            else:
                logger.info("Save results is False, skipping save")

            return response_data

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
        finally:
            try:
                from pymilvus import connections; connections.disconnect("default")
            except Exception:
                pass

    async def multi_search(
        self,
        queries: List[str],
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        多查询检索 + 结果融合。

        对每个查询独立执行向量搜索，然后使用 RRF (Reciprocal Rank Fusion)
        或其他融合策略合并去重结果。

        Args:
            queries: 查询文本列表（如经过改写/分解/扩展后的多个查询）
            collection_id: 要搜索的集合 ID
            top_k: 每个查询返回的最大结果数
            threshold: 相似度阈值
            word_count_threshold: 最少字数过滤
            fusion: 融合策略，目前支持 "rrf"
            rrf_k: RRF 算法的 k 常数（默认 60）

        Returns:
            {"results": [...]}，格式与 search() 一致
        """
        if not queries:
            return {"results": []}

        if len(queries) == 1:
            # 单查询直接走原方法，避免无谓开销
            return await self.search(
                query=queries[0],
                collection_id=collection_id,
                top_k=top_k,
                threshold=threshold,
                word_count_threshold=word_count_threshold,
            )

        logger.info(f"[multi_search] Processing {len(queries)} queries with fusion={fusion}")

        # 并行执行所有查询
        tasks = [
            self.search(
                query=q,
                collection_id=collection_id,
                top_k=top_k,
                threshold=threshold,
                word_count_threshold=word_count_threshold,
            )
            for q in queries
        ]
        all_responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集各查询的结果，附带 rank 信息
        # query_results: List of (rank, result_item)
        all_items: List[tuple[int, dict]] = []  # (rank, item)

        for resp in all_responses:
            if isinstance(resp, Exception):
                logger.warning(f"[multi_search] Query failed: {resp}")
                continue
            results = resp.get("results", []) if isinstance(resp, dict) else []
            for rank, item in enumerate(results):
                all_items.append((rank, item))

        if not all_items:
            return {"results": []}

        # RRF 融合
        if fusion == "rrf":
            merged = self._rrf_fusion(all_items, rrf_k=rrf_k)
        else:
            logger.warning(f"[multi_search] Unknown fusion='{fusion}', falling back to RRF")
            merged = self._rrf_fusion(all_items, rrf_k=rrf_k)

        # 按融合分数降序，取 top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        final_results = merged[:top_k]

        logger.info(f"[multi_search] {len(queries)} queries → {len(all_items)} raw items → {len(final_results)} fused results")
        return {"results": final_results}

    @staticmethod
    def _rrf_fusion(
        items: List[tuple[int, dict]],
        rrf_k: int = 60,
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion。

        对每个唯一片段累加其在不同查询中的 RRF 分数。唯一性优先使用
        文档身份 + parent/chunk/page，缺少结构信息时回退到文本指纹：
            score = sum(1 / (k + rank_i))

        Args:
            items: (rank, result_item) 元组列表
            rrf_k: RRF 常数 k

        Returns:
            融合后的结果列表，score 为 RRF 分数
        """
        doc_map: Dict[str, dict] = {}

        for rank, item in items:
            text = item.get("text", "")
            metadata = item.get("metadata", {}) or {}
            key = SearchService._result_identity_key(item)

            if key not in doc_map:
                doc_map[key] = {
                    "text": text,
                    "score": 0.0,
                    "metadata": {
                        **metadata,
                        "retrieval_score": float(metadata.get("retrieval_score", item.get("score", 0.0))),
                        "original_score": float(metadata.get("original_score", item.get("score", 0.0))),
                    },
                }

            rrf_score = 1.0 / (rrf_k + rank + 1)  # rank 从 0 开始，+1 确保分母 ≥ 1
            doc_map[key]["score"] += rrf_score
            doc_map[key]["metadata"]["fusion_score"] = doc_map[key]["score"]
            doc_map[key]["metadata"]["final_score"] = doc_map[key]["score"]
            doc_map[key]["metadata"]["postprocess_reason"] = doc_map[key]["metadata"].get("postprocess_reason")

        merged = list(doc_map.values())
        max_fusion = max((float(item.get("score", 0.0)) for item in merged), default=0.0)
        for item in merged:
            raw_fusion = float(item.get("score", 0.0))
            normalized = float(item.get("score", 0.0)) / max_fusion if max_fusion > 0 else 0.0
            item["metadata"]["fusion_score"] = raw_fusion
            item["metadata"]["fusion_score_normalized"] = normalized
            # Use normalized score for the public score so UI and thresholding share the same scale.
            item["score"] = normalized
            item["metadata"]["final_score"] = normalized
        return merged

    @staticmethod
    def _result_identity_key(item: Dict[str, Any]) -> str:
        import hashlib

        metadata = item.get("metadata", {}) or {}
        text = item.get("text", "")
        normalized_text = re.sub(r"\s+", "", text).lower()
        text_fingerprint = hashlib.md5(normalized_text.encode("utf-8", errors="replace")).hexdigest()
        doc_key = SearchService._metadata_doc_key(metadata)
        parent_id = SearchService._metadata_value(metadata, "parent_id")
        chunk_key = SearchService._metadata_locator_key(metadata)

        if doc_key and parent_id:
            return f"doc:{doc_key}|parent:{parent_id}"
        if doc_key and chunk_key:
            return f"doc:{doc_key}|loc:{chunk_key}"
        if doc_key:
            return f"doc:{doc_key}|text:{text_fingerprint}"
        if parent_id:
            return f"parent:{parent_id}|text:{text_fingerprint}"
        if chunk_key:
            return f"loc:{chunk_key}|text:{text_fingerprint}"
        return f"text:{text_fingerprint}"

    @staticmethod
    def _metadata_doc_key(metadata: Dict[str, Any]) -> str:
        parts = [
            SearchService._metadata_value(metadata, "source"),
            SearchService._metadata_value(metadata, "document_name"),
            SearchService._metadata_value(metadata, "file_name"),
            SearchService._metadata_value(metadata, "filename"),
        ]
        return "|".join(part for part in parts if part)

    @staticmethod
    def _metadata_locator_key(metadata: Dict[str, Any]) -> str:
        parts = [
            SearchService._metadata_value(metadata, "page"),
            SearchService._metadata_value(metadata, "page_range"),
            SearchService._metadata_value(metadata, "chunk"),
            SearchService._metadata_value(metadata, "chunk_id"),
        ]
        return "|".join(part for part in parts if part)

    @staticmethod
    def _metadata_value(metadata: Dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _count_words(text: str) -> int:
        if not text:
            return 0
        ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return len(ascii_words) + len(chinese_chars)
