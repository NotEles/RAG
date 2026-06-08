"""
检索后优化服务。

提供轻量去重、本地 CrossEncoder 重排、抽取式压缩和上下文预算打包。
所有策略均为可选，默认不改变现有检索行为。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from utils.model_utils import get_huggingface_model_path

logger = logging.getLogger(__name__)


@dataclass
class PostRetrievalConfig:
    enabled: bool = False
    strategies: List[str] = field(default_factory=list)
    rerank_method: str = "cross_encoder"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5
    compress_method: str = "extractive"
    max_context_chars: int = 8000
    max_context_tokens: int = 3000
    mmr_lambda: float = 0.7
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    api_key: str | None = None
    final_top_k: int | None = None
    final_threshold: float | None = None
    allow_drop_irrelevant: bool = False
    trace_enabled: bool = True
    llm_compress_top_n: int = 2


class PostRetrievalService:
    """在向量召回之后、RAG 生成之前优化候选上下文。"""

    _rerankers: Dict[str, Any] = {}

    def process(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: PostRetrievalConfig,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not config.enabled or not results:
            return results, []

        trace: List[Dict[str, Any]] = []
        strategies = set(config.strategies or [])
        processed = [self._normalize_item(item) for item in results if item.get("text")]
        self._trace(trace, "normalize", len(results), len(processed))

        if "deduplicate" in strategies:
            before = len(processed)
            started = time.perf_counter()
            processed = self.deduplicate(processed)
            self._trace(trace, "deduplicate", before, len(processed), started)

        if "rerank" in strategies:
            before = len(processed)
            started = time.perf_counter()
            processed = self.rerank(query, processed, config)
            rerank_extra = {
                "method": config.rerank_method,
                "model": config.llm_model if config.rerank_method == "llm" else config.rerank_model,
            }
            fallback_from = self._rerank_fallback_from(processed)
            if fallback_from:
                rerank_extra.update(
                    {
                        "fallback": True,
                        "fallback_from": fallback_from,
                        "fallback_method": "lexical",
                    }
                )
            self._trace(
                trace,
                "rerank",
                before,
                len(processed),
                started,
                rerank_extra,
            )
        else:
            for item in processed:
                meta = item.setdefault("metadata", {})
                meta.setdefault("retrieval_score", float(item.get("score", 0.0)))
                meta.setdefault("original_score", float(item.get("score", 0.0)))
                meta.setdefault("final_score", float(item.get("score", 0.0)))

        if "diversify" in strategies:
            before = len(processed)
            started = time.perf_counter()
            processed = self.diversify_mmr(query, processed, config)
            self._trace(trace, "diversify", before, len(processed), started, {"mmr_lambda": config.mmr_lambda})

        if config.final_threshold is not None:
            before = len(processed)
            processed = [
                item
                for item in processed
                if self._threshold_score(item) >= config.final_threshold
            ]
            self._trace(trace, "final_threshold", before, len(processed), extra={"threshold": config.final_threshold})

        if "compress" in strategies:
            before = len(processed)
            started = time.perf_counter()
            trace_source = processed
            if config.compress_method == "llm":
                trace_source = list(processed)
                processed = self.compress_llm(query, processed, config)
            elif config.compress_method == "extractive":
                processed = self.compress_extractive(query, processed)
            self._trace(
                trace,
                "compress",
                before,
                len(processed),
                started,
                self._compression_trace_extra(trace_source, config),
            )

        if "context_pack" in strategies:
            before = len(processed)
            started = time.perf_counter()
            processed = self.pack_context(processed, config.max_context_chars, config.max_context_tokens)
            self._trace(
                trace,
                "context_pack",
                before,
                len(processed),
                started,
                {"max_context_chars": config.max_context_chars, "max_context_tokens": config.max_context_tokens},
            )

        if config.final_top_k:
            before = len(processed)
            processed = processed[: max(1, config.final_top_k)]
            self._trace(trace, "final_top_k", before, len(processed), extra={"top_k": config.final_top_k})

        return processed, trace if config.trace_enabled else []

    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in results:
            meta = item.get("metadata", {})
            key = self._result_identity_key(item)
            if key in seen:
                continue
            seen.add(key)
            meta["postprocess_reason"] = self._append_reason(meta.get("postprocess_reason"), "deduplicate")
            deduped.append(item)
        return deduped

    def diversify_mmr(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: PostRetrievalConfig,
    ) -> List[Dict[str, Any]]:
        """使用轻量 MMR 在相关性和多样性之间折中。"""
        query_terms = self._terms(query)
        candidates = list(results)
        selected: List[Dict[str, Any]] = []
        target_count = min(len(candidates), max(config.rerank_top_k, 1))

        while candidates and len(selected) < target_count:
            best_idx = 0
            best_score = -float("inf")
            for idx, item in enumerate(candidates):
                relevance = float(item.get("score", 0.0)) or self._term_overlap_score(query_terms, item.get("text", ""))
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        self._text_similarity(item.get("text", ""), chosen.get("text", ""))
                        for chosen in selected
                    )
                mmr_score = (config.mmr_lambda * relevance) - ((1 - config.mmr_lambda) * diversity_penalty)
                if mmr_score > best_score:
                    best_idx = idx
                    best_score = mmr_score

            chosen = candidates.pop(best_idx)
            meta = chosen.setdefault("metadata", {})
            meta["mmr_score"] = best_score
            meta["postprocess_reason"] = self._append_reason(meta.get("postprocess_reason"), "diversify_mmr")
            selected.append(chosen)

        return selected

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: PostRetrievalConfig,
    ) -> List[Dict[str, Any]]:
        rerank_reason = "llm_rerank" if config.rerank_method == "llm" else "rerank"
        rerank_reasons: List[str | None] = [None] * len(results)
        if config.rerank_method == "llm":
            judgments = self._llm_rerank_judgments(query, results, config)
            scores = None
            if judgments is not None:
                scores = [float(item.get("score", 0.0)) for item in judgments]
                rerank_reasons = [item.get("reason") for item in judgments]
        else:
            pairs = [[query, item.get("text", "")] for item in results]
            scores = self._cross_encoder_scores(pairs, config.rerank_model)
        if scores is None:
            scores = [self._lexical_score(query, item.get("text", "")) for item in results]
            rerank_reason = "lexical_rerank_fallback"

        reranked = []
        for idx, (item, rerank_score) in enumerate(zip(results, scores)):
            meta = item.setdefault("metadata", {})
            retrieval_score = float(meta.get("retrieval_score", item.get("score", 0.0)))
            original_score = float(meta.get("original_score", item.get("score", 0.0)))
            fusion_score = meta.get("fusion_score")
            fusion_norm = meta.get("fusion_score_normalized")
            if fusion_score is not None:
                fusion_component = float(fusion_norm if fusion_norm is not None else fusion_score)
                final_score = (0.10 * fusion_component) + (0.90 * float(rerank_score))
            else:
                final_score = (0.25 * retrieval_score) + (0.75 * float(rerank_score))
            meta["retrieval_score"] = retrieval_score
            meta["original_score"] = original_score
            meta["rerank_score"] = float(rerank_score)
            meta["final_score"] = final_score
            if rerank_reasons[idx]:
                meta["rerank_reason"] = rerank_reasons[idx]
            if rerank_reason == "lexical_rerank_fallback":
                meta["rerank_fallback_from"] = config.rerank_method
            meta["postprocess_reason"] = self._append_reason(
                meta.get("postprocess_reason"),
                rerank_reason,
            )
            item["score"] = final_score
            reranked.append(item)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked[: config.rerank_top_k]

    def compress_llm(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: PostRetrievalConfig,
    ) -> List[Dict[str, Any]]:
        for idx, item in enumerate(results):
            text = item.get("text", "")
            if not text:
                continue
            if idx >= max(0, config.llm_compress_top_n):
                meta = item.setdefault("metadata", {})
                meta["llm_compress_skipped_rank"] = idx + 1
                meta["postprocess_reason"] = self._append_reason(
                    meta.get("postprocess_reason"), "llm_compress_skipped_rank"
                )
                continue
            if len(text) < 300:
                meta = item.setdefault("metadata", {})
                meta["llm_compress_skipped_length"] = len(text)
                meta["postprocess_reason"] = self._append_reason(
                    meta.get("postprocess_reason"), "llm_compress_skipped_short"
                )
                continue
            prompt = (
                "请根据用户问题压缩检索片段，只保留能帮助回答问题的信息，输出 JSON 对象。\n"
                "要求：\n"
                "1. 不添加片段中没有的信息。\n"
                "2. 保留关键术语、数据、因果关系和限制条件。\n"
                "3. 如果片段与问题无关，relevant=false，compressed_text 置空。\n"
                "4. 如果片段相关，使用与原片段主要语言一致的语言输出约 80-180 字/词的证据摘要。\n"
                "5. 只输出 JSON，不要解释。格式：{\"relevant\":true,\"compressed_text\":\"...\",\"kept_facts\":[\"...\"]}\n\n"
                f"用户问题：{query}\n\n检索片段：\n{text}"
            )
            compressed = self._call_llm(
                system_msg="你是严谨的 RAG 上下文压缩器，只输出合法 JSON。",
                user_msg=prompt,
                config=config,
                max_tokens=512,
            )
            if not compressed:
                meta = item.setdefault("metadata", {})
                meta["postprocess_reason"] = self._append_reason(
                    meta.get("postprocess_reason"), "llm_compress_failed"
                )
                continue
            payload = self._parse_compression_payload(compressed)
            compressed_text = str(payload.get("compressed_text") or "").strip()
            is_relevant = self._parse_bool(payload.get("relevant", True), default=True)
            kept_facts = payload.get("kept_facts") if isinstance(payload.get("kept_facts"), list) else []

            if not is_relevant or compressed.strip() in {"无关", "无关。"}:
                meta = item.setdefault("metadata", {})
                if config.allow_drop_irrelevant and self._can_drop_as_irrelevant(query, text, item):
                    item["text"] = ""
                    meta["postprocess_reason"] = self._append_reason(
                        meta.get("postprocess_reason"), "llm_compress_drop"
                    )
                else:
                    meta["postprocess_reason"] = self._append_reason(
                        meta.get("postprocess_reason"), "llm_compress_rejected"
                    )
                continue
            if not compressed_text:
                compressed_text = compressed.strip()

            rejection_reason = self._llm_compression_rejection_reason(query, text, compressed_text)
            if not rejection_reason:
                rejection_reason = self._semantic_compression_rejection_reason(query, text, compressed_text, config)

            if not rejection_reason:
                meta = item.setdefault("metadata", {})
                meta["compressed"] = True
                meta["original_length"] = len(text)
                meta["compressed_length"] = len(compressed_text)
                if kept_facts:
                    meta["kept_facts"] = [str(fact)[:180] for fact in kept_facts[:8]]
                meta["postprocess_reason"] = self._append_reason(meta.get("postprocess_reason"), "llm_compress")
                item["text"] = compressed_text
            else:
                meta = item.setdefault("metadata", {})
                meta["llm_compress_rejected_length"] = len(compressed_text)
                meta["llm_compress_rejected_reason"] = rejection_reason
                meta["postprocess_reason"] = self._append_reason(
                    meta.get("postprocess_reason"), "llm_compress_rejected"
                )

        return [item for item in results if item.get("text")]

    def compress_extractive(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in results:
            text = item.get("text", "")
            sentences = self._split_sentences(text)
            if len(sentences) <= 2:
                continue

            ranked = self._rank_sentences(query, sentences)
            keep = set()
            for score, idx in ranked[: min(4, max(2, len(sentences) // 3))]:
                if score <= 0 and keep:
                    continue
                keep.update({idx - 1, idx, idx + 1})
            keep = {idx for idx in keep if 0 <= idx < len(sentences)}
            if not keep:
                continue

            compressed = "\n".join(sentences[idx] for idx in sorted(keep)).strip()
            if compressed and len(compressed) < len(text):
                meta = item.setdefault("metadata", {})
                meta["compressed"] = True
                meta["original_length"] = len(text)
                meta["compressed_length"] = len(compressed)
                meta["kept_sentence_indexes"] = sorted(keep)
                meta["postprocess_reason"] = self._append_reason(meta.get("postprocess_reason"), "compress")
                item["text"] = compressed
        return results

    def _is_safe_llm_compression(self, query: str, original: str, compressed: str) -> bool:
        return self._llm_compression_rejection_reason(query, original, compressed) is None

    def _llm_compression_rejection_reason(self, query: str, original: str, compressed: str) -> str | None:
        if not compressed or len(compressed) >= len(original):
            return "length_check"

        min_chars = min(80, max(40, int(len(original) * 0.12)))
        if len(compressed) < min_chars:
            return "too_short"

        query_terms = self._terms(query)
        compressed_terms = self._terms(compressed)
        original_terms = self._terms(original)
        # In cross-lingual retrieval the query and source may not share surface terms.
        # Only require query-term preservation when the original itself contained them.
        if query_terms and (query_terms & original_terms) and not (query_terms & compressed_terms):
            return "query_terms_missing"

        if compressed_terms and original_terms:
            overlap_ratio = len(compressed_terms & original_terms) / max(1, len(compressed_terms))
            if overlap_ratio < 0.55:
                return "source_overlap_low"

        return None

    def _semantic_compression_rejection_reason(
        self,
        query: str,
        original: str,
        compressed: str,
        config: PostRetrievalConfig,
    ) -> str | None:
        query_terms = self._terms(query)
        original_terms = self._terms(original)
        if not query_terms or (query_terms & original_terms):
            return None

        prompt = (
            "请判断压缩片段是否仍然保留了原始检索片段中能够回答用户问题的核心信息。\n"
            "只输出 yes 或 no，不要解释。\n\n"
            f"用户问题：{query}\n\n"
            f"原始检索片段：\n{original[:1800]}\n\n"
            f"压缩片段：\n{compressed[:900]}"
        )
        verdict = self._call_llm(
            system_msg="你是 RAG 压缩质量审查器，只输出 yes 或 no。",
            user_msg=prompt,
            config=config,
            max_tokens=8,
        )
        if not verdict:
            return "semantic_check_failed"

        normalized = verdict.strip().lower()
        if normalized.startswith(("yes", "y", "是", "相关", "true")):
            return None
        if normalized.startswith(("no", "n", "否", "不", "false")):
            return "semantic_check_failed"
        return "semantic_check_unparseable"

    def _can_drop_as_irrelevant(self, query: str, text: str, item: Dict[str, Any]) -> bool:
        meta = item.get("metadata", {})
        rerank_score = meta.get("rerank_score")
        if rerank_score is not None and float(rerank_score) >= 0.35:
            return False
        return self._lexical_score(query, text) < 0.08

    def pack_context(
        self,
        results: List[Dict[str, Any]],
        max_context_chars: int,
        max_context_tokens: int,
    ) -> List[Dict[str, Any]]:
        packed = []
        used_chars = 0
        used_tokens = 0
        for item in results:
            text = item.get("text", "")
            if not text:
                continue
            text_tokens = self._count_tokens(text)
            remaining_chars = max_context_chars - used_chars
            remaining_tokens = max_context_tokens - used_tokens
            if remaining_chars <= 0 or remaining_tokens <= 0:
                break

            if len(text) > remaining_chars or text_tokens > remaining_tokens:
                text = self._truncate_to_budget(text, remaining_chars, remaining_tokens)
                item = {**item, "text": text}
                text_tokens = self._count_tokens(text)
                item.setdefault("metadata", {})["context_truncated"] = True
            item.setdefault("metadata", {})["postprocess_reason"] = self._append_reason(
                item.get("metadata", {}).get("postprocess_reason"), "context_pack"
            )
            item.setdefault("metadata", {})["packed_tokens"] = text_tokens
            item.setdefault("metadata", {})["context_index"] = len(packed) + 1
            packed.append(item)
            used_chars += len(text)
            used_tokens += text_tokens
        return packed

    def _llm_rerank_judgments(
        self,
        query: str,
        results: List[Dict[str, Any]],
        config: PostRetrievalConfig,
    ) -> List[Dict[str, Any]] | None:
        snippets = []
        for idx, item in enumerate(results, start=1):
            text = item.get("text", "")
            snippets.append(f"{idx}. id={idx}\n{text[:900]}")
        user_msg = (
            "请判断每个检索片段对回答用户问题的相关性，输出 JSON 数组。\n"
            "要求：\n"
            "1. 数组长度必须等于片段数量。\n"
            "2. 每个元素包含 id、score、reason，score 是 0 到 1 的小数，1 表示高度相关。\n"
            "3. reason 用不超过 30 个汉字说明依据。\n"
            "4. 只输出 JSON 数组，不要输出解释。例如 [{\"id\":1,\"score\":0.9,\"reason\":\"直接回答问题\"}]\n\n"
            f"用户问题：{query}\n\n检索片段：\n" + "\n\n".join(snippets)
        )
        raw = self._call_llm(
            system_msg="你是检索重排器，只输出合法 JSON。",
            user_msg=user_msg,
            config=config,
            max_tokens=max(128, len(results) * 16),
        )
        if not raw:
            return None
        try:
            parsed = json.loads(self._extract_json_array(raw))
            if not isinstance(parsed, list) or len(parsed) != len(results):
                return None
            by_id: Dict[int, Dict[str, Any]] = {}
            for fallback_idx, item in enumerate(parsed, start=1):
                if isinstance(item, dict):
                    raw_id = item.get("id", fallback_idx)
                    score = item.get("score", 0.0)
                    reason = item.get("reason")
                else:
                    raw_id = fallback_idx
                    score = item
                    reason = None
                doc_id = int(raw_id)
                if doc_id < 1 or doc_id > len(results) or doc_id in by_id:
                    return None
                by_id[doc_id] = {
                    "id": doc_id,
                    "score": max(0.0, min(1.0, float(score))),
                    "reason": str(reason)[:80] if reason else None,
                }
            if set(by_id) != set(range(1, len(results) + 1)):
                return None
            return [by_id[idx] for idx in range(1, len(results) + 1)]
        except Exception as exc:
            logger.warning(f"Could not parse LLM rerank scores: {exc}; raw={raw[:200]}")
            return None

    def _call_llm(
        self,
        system_msg: str,
        user_msg: str,
        config: PostRetrievalConfig,
        max_tokens: int,
    ) -> str | None:
        try:
            if config.llm_provider == "ollama":
                return self._call_ollama(system_msg, user_msg, config.llm_model, max_tokens)

            from services.generation_service import GenerationService

            api_key = config.api_key or self._env_api_key(config.llm_provider)
            service = GenerationService()
            return service.generate_raw(
                provider=config.llm_provider,
                model_name=config.llm_model,
                system_msg=system_msg,
                user_msg=user_msg,
                api_key=api_key,
                generation_params={"temperature": 0.1, "top_p": 0.85, "max_tokens": max_tokens},
            )
        except Exception as exc:
            logger.warning(f"LLM post-retrieval call failed: {exc}")
            return None

    @staticmethod
    def _call_ollama(system_msg: str, user_msg: str, model_name: str, max_tokens: int) -> str:
        import requests

        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.85,
                "num_predict": max_tokens,
            },
        }
        last_error = None
        timeout = int(os.getenv("OLLAMA_TIMEOUT", str(max(30, min(120, max_tokens // 4)))))
        session = requests.Session()
        session.trust_env = False
        for _ in range(2):
            try:
                response = session.post(
                    f"{host}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                return (data.get("message") or {}).get("content", "").strip()
            except requests.RequestException as exc:
                last_error = exc
                if getattr(exc, "response", None) is None or exc.response.status_code not in {502, 503, 504}:
                    raise
        raise last_error

    def _cross_encoder_scores(self, pairs: List[List[str]], model_name: str) -> List[float] | None:
        try:
            from sentence_transformers import CrossEncoder

            resolved_model = get_huggingface_model_path(model_name)
            if resolved_model not in self._rerankers:
                logger.info(f"Loading reranker model: {resolved_model}")
                self._rerankers[resolved_model] = CrossEncoder(resolved_model)
            raw_scores = self._rerankers[resolved_model].predict(pairs)
            return [self._sigmoid(float(score)) for score in raw_scores]
        except Exception as exc:
            logger.warning(f"CrossEncoder rerank failed, falling back to lexical score: {exc}")
            return None

    @staticmethod
    def _env_api_key(provider: str) -> str | None:
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        if provider == "aliyun":
            return os.getenv("DASHSCOPE_API_KEY")
        return None

    @staticmethod
    def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
        score = float(item.get("score", 0.0))
        metadata = dict(item.get("metadata") or {})
        normalized = {
            "text": item.get("text", ""),
            "score": score,
            "metadata": metadata,
        }
        normalized["metadata"].setdefault("retrieval_score", score)
        normalized["metadata"].setdefault("original_score", score)
        normalized["metadata"].setdefault("final_score", score)
        return normalized

    def _result_identity_key(self, item: Dict[str, Any]) -> str:
        meta = item.get("metadata", {}) or {}
        text_fingerprint = self._fingerprint(item.get("text", ""))
        doc_key = self._metadata_doc_key(meta)
        parent_id = self._metadata_value(meta, "parent_id")
        chunk_key = self._metadata_locator_key(meta)

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

    @classmethod
    def _metadata_doc_key(cls, meta: Dict[str, Any]) -> str:
        parts = [
            cls._metadata_value(meta, "source"),
            cls._metadata_value(meta, "document_name"),
            cls._metadata_value(meta, "file_name"),
            cls._metadata_value(meta, "filename"),
        ]
        return "|".join(part for part in parts if part)

    @classmethod
    def _metadata_locator_key(cls, meta: Dict[str, Any]) -> str:
        parts = [
            cls._metadata_value(meta, "page"),
            cls._metadata_value(meta, "page_range"),
            cls._metadata_value(meta, "chunk"),
            cls._metadata_value(meta, "chunk_id"),
        ]
        return "|".join(part for part in parts if part)

    @staticmethod
    def _metadata_value(meta: Dict[str, Any], key: str) -> str:
        value = meta.get(key)
        if value is None:
            return ""
        text = str(value).strip()
        return text

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = re.sub(r"\s+", "", text).lower()
        return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1 / (1 + z)
        z = math.exp(value)
        return z / (1 + z)

    @staticmethod
    def _terms(text: str) -> set[str]:
        text = text.lower()
        ascii_terms = re.findall(r"[a-z0-9_]{2,}", text)
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        return set(ascii_terms + chinese_terms + chars)

    def _lexical_score(self, query: str, text: str) -> float:
        return self._term_overlap_score(self._terms(query), text)

    def _term_overlap_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = self._terms(text)
        if not text_terms:
            return 0.0
        overlap = len(query_terms & text_terms)
        return min(1.0, overlap / max(1, len(query_terms)))

    def _rank_sentences(self, query: str, sentences: List[str]) -> List[tuple[float, int]]:
        query_terms = self._tokenize_terms(query)
        sentence_terms = [self._tokenize_terms(sentence) for sentence in sentences]
        if not query_terms:
            return [(0.0, idx) for idx in range(len(sentences))]

        doc_count = len(sentences)
        avg_len = sum(len(terms) for terms in sentence_terms) / max(1, doc_count)
        doc_freq: Dict[str, int] = {}
        for terms in sentence_terms:
            for term in set(terms):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        ranked = []
        query_set = set(query_terms)
        for idx, terms in enumerate(sentence_terms):
            if not terms:
                ranked.append((0.0, idx))
                continue
            term_counts: Dict[str, int] = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1

            bm25 = 0.0
            for term in query_set:
                if term not in term_counts:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                tf = term_counts[term]
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(terms) / max(1.0, avg_len))
                bm25 += idf * (tf * 2.5 / denom)

            overlap = len(query_set & set(terms)) / max(1, len(query_set))
            position_bonus = 0.05 if idx == 0 else 0.0
            ranked.append((bm25 + overlap + position_bonus, idx))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    @staticmethod
    def _tokenize_terms(text: str) -> List[str]:
        lowered = text.lower()
        tokens: List[str] = []
        try:
            import jieba

            tokens.extend(token.strip() for token in jieba.cut(lowered) if len(token.strip()) >= 2)
        except Exception:
            pass
        tokens.extend(re.findall(r"[a-z0-9_]{2,}", lowered))
        tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
        return [token for token in tokens if token]

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        terms_a = self._terms(text_a)
        terms_b = self._terms(text_b)
        if not terms_a or not terms_b:
            return 0.0
        return len(terms_a & terms_b) / len(terms_a | terms_b)

    @staticmethod
    def _extract_json_array(text: str) -> str:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON array found")
        return text[start : end + 1]

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        return text[start : end + 1]

    def _parse_compression_payload(self, raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(self._extract_json_object(raw))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        text = raw.strip()
        return {
            "relevant": text not in {"无关", "无关。"},
            "compressed_text": "" if text in {"无关", "无关。"} else text,
            "kept_facts": [],
        }

    @staticmethod
    def _parse_bool(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "是", "相关"}:
                return True
            if normalized in {"false", "no", "n", "0", "否", "无关", "不相关"}:
                return False
        return default

    @staticmethod
    def _rerank_fallback_from(results: List[Dict[str, Any]]) -> str | None:
        for item in results:
            fallback_from = (item.get("metadata", {}) or {}).get("rerank_fallback_from")
            if fallback_from:
                return str(fallback_from)
        return None

    @staticmethod
    def _compression_trace_extra(results: List[Dict[str, Any]], config: PostRetrievalConfig) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"method": config.compress_method}
        if config.compress_method != "llm":
            return extra

        extra["llm_compress_top_n"] = config.llm_compress_top_n
        counters = {
            "llm_compressed": 0,
            "llm_skipped_rank": 0,
            "llm_skipped_short": 0,
            "llm_rejected": 0,
            "llm_dropped": 0,
            "llm_failed": 0,
        }
        for item in results:
            meta = item.get("metadata", {}) or {}
            reason = meta.get("postprocess_reason") or ""
            if meta.get("compressed") and "llm_compress" in reason:
                counters["llm_compressed"] += 1
            if meta.get("llm_compress_skipped_rank") is not None:
                counters["llm_skipped_rank"] += 1
            if meta.get("llm_compress_skipped_length") is not None:
                counters["llm_skipped_short"] += 1
            if meta.get("llm_compress_rejected_reason") is not None or "llm_compress_rejected" in reason:
                counters["llm_rejected"] += 1
            if "llm_compress_drop" in reason:
                counters["llm_dropped"] += 1
            if "llm_compress_failed" in reason:
                counters["llm_failed"] += 1
        extra.update(counters)
        return extra

    @staticmethod
    def _threshold_score(item: Dict[str, Any]) -> float:
        meta = item.get("metadata", {}) or {}
        if meta.get("rerank_score") is not None:
            return float(meta.get("final_score", item.get("score", 0.0)))
        if meta.get("fusion_score_normalized") is not None:
            return float(meta["fusion_score_normalized"])
        if meta.get("retrieval_score") is not None:
            return float(meta.get("retrieval_score", item.get("score", 0.0)))
        return float(item.get("score", 0.0))

    @staticmethod
    def _count_tokens(text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # 粗略估计：中文按 1 字 1 token，英文约 4 字符 1 token。
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            other_chars = max(0, len(text) - chinese_chars)
            return chinese_chars + math.ceil(other_chars / 4)

    def _truncate_to_budget(self, text: str, max_chars: int, max_tokens: int) -> str:
        sentences = self._split_sentences(text)
        if sentences:
            kept = []
            for sentence in sentences:
                candidate = "".join(kept + [sentence]).strip()
                if len(candidate) > max_chars or self._count_tokens(candidate) > max_tokens:
                    break
                kept.append(sentence)
            if kept:
                return "".join(kept).strip()

        candidate = text[:max_chars].rstrip()
        while candidate and self._count_tokens(candidate) > max_tokens:
            shrink_to = max(1, int(len(candidate) * 0.85))
            candidate = candidate[:shrink_to].rstrip()
        return candidate

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？!?；;.\n])\s*", text)
        return [part.strip() for part in parts if part and part.strip()]

    @staticmethod
    def _append_reason(existing: str | None, reason: str) -> str:
        if not existing:
            return reason
        parts = existing.split("+")
        if reason in parts:
            return existing
        return f"{existing}+{reason}"

    @staticmethod
    def _trace(
        trace: List[Dict[str, Any]],
        step: str,
        input_count: int,
        output_count: int,
        started: float | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "step": step,
            "input_count": input_count,
            "output_count": output_count,
        }
        if started is not None:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if extra:
            entry.update(extra)
        trace.append(entry)
