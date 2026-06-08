#!/usr/bin/env python3
"""Compare post-retrieval optimization variants on saved search results.

Usage:
    python scripts/evaluate_postprocess.py --file 04-search-results/search_xxx.json
    python scripts/evaluate_postprocess.py --query "RAG 为什么需要重排和压缩？"

The default variants avoid LLM calls. Add --use-llm to include the local
Ollama qwen2.5:3b quality pipeline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.post_retrieval_service import PostRetrievalConfig, PostRetrievalService


def sample_results() -> tuple[str, list[dict[str, Any]]]:
    query = "RAG 检索后优化为什么需要重排和压缩？"
    long_text = (
        "检索增强生成（RAG）系统通常先通过向量检索召回候选片段，然后在生成答案前进行检索后优化。"
        "常见优化包括去重、重排、多样性选择、上下文压缩和预算打包。"
        "重排会根据用户问题重新评估候选片段的相关性，降低向量召回误差带来的排序偏差。"
        "压缩会保留能回答问题的事实、术语、限制条件和因果关系，删除噪声内容。"
        "相比直接把所有召回片段放进上下文，这能减少 token 消耗、降低无关信息干扰，并改善答案质量。"
        "实现时需要注意本地模型延迟、输出格式约束、失败回退和上下文预算。"
    )
    return query, [
        {"text": long_text, "score": 0.63, "metadata": {"source": "rag.md", "retrieval_score": 0.63}},
        {"text": long_text, "score": 0.61, "metadata": {"source": "rag-copy.md", "retrieval_score": 0.61}},
        {
            "text": "向量数据库负责存储嵌入向量并支持近邻检索，但通常不负责后续的 LLM 重排或上下文压缩。",
            "score": 0.58,
            "metadata": {"source": "vector.md", "retrieval_score": 0.58},
        },
        {
            "text": "前端页面主题、按钮圆角和图标选择会影响用户体验，但与 RAG 后处理质量没有直接关系。",
            "score": 0.42,
            "metadata": {"source": "ui.md", "retrieval_score": 0.42},
        },
    ]


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("query", ""), data.get("results", [])


def count_tokens(service: PostRetrievalService, results: list[dict[str, Any]]) -> int:
    return sum(service._count_tokens(item.get("text", "")) for item in results)


def summarize(name: str, elapsed_ms: float, service: PostRetrievalService, before: list[dict], after: list[dict]) -> dict:
    before_tokens = count_tokens(service, before)
    after_tokens = count_tokens(service, after)
    scores = [float(item.get("score", 0.0)) for item in after]
    compressed = [item for item in after if item.get("metadata", {}).get("compressed")]
    return {
        "variant": name,
        "count": len(after),
        "tokens": after_tokens,
        "token_delta": after_tokens - before_tokens,
        "compression_ratio": round(after_tokens / max(1, before_tokens), 3),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0,
        "compressed_items": len(compressed),
        "elapsed_ms": round(elapsed_ms, 2),
    }


def run_variant(
    service: PostRetrievalService,
    name: str,
    query: str,
    results: list[dict[str, Any]],
    config: PostRetrievalConfig | None,
) -> dict:
    started = time.perf_counter()
    if config is None:
        output = [dict(item) for item in results]
    else:
        output, _trace = service.process(query, [dict(item) for item in results], config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return summarize(name, elapsed_ms, service, results, output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate post-retrieval optimization variants.")
    parser.add_argument("--file", type=Path, help="Saved search result JSON file.")
    parser.add_argument("--query", help="Override query for the sample or loaded file.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true", help="Include Ollama qwen2.5:3b rerank/compress variant.")
    args = parser.parse_args()

    if args.file:
        query, results = load_results(args.file)
    else:
        query, results = sample_results()
    if args.query:
        query = args.query

    service = PostRetrievalService()
    variants: list[tuple[str, PostRetrievalConfig | None]] = [
        ("baseline", None),
        (
            "dedupe+cross_rerank",
            PostRetrievalConfig(
                enabled=True,
                strategies=["deduplicate", "rerank", "context_pack"],
                rerank_method="cross_encoder",
                rerank_top_k=args.top_k,
                final_top_k=args.top_k,
                final_threshold=None,
                trace_enabled=True,
            ),
        ),
        (
            "dedupe+cross_rerank+extractive",
            PostRetrievalConfig(
                enabled=True,
                strategies=["deduplicate", "rerank", "compress", "context_pack"],
                rerank_method="cross_encoder",
                compress_method="extractive",
                rerank_top_k=args.top_k,
                final_top_k=args.top_k,
                final_threshold=None,
                trace_enabled=True,
            ),
        ),
    ]
    if args.use_llm:
        variants.append(
            (
                "local_ollama_quality",
                PostRetrievalConfig(
                    enabled=True,
                    strategies=["deduplicate", "rerank", "compress", "context_pack"],
                    rerank_method="llm",
                    compress_method="llm",
                    rerank_top_k=args.top_k,
                    final_top_k=args.top_k,
                    final_threshold=None,
                    llm_provider="ollama",
                    llm_model="qwen2.5:3b",
                    allow_drop_irrelevant=False,
                    trace_enabled=True,
                ),
            )
        )

    report = [run_variant(service, name, query, results, config) for name, config in variants]
    print(json.dumps({"query": query, "input_count": len(results), "variants": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
