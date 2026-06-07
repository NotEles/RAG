#!/usr/bin/env python3
"""Smoke test for local Ollama post-retrieval rerank and compression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.post_retrieval_service import PostRetrievalConfig, PostRetrievalService


def build_results() -> list[dict]:
    rag_text = (
        "检索增强生成（RAG）系统通常先通过向量检索召回候选片段，然后在生成答案前进行检索后优化。"
        "常见优化包括去重、重排、多样性选择、上下文压缩和预算打包。"
        "LLM 重排会让模型根据用户问题判断每个片段的相关性，并返回归一化分数。"
        "LLM 压缩会保留能回答问题的事实、术语、限制条件和因果关系，删除噪声内容。"
        "相比直接把所有召回片段塞进上下文，这能降低上下文长度、减少无关信息干扰，并改善答案质量。"
        "实现时需要注意本地模型延迟、输出格式约束、失败回退和上下文 token 预算。"
        "例如，召回结果里可能混入重复段落、历史配置说明、与问题只弱相关的背景介绍，"
        "这些内容会消耗上下文窗口，还可能让生成模型偏离用户真正的问题。"
        "因此检索后优化通常会先保留高相关片段，再把片段压缩成更短的证据文本，"
        "最后按 token 或字符预算打包给生成阶段使用。"
    )
    vector_text = (
        "向量数据库负责存储嵌入向量并支持近邻检索。"
        "它可以用 HNSW、IVF、PQ 等索引结构提升查询效率。"
        "在 RAG 中，向量数据库通常处于召回阶段，但它本身不一定负责后续的 LLM 重排或压缩。"
    )
    unrelated_text = (
        "前端页面的颜色主题、按钮圆角和图标选择会影响用户体验。"
        "这些设计细节与 RAG 检索后优化的排序和压缩策略没有直接关系。"
    )
    return [
        {"text": rag_text, "score": 0.52, "metadata": {"source": "rag_postprocess.md"}},
        {"text": vector_text, "score": 0.61, "metadata": {"source": "vector_db.md"}},
        {"text": unrelated_text, "score": 0.49, "metadata": {"source": "ui_notes.md"}},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local Ollama LLM rerank/compress.")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name.")
    parser.add_argument("--query", default="RAG 检索后优化为什么需要重排和压缩？", help="Query to test.")
    args = parser.parse_args()

    config = PostRetrievalConfig(
        enabled=True,
        strategies=["deduplicate", "rerank", "compress", "context_pack"],
        rerank_method="llm",
        compress_method="llm",
        rerank_top_k=2,
        max_context_chars=900,
        max_context_tokens=450,
        llm_provider="ollama",
        llm_model=args.model,
    )

    service = PostRetrievalService()
    results = service.process(args.query, build_results(), config)

    print(f"model={args.model}")
    print(f"query={args.query}")
    print(f"result_count={len(results)}")
    for idx, item in enumerate(results, start=1):
        meta = item.get("metadata", {})
        text = item.get("text", "")
        print("-" * 72)
        print(f"#{idx} source={meta.get('source')} score={item.get('score'):.4f}")
        print(f"reason={meta.get('postprocess_reason')}")
        print(f"compressed={meta.get('compressed', False)} length={len(text)}")
        print(text[:500])

    if not results:
        print("ERROR: no post-processed results returned", file=sys.stderr)
        return 1
    if not any("llm_rerank" in (item.get("metadata", {}).get("postprocess_reason") or "") for item in results):
        print("ERROR: LLM rerank did not run", file=sys.stderr)
        return 1
    reasons = [item.get("metadata", {}).get("postprocess_reason") or "" for item in results]
    if not any("llm_compress" in reason for reason in reasons):
        print("ERROR: LLM compression was not attempted", file=sys.stderr)
        return 1
    if not any("+llm_compress+" in f"+{reason}+" for reason in reasons):
        print("WARN: LLM compression was attempted but rejected by safety checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
