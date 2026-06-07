# Post-Retrieval Optimization Summary

本文档总结当前分支中最近两次功能提交的主要改动。

## Commit 1: Add post-retrieval optimization pipeline

提交哈希：`ac60e4eb`

该提交建立了检索后优化的核心后端能力，覆盖向量召回之后、RAG 生成之前的结果加工流程。

主要改动：

- 新增 `PostRetrievalService`，集中处理去重、重排、压缩、多样性选择和上下文打包。
- 支持本地 CrossEncoder 重排，以及基于 Ollama `qwen2.5:3b` 的本地 LLM 重排。
- 支持抽取式压缩和 LLM 压缩，LLM 压缩默认只处理前 N 条结果，降低本地模型开销。
- 为 LLM 调用失败、解析失败或压缩结果不安全的情况增加保守回退，避免静默删除可用上下文。
- 在搜索入口中增加候选召回数量和候选阈值逻辑，使重排前能拿到更宽的候选集合。
- 新增本地 Ollama 后处理测试脚本，便于验证本地模型重排和压缩链路。

## Commit 2: Improve post-retrieval optimization flow

提交哈希：`6989f054`

该提交完善了后端接口、前端配置和可观测性，让检索后优化可以在 Search、QA、Generation 三个主要页面中稳定使用。

主要改动：

- 在 `/search`、`/qa`、`/generate` 响应中返回 `postprocess_trace`，用于展示每一步优化的输入输出数量和耗时。
- 新增前端 `PostprocessTrace` 组件，统一展示标准化、去重、重排、阈值过滤、压缩、上下文打包和最终截断信息。
- Search、QA、Generation 页面增加检索后优化预设，包括质量优先、速度优先和本地离线模式。
- 前端暴露候选数、重排后数量、重排方式、压缩方式、LLM provider/model、LLM 压缩前 N 条和上下文预算等参数。
- 保存搜索结果时一并保存 `postprocess_trace`，Generation 页面加载保存文件时可恢复 Trace。
- 修复 `/generate` 二次后处理默认按 `rerank_top_k` 截断上下文的问题，避免生成页无意丢失证据。
- 增加后处理评估脚本，用于比较 baseline、抽取式后处理和本地 Ollama 后处理方案。

## Current Behavior

当前检索后优化链路支持以下策略组合：

- `deduplicate`：基于文档、页码、块编号、父块和文本指纹去重。
- `rerank`：支持 CrossEncoder、LLM 重排，并在失败时降级为词面相关性重排。
- `compress`：支持抽取式压缩和 LLM 压缩。
- `diversify`：使用轻量 MMR 平衡相关性和多样性。
- `context_pack`：按字符预算和 token 预算打包最终上下文。

## Verification

已执行并通过：

- `python3 -m py_compile backend/main.py backend/schemas.py backend/services/post_retrieval_service.py backend/services/search_service.py backend/scripts/test_ollama_postprocess.py backend/scripts/evaluate_postprocess.py`
- `npm run build`

前端构建仅提示 Browserslist 数据过期，不影响当前功能。
