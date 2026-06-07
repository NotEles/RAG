"""
查询处理服务 — 检索前查询侧优化的核心模块

支持的策略：
  - clean       : 规则去噪（去语气词、统一格式），无 LLM 调用
  - rewrite     : LLM 改写为检索友好的表述
  - decompose   : LLM 将复杂问题拆解为多个子问题
  - expand      : LLM 生成多个语义相近的查询变体
  - hyde        : LLM 生成假设文档片段，用于替代原始 query 做检索

设计原则：
  - 所有策略都是可选的，默认不启用（空策略列表 = 返回原始查询）
  - process() 返回 ProcessedQuery，其中 queries 字段即为最终用于检索的查询列表
  - 内置结果去重：相同或高度相似的查询会被合并
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.generation_service import GenerationService

logger = logging.getLogger(__name__)

# ──────────────────────── 数据结构 ────────────────────────


@dataclass
class ProcessedQuery:
    """查询处理结果"""
    original: str                             # 原始用户输入
    queries: List[str]                        # 最终用于检索的查询列表（去重后）
    strategies_applied: List[str]             # 实际应用的策略
    intermediates: dict = field(default_factory=dict)  # 中间结果（用于调试/日志）


# ──────────────────────── 规则清洗 ────────────────────────

# 中文语气词/礼貌用语（匹配后移除）
_POLITE_PATTERNS = [
    re.compile(r"^(请问[一下]*|那个[,，]*|我想问[一下]*[,，]*|能不能[帮我]*|"
               r"可以[帮我]*|麻烦[你您]*|帮忙[问]*|我想知道[,，]*)"),
    re.compile(r"(啊|吧|呢|吗|嘛|哈|哦|噢|呀|啦|呗|么)[？?！!。.]*$"),
    re.compile(r"[,，。.]*$"),
]

# 多余空白
_WHITESPACE_RE = re.compile(r"\s+")

# 全角转半角
_FULLWIDTH_MAP = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "０１２３４５６７８９"
    "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


def _clean_query(query: str) -> str:
    """
    规则化查询清洗（无 LLM 调用）。
    步骤：全角转半角 → 去礼貌前缀 → 去句末语气词 → 合并空白 → strip。
    """
    q = query.translate(_FULLWIDTH_MAP)
    for pat in _POLITE_PATTERNS:
        q = pat.sub("", q, count=1)
    q = _WHITESPACE_RE.sub(" ", q)
    return q.strip()


def _deduplicate_queries(queries: List[str]) -> List[str]:
    """去重：按原始顺序保留首次出现的查询，忽略大小写和首尾空白差异。"""
    seen = set()
    result = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q.strip())
    return result


# ──────────────────────── QueryService ────────────────────────


class QueryService:
    """
    查询处理服务，提供查询清洗、改写、分解、扩展和 HyDE 功能。

    使用方式：
        qs = QueryService()
        result = qs.process(
            query="请问 RAG 和微调有什么区别？",
            strategies=["clean", "rewrite"],
            provider="deepseek",
            model_name="deepseek-v3",
            api_key="sk-...",
        )
        # result.queries → ["RAG与微调的区别是什么？"]（去重后的检索查询列表）
    """

    # 需要 LLM 的策略集合
    _LLM_STRATEGIES = {"rewrite", "decompose", "expand", "hyde"}

    def __init__(self, generation_service: "GenerationService | None" = None):
        """
        参数:
            generation_service: 可选的 GenerationService 实例。
                               如不传入，将在首次需要时延迟创建。
        """
        self._gen = generation_service

    @property
    def gen(self):
        """延迟加载 GenerationService，避免循环导入和提前初始化。"""
        if self._gen is None:
            from services.generation_service import GenerationService
            self._gen = GenerationService()
        return self._gen

    # ── 公开方法 ──

    def clean(self, query: str) -> str:
        """规则清洗（无 LLM 调用）。"""
        result = _clean_query(query)
        logger.info(f"[clean] '{query[:60]}...' → '{result[:60]}...'")
        return result

    def rewrite(
        self,
        query: str,
        provider: str = "deepseek",
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
    ) -> str:
        """
        LLM 查询改写：将口语化问题改写为更适合向量检索的精确表述。

        返回: 改写后的单个查询字符串
        """
        from services.prompt_service import TaskType
        prompt_info = self.gen.prompt_service.get_prompt(
            task_type=TaskType.QUERY_REWRITE.value,
            query=query,
            context="",
            model_name=self.gen.models.get(provider, {}).get(model_name, model_name),
        )
        result = self.gen.generate_raw(
            provider=provider,
            model_name=model_name,
            system_msg=prompt_info["system_msg"],
            user_msg=prompt_info["user_msg"],
            api_key=api_key,
            generation_params=prompt_info["generation_params"],
        )
        rewritten = result.strip().strip('"').strip("'").strip()
        logger.info(f"[rewrite] '{query[:60]}...' → '{rewritten[:60]}...'")
        return rewritten if rewritten else query

    def decompose(
        self,
        query: str,
        provider: str = "deepseek",
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
    ) -> List[str]:
        """
        LLM 问题分解：将复杂问题拆解为多个独立的子问题。

        返回: 子问题列表（不包含原始查询）
        """
        from services.prompt_service import TaskType
        prompt_info = self.gen.prompt_service.get_prompt(
            task_type=TaskType.QUERY_DECOMPOSE.value,
            query=query,
            context="",
            model_name=self.gen.models.get(provider, {}).get(model_name, model_name),
        )
        result = self.gen.generate_raw(
            provider=provider,
            model_name=model_name,
            system_msg=prompt_info["system_msg"],
            user_msg=prompt_info["user_msg"],
            api_key=api_key,
            generation_params=prompt_info["generation_params"],
        )
        # 按行解析，过滤空行、编号前缀
        lines = []
        for line in result.strip().split("\n"):
            line = re.sub(r"^[\d]+[\.\)、\s]+", "", line).strip()
            if line and len(line) > 2:
                lines.append(line)
        logger.info(f"[decompose] '{query[:60]}...' → {len(lines)} sub-queries")
        return lines if lines else [query]

    def expand(
        self,
        query: str,
        provider: str = "deepseek",
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
    ) -> List[str]:
        """
        LLM 查询扩展：生成多个语义相近但表述不同的查询变体。

        返回: 查询变体列表（不包含原始查询，原始查询由 process() 统一添加）
        """
        from services.prompt_service import TaskType
        prompt_info = self.gen.prompt_service.get_prompt(
            task_type=TaskType.QUERY_EXPAND.value,
            query=query,
            context="",
            model_name=self.gen.models.get(provider, {}).get(model_name, model_name),
        )
        result = self.gen.generate_raw(
            provider=provider,
            model_name=model_name,
            system_msg=prompt_info["system_msg"],
            user_msg=prompt_info["user_msg"],
            api_key=api_key,
            generation_params=prompt_info["generation_params"],
        )
        lines = []
        for line in result.strip().split("\n"):
            line = re.sub(r"^[\d]+[\.\)、\s]+", "", line).strip()
            if line and len(line) > 2:
                lines.append(line)
        logger.info(f"[expand] '{query[:60]}...' → {len(lines)} variants")
        return lines

    def hyde_document(
        self,
        query: str,
        provider: str = "deepseek",
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
    ) -> str:
        """
        HyDE（假设文档嵌入）：生成一段假设的知识库文档片段，
        用这段文档片段替代原始 query 做向量检索。

        返回: 假设文档文本
        """
        from services.prompt_service import TaskType
        prompt_info = self.gen.prompt_service.get_prompt(
            task_type=TaskType.HYDE.value,
            query=query,
            context="",
            model_name=self.gen.models.get(provider, {}).get(model_name, model_name),
        )
        result = self.gen.generate_raw(
            provider=provider,
            model_name=model_name,
            system_msg=prompt_info["system_msg"],
            user_msg=prompt_info["user_msg"],
            api_key=api_key,
            generation_params=prompt_info["generation_params"],
        )
        doc = result.strip()
        logger.info(f"[hyde] '{query[:60]}...' → hypothetical doc ({len(doc)} chars)")
        return doc if doc else query

    # ── 策略编排 ──

    def process(
        self,
        query: str,
        strategies: List[str],
        provider: str = "deepseek",
        model_name: str = "deepseek-v3",
        api_key: Optional[str] = None,
    ) -> ProcessedQuery:
        """
        按 strategies 列表顺序依次处理查询。

        处理流程：
          clean → rewrite → decompose → expand → hyde
          （后一步的输入是前一步的输出）

        参数:
            query: 原始用户输入
            strategies: 策略列表，如 ["clean", "rewrite", "expand"]
            provider: LLM 提供商（仅当开启需要 LLM 的策略时使用）
            model_name: LLM 模型名称
            api_key: API Key

        返回:
            ProcessedQuery，其中 .queries 即为最终用于检索的查询列表

        ── 策略语义 ──
          clean     : 输入 1 个原始 query → 输出 1 个清洗后 query
          rewrite   : 输入 1 个 query → 输出 1 个改写后 query
          decompose : 输入 1 个 query → 输出 N 个子问题
          expand    : 对当前 queries 中的每个 query 进行扩展
          hyde      : 输入 1 个 query → 输出 1 个假设文档（代替 query）
        """
        if not strategies:
            return ProcessedQuery(
                original=query,
                queries=[query.strip()],
                strategies_applied=[],
            )

        intermediates = {"0_original": query}
        applied: List[str] = []
        # 当前活跃的查询池：初始只有一个原始 query
        pool: List[str] = [query]

        for strategy in strategies:
            s = strategy.strip().lower()
            if not s:
                continue

            applied.append(s)

            if s == "clean":
                pool = [_clean_query(q) for q in pool]
                intermediates["clean"] = list(pool)

            elif s == "rewrite":
                # 对 pool 中的每个 query 分别改写
                rewritten = []
                for q in pool:
                    r = self.rewrite(q, provider, model_name, api_key)
                    rewritten.append(r)
                pool = rewritten
                intermediates["rewrite"] = list(pool)

            elif s == "decompose":
                # 对 pool 中的每个 query 分别分解，展平结果
                all_subs = []
                for q in pool:
                    subs = self.decompose(q, provider, model_name, api_key)
                    all_subs.extend(subs)
                pool = all_subs if all_subs else pool
                intermediates["decompose"] = list(pool)

            elif s == "expand":
                # 对 pool 中的每个 query 分别扩展
                expanded = list(pool)  # 保留原始
                for q in pool:
                    variants = self.expand(q, provider, model_name, api_key)
                    expanded.extend(variants)
                pool = _deduplicate_queries(expanded)
                intermediates["expand"] = list(pool)

            elif s == "hyde":
                # HyDE：用假设文档替代查询
                hyde_docs = []
                for q in pool:
                    doc = self.hyde_document(q, provider, model_name, api_key)
                    hyde_docs.append(doc)
                pool = hyde_docs
                intermediates["hyde"] = list(pool)

            else:
                logger.warning(f"Unknown query strategy: '{s}', skipping")

        # 最终去重
        final_queries = _deduplicate_queries(pool)
        if not final_queries:
            final_queries = [query.strip()]

        return ProcessedQuery(
            original=query,
            queries=final_queries,
            strategies_applied=applied,
            intermediates=intermediates,
        )
