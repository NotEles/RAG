"""
Prompt 模板引擎 — 基于任务类型的智能 prompt 构建 + 自动任务检测

功能：
- 6 种任务类型（QA/摘要/分析/对比/解释/创意），各有专属角色设定、输出格式、思维链策略
- 每种任务区分大模型和小模型（≤3B）prompt 变体
- 每种任务配置独立的 generation_params（temperature/top_p/max_tokens）
- 自动任务类型检测：基于关键词 + 正则
"""
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TaskType(str, Enum):
    QA = "qa"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    COMPARE = "compare"
    EXPLAIN = "explain"
    CREATIVE = "creative"


@dataclass
class PromptTemplate:
    system_prompt: str
    user_template: str           # 含 {context} 和 {query} 占位符
    small_model_system: str      # ≤3B 模型专用 system prompt（更简短明确）
    small_model_user: str        # ≤3B 模型专用 user template
    techniques: list[str] = field(default_factory=list)
    generation_params: dict = field(default_factory=dict)


# ──────────────────────── 模板注册表 ────────────────────────

TEMPLATES: dict[TaskType, PromptTemplate] = {

    TaskType.QA: PromptTemplate(
        system_prompt=(
            "你是一位专业的知识库问答助手。请严格依据提供的参考资料回答问题。\n"
            "要求：\n"
            "1. 先用一句话直接回答问题要点\n"
            "2. 再展开补充细节和依据\n"
            "3. 引用资料时标注来源编号，如 [1][2]\n"
            "4. 如果参考资料不足以完整回答，请明确说明哪些部分缺乏依据\n"
            "5. 回答使用中文，语言简洁专业"
        ),
        user_template=(
            "## 参考资料\n{context}\n\n## 用户问题\n{query}"
        ),
        small_model_system=(
            "你是知识库问答助手。根据参考资料简明回答问题。"
        ),
        small_model_user=(
            "参考资料：\n{context}\n\n问题：{query}\n\n回答："
        ),
        techniques=["structured", "citation", "role"],
        generation_params={"temperature": 0.3, "top_p": 0.9, "max_tokens": 1024},
    ),

    TaskType.SUMMARIZE: PromptTemplate(
        system_prompt=(
            "你是一位文档摘要专家。请对提供的内容进行结构化摘要。\n"
            "输出格式：\n"
            "## 一句话概括\n用一句话说明核心内容\n\n"
            "## 关键要点\n- 用列表形式列出 3-5 个关键要点\n\n"
            "## 重要细节\n补充需要特别注意的数据、结论或引用\n\n"
            "要求忠实原文，不添加原文没有的信息。"
        ),
        user_template=(
            "## 待摘要内容\n{context}\n\n## 摘要要求\n{query}"
        ),
        small_model_system=(
            "你是摘要助手。用3句话总结以下内容的要点。"
        ),
        small_model_user=(
            "内容：\n{context}\n\n要求：{query}\n\n总结："
        ),
        techniques=["structured", "hierarchical", "role"],
        generation_params={"temperature": 0.2, "top_p": 0.85, "max_tokens": 512},
    ),

    TaskType.ANALYZE: PromptTemplate(
        system_prompt=(
            "你是一位深度分析师。请按以下步骤分析问题：\n\n"
            "## 要素提取\n列出问题涉及的关键要素和变量\n\n"
            "## 逐步推理\n对每个要素进行分析，展示推理过程\n\n"
            "## 综合结论\n基于推理得出结论，并评估结论的可靠程度\n\n"
            "## 建议\n如有必要，给出可操作的建议\n\n"
            "要求逻辑清晰，推理有据，避免主观臆断。"
        ),
        user_template=(
            "## 分析材料\n{context}\n\n## 分析任务\n{query}"
        ),
        small_model_system=(
            "你是分析师。分析以下材料，先列出要点，再给出结论。"
        ),
        small_model_user=(
            "材料：\n{context}\n\n分析任务：{query}\n\n分析："
        ),
        techniques=["chain_of_thought", "structured", "role"],
        generation_params={"temperature": 0.4, "top_p": 0.9, "max_tokens": 1536},
    ),

    TaskType.COMPARE: PromptTemplate(
        system_prompt=(
            "你是一位比较分析专家。请对提供的内容进行系统对比。\n"
            "输出格式：\n\n"
            "## 相同点\n列出被比较对象的共同特征\n\n"
            "## 不同点\n用表格或列表对比关键差异：\n"
            "| 对比维度 | 对象A | 对象B |\n\n"
            "## 优劣势分析\n分别说明各自的优势和不足\n\n"
            "## 总结建议\n说明在什么场景下选择哪个更合适"
        ),
        user_template=(
            "## 对比材料\n{context}\n\n## 对比要求\n{query}"
        ),
        small_model_system=(
            "你是对比助手。对比以下内容，列出相同点和不同点。"
        ),
        small_model_user=(
            "内容：\n{context}\n\n对比要求：{query}\n\n对比结果："
        ),
        techniques=["structured", "table", "role"],
        generation_params={"temperature": 0.3, "top_p": 0.85, "max_tokens": 1024},
    ),

    TaskType.EXPLAIN: PromptTemplate(
        system_prompt=(
            "你是一位耐心的教师。请由浅入深地解释概念：\n\n"
            "## 简单定义\n用通俗易懂的语言解释核心概念\n\n"
            "## 举例说明\n给出一个贴近生活的例子帮助理解\n\n"
            "## 深入讲解\n展开技术细节或原理\n\n"
            "## 常见误区\n指出容易混淆或理解错误的地方（如有）\n\n"
            "要求语言平实，避免不必要的术语堆砌。"
        ),
        user_template=(
            "## 参考资料\n{context}\n\n## 解释要求\n{query}"
        ),
        small_model_system=(
            "你是教师。用简单语言解释以下概念，举一个例子。"
        ),
        small_model_user=(
            "资料：\n{context}\n\n问题：{query}\n\n解释："
        ),
        techniques=["progressive", "example", "role"],
        generation_params={"temperature": 0.4, "top_p": 0.9, "max_tokens": 1024},
    ),

    TaskType.CREATIVE: PromptTemplate(
        system_prompt=(
            "你是一位创意写作助手。请基于参考资料进行创意发挥。\n"
            "要求：\n"
            "1. 从多个角度思考，提供 2-3 种不同方案\n"
            "2. 每种方案附带简要理由\n"
            "3. 最后推荐最优方案并说明原因\n"
            "4. 使用 Markdown 格式输出，层次清晰\n\n"
            "鼓励创新思维，但方案需要有可行性依据。"
        ),
        user_template=(
            "## 参考资料\n{context}\n\n## 创意需求\n{query}"
        ),
        small_model_system=(
            "你是创意助手。根据资料，写出一种方案回答以下需求。"
        ),
        small_model_user=(
            "资料：\n{context}\n\n需求：{query}\n\n方案："
        ),
        techniques=["divergent", "structured", "role"],
        generation_params={"temperature": 0.8, "top_p": 0.95, "max_tokens": 2048},
    ),
}


# ──────────────────────── 任务自动检测 ────────────────────────

_DETECT_PATTERNS: list[tuple[TaskType, list[str]]] = [
    (TaskType.SUMMARIZE, ["总结", "摘要", "概括", "归纳", "简述", "summarize", "summary"]),
    (TaskType.COMPARE,   ["对比", "比较", "区别", "异同", "差异", "相比", "compare", "difference"]),
    (TaskType.ANALYZE,   ["分析", "为什么", "原因", "影响", "因素", "根因", "analyze", "why"]),
    (TaskType.EXPLAIN,   ["解释", "什么是", "定义", "概念", "含义", "原理", "explain", "what is"]),
    (TaskType.CREATIVE,  ["写", "创作", "设计", "方案", "建议", "头脑风暴", "创意", "create", "design"]),
]


def detect_task_type(query: str) -> TaskType:
    """基于关键词自动检测任务类型，返回最匹配的 TaskType，默认 QA"""
    q = query.lower()
    for task_type, keywords in _DETECT_PATTERNS:
        for kw in keywords:
            if kw in q:
                return task_type
    return TaskType.QA


# ──────────────────────── 小模型阈值 ────────────────────────

SMALL_MODEL_PARAMS = {
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen3-1.7B": 1.7,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 1.5,
}
SMALL_MODEL_THRESHOLD = 3  # ≤3B 视为小模型


def is_small_model(model_name: str) -> bool:
    """判断是否为小参数模型（≤3B）"""
    params = SMALL_MODEL_PARAMS.get(model_name, None)
    if params is not None:
        return params <= SMALL_MODEL_THRESHOLD
    # 尝试从名称猜测：含 0.6B/1.5B/1.7B 等标记
    m = re.search(r"(\d+\.?\d*)\s*[Bb]", model_name)
    if m:
        return float(m.group(1)) <= SMALL_MODEL_THRESHOLD
    return False


# ──────────────────────── PromptService ────────────────────────

class PromptService:

    def get_prompt(
        self,
        task_type: Optional[str],
        query: str,
        context: str,
        model_name: str = "",
    ) -> dict:
        """
        构建最终 prompt。

        Returns:
            {
                "system_msg": str,
                "user_msg": str,
                "generation_params": dict,
                "detected_task": str,       # 实际使用的任务类型
                "is_small_model": bool,
            }
        """
        # 解析任务类型
        if task_type and task_type != "auto":
            try:
                tt = TaskType(task_type)
            except ValueError:
                tt = detect_task_type(query)
        else:
            tt = detect_task_type(query)

        template = TEMPLATES[tt]
        small = is_small_model(model_name)

        if small:
            system_msg = template.small_model_system
            user_msg = template.small_model_user.format(context=context, query=query)
        else:
            system_msg = template.system_prompt
            user_msg = template.user_template.format(context=context, query=query)

        return {
            "system_msg": system_msg,
            "user_msg": user_msg,
            "generation_params": dict(template.generation_params),  # 返回副本
            "detected_task": tt.value,
            "is_small_model": small,
        }

    def get_task_types(self) -> list[dict]:
        """返回所有任务类型，供前端展示"""
        labels = {
            TaskType.QA: ("知识问答", "基于参考资料回答问题"),
            TaskType.SUMMARIZE: ("摘要总结", "提取核心要点，分层概括"),
            TaskType.ANALYZE: ("深度分析", "逐步推理，得出结论"),
            TaskType.COMPARE: ("对比比较", "系统对比异同和优劣"),
            TaskType.EXPLAIN: ("概念解释", "由浅入深讲解概念"),
            TaskType.CREATIVE: ("创意写作", "发散思维，多角度方案"),
        }
        result = [{"value": "auto", "label": "自动识别", "description": "系统根据问题内容自动选择最优任务类型"}]
        for tt in TaskType:
            label, desc = labels[tt]
            result.append({
                "value": tt.value,
                "label": label,
                "description": desc,
            })
        return result
