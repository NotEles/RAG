import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from openai import OpenAI
import requests
from utils.model_utils import get_huggingface_model_path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import time
from services.prompt_service import PromptService

# 设置环境变量以启用 Apple Silicon (MPS) 回退到 CPU (当遇到不支持的操作时会自动回退到 CPU 执行)
# 目前 PyTorch 版本 ≥ 1.13 时，才支持 Apple 的 Metal Performance Shaders (MPS) ，而且暂不支持「多 GPU」，另外，部分训练操作尚未完全实现
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)


class GenerationService:
    """
    生成服务类：负责调用不同的模型提供商（HuggingFace、OpenAI、DeepSeek）生成回答
    支持本地模型和API调用，并将生成结果保存到文件
    """

    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和创建输出目录
        """
        self.model = ""
        self.tokenizer = ""
        self.history = ""
        self.prompt_service = PromptService()
        self.models = {
            "huggingface": {
                "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
            },
            "openai": {
                "gpt-4o": "gpt-4o",
                "gpt-4o-mini": "gpt-4o-mini",
                "gpt-3.5-turbo": "gpt-3.5-turbo",
            },
            "aliyun": {
                "qwen-turbo": "qwen-turbo",
                "qwen-plus": "qwen-plus",
            },
            "deepseek": {
                "deepseek-v3": "deepseek-chat",
                "deepseek-r1": "deepseek-reasoner",
            }
        }

        # 确保输出目录存在
        os.makedirs("05-generation-results", exist_ok=True)

    def _load_huggingface_model(self, model_name: str):
        """
        加载HuggingFace模型

        参数:
            model_name: 模型名称，对应self.models["huggingface"]中的键

        返回:
            model: 加载的模型
            tokenizer: 对应的分词器
        """
        try:
            tensor_device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.models["huggingface"][model_name]
            model_name = get_huggingface_model_path(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=tensor_device,  # 自动分配GPU/CPU
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,  # 控制生成文本的最大长度
                temperature=0.7,  # 控制生成随机性
                top_k=50,  # 限制候选词数量
                top_p=0.9,  # 核采样参数
                num_return_sequences=1,  # 生成单个序列
                truncation=True,  # 启用输入截断
                pad_token_id=tokenizer.eos_token_id,  # 填充token设置
                clean_up_tokenization_spaces=False  # 保留原始分词空格
            )

            hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
            chat_model = ChatHuggingFace(llm=hf_pipeline)

            return chat_model, tokenizer
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise

    def _generate_with_huggingface(
            self,
            model_name: str,
            query: str,
            context: str,
            load_model: bool,
            system_msg: str,
            user_msg: str,
            generation_params: dict = None,
    ) -> str:
        """
        使用HuggingFace模型生成回答

        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            system_msg: 系统角色 prompt（由 PromptService 生成）
            user_msg: 用户消息 prompt（由 PromptService 生成）
            generation_params: 任务相关的生成参数（temperature, top_p, max_new_tokens）

        返回:
            生成的回答文本
        """
        try:
            # 自动加载模型：load_model=True 或模型尚未加载时自动加载
            if load_model or not isinstance(self.model, ChatHuggingFace):
                self.model, self.tokenizer = self._load_huggingface_model(model_name)

            # 使用 ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("human", user_msg),
            ])

            # 合并任务参数（任务参数优先于默认值）
            params = generation_params or {}
            if params and hasattr(self.model, 'llm') and hasattr(self.model.llm, 'pipeline'):
                pipe = self.model.llm.pipeline
                if "temperature" in params:
                    pipe.temperature = params["temperature"]
                if "top_p" in params:
                    pipe.top_p = params["top_p"]
                if "max_tokens" in params:
                    pipe.max_new_tokens = params["max_tokens"]

            ts = time.time()
            answer = self.model.invoke(prompt.format_messages())
            spent_sec = int(time.time() - ts)
            text = answer.content

            # 解析思维链（部分小模型可能不输出 <think> 标签）
            if "<think>" in text and "</think>" in text:
                text_parts = text.split("<think>")
                parts = text_parts[1].split("</think>")
                thinkingInfo = parts[0].strip()
                responseInfo = parts[1].strip()
            else:
                thinkingInfo = ""
                responseInfo = text.strip()

            self.history += f'用户提问：{query}\nAI回复：{responseInfo}\n\n'

            if thinkingInfo:
                answer_content = f's:{spent_sec}\n用户提问：{query}\nAI思考过程：{thinkingInfo}\nAI回复：{responseInfo}\n\n'
            else:
                answer_content = f's:{spent_sec}\n用户提问：{query}\nAI回复：{responseInfo}\n\n'
            return answer_content

        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            raise

    def _generate_with_openai(
            self,
            model_name: str,
            query: str,
            context: str,
            api_key: Optional[str] = None,
            system_msg: str = "",
            user_msg: str = "",
            generation_params: dict = None,
    ) -> str:
        try:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
            client = OpenAI(api_key=api_key)

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            params = generation_params or {}
            response = client.chat.completions.create(
                model=self.models["openai"][model_name],
                messages=messages,
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                max_tokens=params.get("max_tokens", 1024),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise

    def _generate_with_aliyun(
            self,
            model_name: str,
            query: str,
            context: str,
            api_key: Optional[str] = None,
            system_msg: str = "",
            user_msg: str = "",
            generation_params: dict = None,
    ) -> str:
        """
        使用阿里云 DashScope API 生成回答（OpenAI 兼容模式，流式输出）
        """
        try:
            if not api_key:
                api_key = os.getenv("DASHSCOPE_API_KEY")
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            params = generation_params or {}
            completion = client.chat.completions.create(
                model=self.models["aliyun"][model_name],
                messages=messages,
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                max_tokens=params.get("max_tokens", 1024),
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={"include_usage": True},
            )
            reasoning_content = ""
            answer_content = ""
            is_answering = False
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

            for chunk in completion:
                if not chunk.choices:
                    print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                    print(chunk.usage)
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end="", flush=True)
                    answer_content += delta.content
            return answer_content.strip()

        except Exception as e:
            logger.error(f"Error generating with Aliyun: {str(e)}")
            raise

    def _generate_with_deepseek(
            self,
            model_name: str,
            query: str,
            context: str,
            api_key: Optional[str] = None,
            show_reasoning: bool = True,
            system_msg: str = "",
            user_msg: str = "",
            generation_params: dict = None,
    ) -> str:
        """
        使用DeepSeek API生成回答
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            params = generation_params or {}
            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                max_tokens=params.get("max_tokens", 1024),
                stream=False,
            )

            if model_name == "deepseek-r1":
                message = response.choices[0].message
                reasoning = message.reasoning_content
                answer = message.content

                if show_reasoning and reasoning:
                    return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"
                return answer

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            raise

    def generate(
            self,
            provider: str,
            model_name: str,
            query: str,
            search_results: List[Dict],
            load_model: bool,
            api_key: Optional[str] = None,
            show_reasoning: bool = True,
            task_type: Optional[str] = None,
    ) -> Dict:
        """
        生成回答并保存结果

        参数:
            provider: 模型提供商
            model_name: 模型名称
            query: 用户查询
            search_results: 搜索结果列表，用于构建上下文
            api_key: API密钥（对于API调用）
            show_reasoning: 是否显示推理过程（仅对DeepSeek推理模型有效）
            load_model: 是否装载模型
            task_type: 任务类型（qa/summarize/analyze/compare/explain/creative），None 则自动检测

        返回:
            包含生成回答和保存路径的字典
        """
        try:
            # 准备上下文
            context = "\n\n".join([
                f"[Source {i + 1}]: {result['text']}"
                for i, result in enumerate(search_results)
            ])

            # 通过 PromptService 获取任务专属的 generation_params
            prompt_info = self.prompt_service.get_prompt(
                task_type=task_type,
                query=query,
                context=context,
                model_name=self.models.get(provider, {}).get(model_name, model_name),
            )
            gen_params = prompt_info["generation_params"]
            detected_task = prompt_info["detected_task"]
            system_msg = prompt_info["system_msg"]
            user_msg = prompt_info["user_msg"]
            logger.info(f"Task detected: {detected_task}, params: {gen_params}")

            ts = time.time()
            # 根据不同提供商生成回答，传递 system_msg/user_msg/generation_params
            if provider == "huggingface":
                response = self._generate_with_huggingface(
                    model_name, query, context, load_model,
                    system_msg=system_msg, user_msg=user_msg, generation_params=gen_params
                )
            elif provider == "openai":
                response = self._generate_with_openai(
                    model_name, query, context, api_key,
                    system_msg=system_msg, user_msg=user_msg, generation_params=gen_params
                )
            elif provider == "aliyun":
                response = self._generate_with_aliyun(
                    model_name, query, context, api_key,
                    system_msg=system_msg, user_msg=user_msg, generation_params=gen_params
                )
            elif provider == "deepseek":
                response = self._generate_with_deepseek(
                    model_name, query, context, api_key, show_reasoning,
                    system_msg=system_msg, user_msg=user_msg, generation_params=gen_params
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # 准备保存的结果
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "task_type": detected_task,
                "generation_params": gen_params,
                "response": response,
                "context": search_results,
            }

            # 生成文件名并保存
            spend_sec = int(time.time() - ts)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generation_{provider}_{model_name}_{detected_task}_{spend_sec}s_{timestamp}.json"
            filepath = os.path.join("05-generation-results", filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return {
                "response": response,
                "saved_filepath": filepath,
                "detected_task": detected_task,
            }

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise

    def generate_raw(
            self,
            provider: str,
            model_name: str,
            system_msg: str,
            user_msg: str,
            api_key: Optional[str] = None,
            generation_params: dict = None,
    ) -> str:
        """
        通用 LLM 调用方法，供 QueryService 查询优化使用。
        与 generate() 不同：不构建 RAG 上下文、不保存文件、直接返回原始文本。

        参数:
            provider: 模型提供商 (openai | deepseek | aliyun | huggingface)
            model_name: 模型名称
            system_msg: 系统角色 prompt
            user_msg: 用户消息 prompt
            api_key: API 密钥
            generation_params: 可选的生成参数 (temperature, top_p, max_tokens)

        返回:
            LLM 生成的原始文本
        """
        # 用占位参数填充原有方法签名，避免重复实现
        dummy_query = user_msg
        dummy_context = ""

        if provider == "huggingface":
            return self._generate_with_huggingface(
                model_name, dummy_query, dummy_context, load_model=False,
                system_msg=system_msg, user_msg=user_msg,
                generation_params=generation_params,
            )
        elif provider == "openai":
            return self._generate_with_openai(
                model_name, dummy_query, dummy_context, api_key,
                system_msg=system_msg, user_msg=user_msg,
                generation_params=generation_params,
            )
        elif provider == "aliyun":
            return self._generate_with_aliyun(
                model_name, dummy_query, dummy_context, api_key,
                system_msg=system_msg, user_msg=user_msg,
                generation_params=generation_params,
            )
        elif provider == "deepseek":
            return self._generate_with_deepseek(
                model_name, dummy_query, dummy_context, api_key,
                show_reasoning=False,
                system_msg=system_msg, user_msg=user_msg,
                generation_params=generation_params,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_available_models(self) -> Dict:
        """
        获取可用的模型列表

        返回:
            包含所有支持模型的字典
        """
        return self.models