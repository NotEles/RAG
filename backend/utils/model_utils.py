import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/models/ 相对于本文件的默认本地模型目录
_LOCAL_MODELS_DIR = Path(__file__).parent.parent / "models"

def get_huggingface_model_path(model_name: str) -> str:
    """
    将模型名称解析为本地路径（如果存在），否则返回原始名称供远程加载。

    解析顺序：
      1. $HF_MODEL_PATH/{org}/{model}   （带 org 前缀，兼容旧行为）
      2. $HF_MODEL_PATH/{model}         （仅模型文件夹名）
      3. backend/models/{model}         （项目内置本地目录）
      4. 原始 model_name                （远程下载）

    Args:
        model_name: HuggingFace 模型名，如 "BAAI/bge-small-zh-v1.5"
    """
    model_folder = model_name.split("/")[-1]  # 取最后一段，如 "bge-small-zh-v1.5"

    # 1 & 2: 优先检查 HF_MODEL_PATH 环境变量
    env_base = os.environ.get("HF_MODEL_PATH")
    if env_base and os.path.exists(env_base):
        # 1. 带 org 前缀：{HF_MODEL_PATH}/BAAI/bge-small-zh-v1.5
        full_path = os.path.join(env_base, *model_name.split("/"))
        if os.path.exists(full_path):
            logger.info(f"Using local model: {full_path}")
            return full_path
        # 2. 仅文件夹名：{HF_MODEL_PATH}/bge-small-zh-v1.5
        short_path = os.path.join(env_base, model_folder)
        if os.path.exists(short_path):
            logger.info(f"Using local model: {short_path}")
            return short_path

    # 3. 项目内置 backend/models/{model_folder}
    default_path = _LOCAL_MODELS_DIR / model_folder
    if default_path.exists():
        logger.info(f"Using local model: {default_path}")
        return str(default_path)

    # 4. 远程加载
    logger.info(f"Using remote model: {model_name}")
    return model_name