from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
documents = SimpleDirectoryReader(input_files=["...txt"]).load_data()

# 创建语义分块器
splitter = SemanticSplitterNodeParser(
    buffer_size=1,  # 缓冲区大小
    breakpoint_percentile_threshold=85, # 断点百分位阈值
    embed_model=embed_model    # 使用的嵌入模型
)
# 创建基础句子分块器（作为对照）
base_splitter = SentenceSplitter(
    chunk_size=512
)

# 使用语义分块器对文档进行分块
semantic_nodes = splitter.get_nodes_from_documents(documents)
print("\n=== 语义分块结果 ===")
print(f"语义分块器生成的块数：{len(semantic_nodes)}")
for i, node in enumerate(semantic_nodes, 1):
    print(f"\n--- 第 {i} 个语义块 ---")
    print(f"内容:\n{node.text}")
    print("-" * 50)


# 使用基础句子分块器对文档进行分块
base_nodes = base_splitter.get_nodes_from_documents(documents)
print("\n=== 基础句子分块结果 ===")
print(f"基础句子分块器生成的块数：{len(base_nodes)}")
for i, node in enumerate(base_nodes, 1):
    print(f"\n--- 第 {i} 个句子块 ---")
    print(f"内容:\n{node.text}")
    print("-" * 50)