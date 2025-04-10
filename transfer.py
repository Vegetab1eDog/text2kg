import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# 初始化本地模型（需提前运行 ollama serve,并下载deepseek-r1:14b模型）
model_name = "deepseek-r1:14b"

# 定义系统提示词（关键）
sys_prompt = """你是一个知识图谱提取专家，请从文本中提取以下内容：
1. 识别具体实体（人物、地点、组织、概念）
2. 提取实体间的关系（动词或动作）
3. 用三元组格式输出：[实体1] - [关系] -> [实体2]
示例输出：
玛丽 - 拥有 -> 小羊羔
小羊羔 - 是 -> 动物
玛丽 - 传递 -> 盘子"""

# 文本分块处理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
with open("text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()
chunks = text_splitter.split_text(full_text)

# 创建数据存储结构
kg_df = pd.DataFrame(columns=["node1", "relation", "node2", "weight", "chunk_id"])

# 知识提取主流程
for idx, chunk in tqdm(enumerate(chunks)):
    # 调用本地模型
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "system",
            "content": sys_prompt
        }, {
            "role": "user",
            "content": chunk
        }]
    )
    
    # 解析输出
    triples = []
    for line in response['message']['content'].split("\n"):
        if "->" in line:
            parts = line.strip().split("->")
            left = parts[0].split("-")
            node1 = left[0].strip()
            relation = left[1].replace("[", "").replace("]", "").strip()
            node2 = parts[1].strip()
            triples.append((node1, relation, node2))
    
    # 构建DataFrame
    temp_df = pd.DataFrame(triples, columns=["node1", "relation", "node2"])
    temp_df["weight"] = 1.0  # 初始权重
    temp_df["chunk_id"] = idx
    
    kg_df = pd.concat([kg_df, temp_df])

# 合并相似关系
merged_df = kg_df.groupby(["node1", "node2"]).agg({
    "relation": lambda x: "|".join(set(x)),
    "weight": "sum"
}).reset_index()

# 创建知识图谱
G = nx.DiGraph()
for _, row in merged_df.iterrows():
    G.add_edge(row["node1"], row["node2"], 
              relation=row["relation"],
              weight=row["weight"])

# 可视化展示
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10,
        edge_color="gray", width=[d['weight']*0.5 for (u,v,d) in G.edges(data=True)])
edge_labels = {(u,v): d["relation"] for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()