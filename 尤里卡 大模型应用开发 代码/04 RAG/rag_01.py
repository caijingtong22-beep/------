# 导入所需库
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# 2. 加载嵌入模型
embedding_model = SentenceTransformer('/home/youlika/models/bert-base-uncased')

# 3. 准备示例文档
documents = [
    "尤里卡的大模型应用开发课程需要的前置知识是Python。",
    "具身智能导学课程的前置知识是多模态大模型。",
    "尤里卡是中南大学的博士研究生，硕士毕业于中国科学院大学，同时是一名创业者。",
    "尤里卡在读博之前，曾在工业界工作了6年，其中1年在创业。目前一边读博，一边做AI前沿技术的培训",
    "目前尤里卡主要运营的社交平台包括：bilibili、小红书。"
]

# 4. 为文档生成嵌入
document_embeddings = embedding_model.encode(documents, convert_to_tensor=False)
document_embeddings = np.array(document_embeddings)

# 1. 初始化设置
api_key = "your_deepseek_api_key_here"  # 替换为你的API密钥
model_name = "deepseek-chat"
api_url = "https://api.deepseek.com/v1/chat/completions"

# 5. 用户查询
user_query = "尤里卡的学历是？"

# 6. 为查询生成嵌入
query_embedding = embedding_model.encode(user_query, convert_to_tensor=False)

# 7. 计算相似度
similarities = np.dot(document_embeddings, query_embedding) / (
    np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
)

# 8. 获取最相关文档
top_k = 2
most_similar_indices = np.argsort(similarities)[-top_k:][::-1]
relevant_docs = [documents[i] for i in most_similar_indices]

# 9. 构建提示
prompt = f"""基于以下上下文信息回答问题。如果上下文不足以回答问题，请说明。

上下文:
{''.join([f'- {doc}\n' for doc in relevant_docs])}

问题: {user_query}

回答:"""

# 10. 调用DeepSeek API
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "你是一个有帮助的AI助手，基于提供的上下文回答问题。"},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}

# 11. 发送请求并获取响应
response = requests.post(api_url, headers=headers, data=json.dumps(payload))
response.raise_for_status()

# 12. 输出结果
generated_answer = response.json()["choices"][0]["message"]["content"]
print("问题:", user_query)
print("检索到的相关文档:", relevant_docs)
print("生成的回答:", generated_answer)