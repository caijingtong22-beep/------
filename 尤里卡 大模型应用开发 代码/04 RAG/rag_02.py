# %%
# 导入所需库
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

# 3. 准备示例文档
documents = [
    "尤里卡的大模型应用开发课程需要的前置知识是Python。",
    "具身智能导学课程的前置知识是多模态大模型。",
    "尤里卡是中南大学的博士研究生，硕士毕业于中国科学院大学，同时是一名创业者。",
    "尤里卡在读博之前，曾在工业界工作了6年，其中1年在创业。目前一边读博，一边做AI前沿技术的培训",
    "目前尤里卡主要运营的社交平台包括：bilibili、小红书。"
]

model = BGEM3FlagModel('/home/youlika/models/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

user_query = "尤里卡的学历是？"

embeddings_1 = model.encode(user_query, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(documents)['dense_vecs']

similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# %%
top_k = 2
most_similar_indices = np.argsort(similarity)[-top_k:][::-1]
relevant_docs = [documents[i] for i in most_similar_indices]
relevant_docs

# %%
# 1. 初始化设置
api_key = "sk-0c695e12d5fd47cfa89d1cfc33bd9671"  # 替换为你的API密钥
model_name = "deepseek-chat"
api_url = "https://api.deepseek.com"

# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url=api_url)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_query},
    ],
    stream=False
)

print(response.choices[0].message.content)

# %%
# 9. 构建提示
prompt = f"""基于以下上下文信息回答问题。如果上下文不足以回答问题，请说明。

上下文:
{''.join([f'- {doc}\n' for doc in relevant_docs])}

问题: {user_query}

回答:"""
print("输入给大模型的prompt是：", prompt)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ],
    stream=False
)

print(response.choices[0].message.content)