# Transformer Attention 机制理解笔记

## 1. 输入：Embedding Matrix

假设一句话：

> 我 爱 水 课

被 tokenizer 切成 4 个 token。

设：

- token 数量：
  
\[
n=4
\]

- embedding 维度：

\[
d_{model}=512
\]

则 embedding 后：

\[
X \in \mathbb{R}^{4\times512}
\]

即：

\[
X=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}
\]

其中：

- \(x_1\)：token “我”
- \(x_2\)：token “爱”
- \(x_3\)：token “水”
- \(x_4\)：token “课”

每个 token 都是一个 512 维向量。

---

# 2. 生成 Q / K / V

Transformer 会对输入做三个不同的线性投影：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

假设：

\[
W_Q,W_K,W_V \in \mathbb{R}^{512\times512}
\]

那么：

\[
Q,K,V \in \mathbb{R}^{4\times512}
\]

即：

- 输入是 4×512
- 输出仍然是 4×512

只是投影到了不同语义空间。

---

# 3. Q / K / V 的含义

## Q（Query）

表示：

> “我需要什么信息？”

每个 token 都会产生自己的 Query。

---

## K（Key）

表示：

> “我具有什么信息特征？”

K 不是人工定义的“标题”。

它是：

模型训练得到的特征表示。

---

## V（Value）

表示：

> “如果你关注我，我真正提供什么内容？”

V 才是真正被聚合的信息。

---

# 4. Attention 核心：QKᵀ

现在：

\[
Q\in\mathbb{R}^{4\times512}
\]

\[
K^T\in\mathbb{R}^{512\times4}
\]

因此：

\[
QK^T\in\mathbb{R}^{4\times4}
\]

---

# 5. Token-to-Token Attention Matrix

得到：

\[
S=QK^T
\]

即：

\[
S=
\begin{bmatrix}
q_1k_1 & q_1k_2 & q_1k_3 & q_1k_4\\
q_2k_1 & q_2k_2 & q_2k_3 & q_2k_4\\
q_3k_1 & q_3k_2 & q_3k_3 & q_3k_4\\
q_4k_1 & q_4k_2 & q_4k_3 & q_4k_4
\end{bmatrix}
\]

每个元素：

\[
s_{ij}=q_i\cdot k_j
\]

表示：

> “第 i 个 token 对第 j 个 token 的关注强度”

---

# 6. Attention 权重矩阵

经过 softmax：

\[
A=\text{softmax}(QK^T)
\]

仍然：

\[
A\in\mathbb{R}^{4\times4}
\]

例如：

\[
A=
\begin{bmatrix}
0.1&0.7&0.1&0.1\\
0.2&0.1&0.6&0.1\\
...&...&...&...
\end{bmatrix}
\]

---

## Attention 权重的含义

第一行：

表示：

> token1 对所有 token 的注意力分布

例如：

“我”：

- 70% 关注 “爱”
- 10% 关注 “水”
- 10% 关注 “课”

---

# 7. 真正输出不是 Attention Matrix

Attention 并不输出：

\[
4\times4
\]

真正输出是：

\[
AV
\]

因为：

\[
A\in\mathbb{R}^{4\times4}
\]

\[
V\in\mathbb{R}^{4\times512}
\]

所以：

\[
AV\in\mathbb{R}^{4\times512}
\]

shape 又回到了：

\[
4\times512
\]

---

# 8. 为什么输出仍然是 4×512？

因为：

Attention 做的不是降维。

而是：

> 上下文混合（Context Mixing）

token 数没变。

embedding 维度没变。

变化的是：

每个 token 的内容已经融合了其它 token 的信息。

---

# 9. 第一行输出的实际意义

例如：

Attention 第一行：

\[
[0.1,0.7,0.1,0.1]
\]

表示：

token1 对：

- token1 给 0.1 权重
- token2 给 0.7 权重
- token3 给 0.1 权重
- token4 给 0.1 权重

于是：

\[
o_1
=
0.1v_1
+
0.7v_2
+
0.1v_3
+
0.1v_4
\]

这里：

\[
v_i\in\mathbb{R}^{512}
\]

因此：

\[
o_1\in\mathbb{R}^{512}
\]

---

# 10. Attention 的本质

Attention 权重不是在：

> “修改 token”

而是在：

> “决定从哪些 token 那里读取多少信息”

所以：

Attention 更像：

> 动态信息聚合

---

# 11. Transformer 中每个 Token 的行为

每个 token：

## Step1：提出需求（Q）

> “我需要什么信息？”

---

## Step2：查看别人（K）

> “谁拥有这种信息？”

---

## Step3：读取内容（V）

> “把那些 token 的内容按比例读取回来”

---

最终得到：

\[
o_i
\]

即：

> 融合上下文后的 token 表示

---

# 12. Attention 前后区别

Attention 前：

\[
x_i
\]

只表示：

> token 自己

---

Attention 后：

\[
o_i
\]

表示：

> “结合整个句子后，对 token i 的重新理解”
