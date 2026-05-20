# KMeans 聚类代码理解笔记：类、对象、labels_、enumerate 与 clustered_sentences

## 1. `KMeans` 为什么是大写？

在这句代码中：

```python
clustering_model = KMeans(n_clusters=num_clusters)
```

`KMeans` 不是变量名，而是一个类（class）名。

也就是说，这里实际上发生的是：

```text
调用 KMeans 类
创建一个 KMeans 对象
```

即：

```python
KMeans(...)
```

类似于：

```python
list(...)
dict(...)
LinearRegression(...)
DecisionTreeClassifier(...)
```

这些都是类名。

Python 约定中：

```text
类名：
通常使用 PascalCase / CamelCase
首字母大写
```

所以在 `sklearn` 里面，常见的类名包括：

```python
KMeans
PCA
RandomForestClassifier
LogisticRegression
```

而变量名、函数名通常小写，例如：

```python
clustering_model
num_clusters
fit_transform
```

因此：

```python
clustering_model = KMeans(n_clusters=5)
```

可以拆成：

```text
左边：
clustering_model 是变量名

右边：
KMeans(n_clusters=5) 是创建一个 KMeans 对象
```

也就是：

```text
clustering_model
=
一个具体的 KMeans 实例
```

所以后面：

```python
clustering_model.fit(...)
```

本质是：

```text
调用这个 KMeans 对象的方法
```

可以用一个简单例子类比：

```python
class Dog:
    pass

d = Dog()
```

这里：

```python
Dog
```

是类。

而：

```python
d
```

是对象。

同理：

```python
KMeans
```

是 `sklearn` 定义好的聚类类。

而：

```python
clustering_model
```

是你创建出来的一个具体的 KMeans 聚类器对象。

这个对象里面后续会保存：

```text
cluster centers
labels
参数
训练结果
```

等状态。

---

## 2. `clustered_sentences` 这几行代码在做什么？

代码如下：

```python
clustered_sentences = [[] for i in range(num_clusters)]

for sentence_id, cluster_id in enumerate(clustering_model.labels_):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
```

你已经理解了第一句：

```python
clustered_sentences = [[] for i in range(num_clusters)]
```

如果：

```python
num_clusters = 5
```

那么初始状态就是：

```python
clustered_sentences =
[
    [],
    [],
    [],
    [],
    []
]
```

也就是：

```text
创建 5 个空列表
每个空列表对应一个 cluster
```

它的结构是：

```text
clustered_sentences
    ↓
5 个 cluster list
        ↓
        每个 list 之后存放属于该 cluster 的句子
```

---

## 3. `clustering_model.labels_` 是什么？

执行完：

```python
clustering_model.fit(corpus_embeddings)
```

之后，KMeans 会给每一个样本分配一个 cluster 编号。

这些编号保存在：

```python
clustering_model.labels_
```

里面。

它大概长这样：

```python
[0, 1, 1, 3, 0, 4, 2, ...]
```

意思是：

```text
第 0 个句子 → cluster 0
第 1 个句子 → cluster 1
第 2 个句子 → cluster 1
第 3 个句子 → cluster 3
第 4 个句子 → cluster 0
...
```

也就是：

```python
labels_[i]
```

表示：

```text
第 i 个句子属于哪个 cluster
```

注意：如果 `num_clusters = 5`，cluster 编号通常是：

```text
0, 1, 2, 3, 4
```

不是：

```text
1, 2, 3, 4, 5
```

因为 Python 的索引从 0 开始。

---

## 4. 为什么 `for` 循环里有两个变量？

代码：

```python
for sentence_id, cluster_id in enumerate(clustering_model.labels_):
```

关键是理解：

```python
enumerate(clustering_model.labels_)
```

`enumerate(...)` 会把一个可迭代对象变成：

```text
(index, value)
```

形式。

假设：

```python
clustering_model.labels_ = [0, 1, 1, 3, 0]
```

那么：

```python
enumerate(clustering_model.labels_)
```

实际产生的是：

```python
(0, 0)
(1, 1)
(2, 1)
(3, 3)
(4, 0)
```

也就是：

```text
(句子编号, 该句子的 cluster 编号)
```

所以：

```python
for sentence_id, cluster_id in enumerate(clustering_model.labels_):
```

中：

```python
sentence_id
```

表示：

```text
句子的编号
```

而：

```python
cluster_id
```

表示：

```text
该句子属于哪个 cluster
```

例如某一次循环：

```python
sentence_id = 2
cluster_id = 1
```

意思是：

```text
第 2 个句子属于 cluster 1
```

---

## 5. `append` 这一句具体做了什么？

代码：

```python
clustered_sentences[cluster_id].append(corpus[sentence_id])
```

假设当前：

```python
sentence_id = 2
cluster_id = 1
```

那么这句代码等价于：

```python
clustered_sentences[1].append(corpus[2])
```

意思是：

```text
把 corpus 里第 2 个句子
放进 clustered_sentences 的第 1 个 cluster list 里面
```

其中：

```python
corpus[sentence_id]
```

取出原始句子。

而：

```python
clustered_sentences[cluster_id]
```

找到该 cluster 对应的列表。

最后：

```python
.append(...)
```

把这个句子加入该 cluster。

---

## 6. 一个完整的小例子

假设：

```python
labels_ = [0, 1, 1, 0]
```

并且：

```python
corpus = [
    "A",
    "B",
    "C",
    "D"
]
```

初始状态：

```python
clustered_sentences =
[
    [],
    [],
    [],
    [],
    []
]
```

### 第一次循环

```python
sentence_id = 0
cluster_id = 0
```

执行：

```python
clustered_sentences[0].append("A")
```

结果：

```python
[
    ["A"],
    [],
    [],
    [],
    []
]
```

### 第二次循环

```python
sentence_id = 1
cluster_id = 1
```

执行：

```python
clustered_sentences[1].append("B")
```

结果：

```python
[
    ["A"],
    ["B"],
    [],
    [],
    []
]
```

### 第三次循环

```python
sentence_id = 2
cluster_id = 1
```

执行：

```python
clustered_sentences[1].append("C")
```

结果：

```python
[
    ["A"],
    ["B", "C"],
    [],
    [],
    []
]
```

### 第四次循环

```python
sentence_id = 3
cluster_id = 0
```

执行：

```python
clustered_sentences[0].append("D")
```

最终结果：

```python
[
    ["A", "D"],
    ["B", "C"],
    [],
    [],
    []
]
```

所以最后：

```python
clustered_sentences
```

本质上是：

```text
按 cluster 分组后的文本
```

结构是：

```python
[
    cluster0里面的句子列表,
    cluster1里面的句子列表,
    cluster2里面的句子列表,
    ...
]
```

也就是一个二维 list。

层级关系：

```text
clustered_sentences
    ↓
cluster list
    ↓
sentence string
```

---

## 7. 最后的打印代码在做什么？

代码：

```python
for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i)
    print(cluster)
```

这里再次使用了：

```python
enumerate(clustered_sentences)
```

假设前面最终：

```python
clustered_sentences =
[
    ["句子A", "句子D"],
    ["句子B", "句子C"],
    ["句子E"]
]
```

那么：

```python
for i, cluster in enumerate(clustered_sentences):
```

实际遍历的是：

```python
(0, ["句子A", "句子D"])
(1, ["句子B", "句子C"])
(2, ["句子E"])
```

所以：

```python
i
```

是：

```text
cluster 编号
```

而：

```python
cluster
```

是：

```text
这个 cluster 里面的句子 list
```

于是：

```python
print("Cluster ", i)
```

输出：

```text
Cluster 0
```

然后：

```python
print(cluster)
```

输出：

```python
["句子A", "句子D"]
```

最终打印效果类似：

```text
Cluster 0
['句子A', '句子D']

Cluster 1
['句子B', '句子C']

Cluster 2
['句子E']
```

---

## 8. 一个容易混淆的点：`clustered_sentences` 本身有没有变成 `(0, 句子列表)`？

没有。

需要区分：

```python
clustered_sentences
```

和：

```python
enumerate(clustered_sentences)
```

`clustered_sentences` 本身仍然只是：

```python
[
    list,
    list,
    list
]
```

它没有真的变成：

```python
(0, 句子列表)
(1, 句子列表)
```

真正临时产生：

```python
(0, cluster0_list)
(1, cluster1_list)
```

的是：

```python
enumerate(clustered_sentences)
```

这个迭代器。

也就是说：

```text
clustered_sentences 本身：
    只是二维 list

enumerate(clustered_sentences)：
    临时生成 (index, value)
```

---

## 9. 总结：这几段代码的完整逻辑

完整逻辑可以压缩成：

```text
1. KMeans(n_clusters=5)
   创建一个 KMeans 聚类器对象

2. clustering_model.fit(corpus_embeddings)
   用 embedding 训练/执行 KMeans 聚类

3. clustering_model.labels_
   得到每个句子的 cluster 编号

4. clustered_sentences = [[] for i in range(num_clusters)]
   创建 5 个空列表，用于存每个 cluster 的句子

5. for sentence_id, cluster_id in enumerate(clustering_model.labels_):
       clustered_sentences[cluster_id].append(corpus[sentence_id])
   按 cluster_id 把原始句子放入对应列表

6. for i, cluster in enumerate(clustered_sentences):
       print("Cluster ", i)
       print(cluster)
   逐个打印 cluster 编号和该 cluster 里的句子
```

最终：

```python
clustered_sentences
```

是一个二维列表：

```python
[
    ["属于 cluster 0 的句子1", "属于 cluster 0 的句子2"],
    ["属于 cluster 1 的句子1", "属于 cluster 1 的句子2"],
    ["属于 cluster 2 的句子1"],
    ...
]
```
