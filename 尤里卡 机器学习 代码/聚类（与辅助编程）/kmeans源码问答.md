# K-means 数据结构与整体流程整理

---

# 1. 数据结构层级（最重要）

K-means 最容易混乱的地方：

- 列表套列表
- 不同变量之间层级不同

必须分清：

- point
- centroid
- centroids
- cluster
- clusters
- distances

之间的层级关系。

---

# 2. point 是什么？

point：

一个样本点。

例如：

```python
point = [1, 2]
```

表示：

二维空间中的点：

- x = 1
- y = 2

结构：

```text
point
├── point[0] = x
└── point[1] = y
```

三维情况：

```python
point = [1, 2, 3]
```

表示：

- x = 1
- y = 2
- z = 3

---

# 3. dataset 是什么？

dataset：

很多 point 组成的数据集。

例如：

```python
dataset = [
    [1,2],
    [2,3],
    [8,9]
]
```

结构：

```text
dataset
├── point1 = [1,2]
├── point2 = [2,3]
└── point3 = [8,9]
```

所以：

```python
dataset[0]
```

得到：

```python
[1,2]
```

---

# 4. centroid 是什么？

centroid：

聚类中心点。

本质：

也是一个 point。

例如：

```python
centroid = [2,3]
```

表示：

当前簇中心位于：

```text
(2,3)
```

所以：

- point 和 centroid 数据结构完全一样
- 区别只是语义不同

| 名称 | 含义 |
|---|---|
| point | 普通样本 |
| centroid | 聚类中心 |

---

# 5. centroids 是什么？

centroids：

所有中心点。

例如：

```python
centroids = [
    [1,2],
    [8,9]
]
```

结构：

```text
centroids
├── centroid0 = [1,2]
└── centroid1 = [8,9]
```

所以：

```python
centroids[0]
```

得到：

```python
[1,2]
```

即：

第 0 个中心点。

---

# 6. cluster 是什么？

cluster：

一个簇。

即：

很多 point 的集合。

例如：

```python
cluster = [
    [1,2],
    [2,3],
    [1,0]
]
```

结构：

```text
cluster
├── point1 = [1,2]
├── point2 = [2,3]
└── point3 = [1,0]
```

---

# 7. clusters 是什么？

clusters：

所有簇。

例如：

```python
clusters = [
    [[1,2], [2,3]],
    [[8,9], [9,10]]
]
```

结构：

```text
clusters
├── cluster0
│   ├── [1,2]
│   └── [2,3]
│
└── cluster1
    ├── [8,9]
    └── [9,10]
```

所以：

```python
clusters[0]
```

得到：

```python
[[1,2], [2,3]]
```

即：

第 0 个簇。

而：

```python
clusters[0][1]
```

得到：

```python
[2,3]
```

即：

第 0 个簇里的第 1 个点。

---

# 8. distances 是什么？

distances：

当前 point 到所有 centroid 的距离。

例如：

```python
distances = [2.1, 9.5]
```

表示：

| centroid | distance |
|---|---|
| centroid0 | 2.1 |
| centroid1 | 9.5 |

---

# 9. append 是什么？

append：

向列表末尾增加元素。

例如：

```python
a = []

a.append(5)
```

得到：

```python
[5]
```

再：

```python
a.append(10)
```

得到：

```python
[5,10]
```

---

# 10. index 是什么？

例如：

```python
distances = [2.1, 9.5]
```

最小值：

```python
min(distances)
```

得到：

```python
2.1
```

再：

```python
distances.index(2.1)
```

得到：

```python
0
```

即：

最近的是第 0 个中心点。

---

# 11. `[[] for _ in centroids]` 是什么？

这是：

列表推导式。

结构：

```python
[生成内容 for 循环]
```

例如：

```python
centroids = [
    [1,2],
    [8,9]
]
```

执行：

```python
[[] for _ in centroids]
```

第一次循环：

生成：

```python
[]
```

第二次循环：

再生成：

```python
[]
```

最终：

```python
[
    [],
    []
]
```

即：

为每个 centroid 准备一个空 cluster。

注意：

这里生成的是：

```python
[]
```

不是：

```python
_
```

所以不会得到：

```python
[[1,2], [8,9]]
```

---

# 12. assign_clusters 的真实逻辑

核心逻辑：

每个 point：

1. 计算到所有 centroid 的距离
2. 找最近 centroid
3. 放进对应 cluster

例如：

```python
point = [2,3]

centroids = [
    [1,1],
    [10,10]
]
```

得到：

```python
distances = [2.2, 10.6]
```

最近：

```python
nearest_index = 0
```

于是：

```python
clusters[0].append(point)
```

最终：

```python
clusters = [
    [[2,3]],
    []
]
```

即：

point 被分到了第 0 类。

---

# 13. calculate_centroid 的作用

作用：

重新计算簇中心。

例如：

```python
cluster = [
    [1,2],
    [2,4],
    [1,0]
]
```

第 0 维平均：

```text
(1+2+1)/3 = 1.333
```

第 1 维平均：

```text
(2+4+0)/3 = 2
```

得到：

```python
new_centroid = [1.333, 2]
```

即：

簇内所有点的平均位置。

---

# 14. update_centroids 的逻辑

代码：

```python
new_centroids = []

for i in range(len(clusters)):

    cluster = clusters[i]

    new_centroid = calculate_centroid(cluster)

    new_centroids.append(new_centroid)
```

例如：

```python
clusters = [
    [[1,2], [2,4], [1,0]],
    [[8,9], [10,8], [9,10]]
]
```

第一次循环：

```python
i = 0
```

得到：

```python
new_centroid = [1.333,2]
```

append 后：

```python
new_centroids = [
    [1.333,2]
]
```

第二次循环：

```python
i = 1
```

得到：

```python
new_centroid = [9,9]
```

append 后：

```python
new_centroids = [
    [1.333,2],
    [9,9]
]
```

---

# 15. 为什么 append 后 index 不会乱？

因为：

```python
for i in range(len(clusters))
```

按顺序循环：

```text
i=0 → cluster0 → centroid0
i=1 → cluster1 → centroid1
```

所以：

```python
new_centroids[0]
```

一定对应：

```python
clusters[0]
```

---

# 16. `centroids = new_centroids` 是什么时候发生的？

注意：

```python
new_centroids.append(...)
```

并不会替换旧中心点。

真正替换发生在：

```python
centroids = new_centroids
```

例如：

旧：

```python
centroids = [
    [1,2],
    [8,9]
]
```

新：

```python
new_centroids = [
    [1.333,2],
    [9,9]
]
```

执行：

```python
centroids = new_centroids
```

后：

```python
centroids
```

变成：

```python
[
    [1.333,2],
    [9,9]
]
```

---

# 17. break 和 return 的区别

## break

只结束循环。

不会结束函数。

## return

结束整个函数。

---

# 18. K-means 里的 break

代码：

```python
if not centroids_changed(centroids, new_centroids):
    break

centroids = new_centroids
```

意思：

```text
如果中心点没变化：
    结束循环
否则：
    用新的覆盖旧的
```

注意：

```python
break
```

执行后：

```python
centroids = new_centroids
```

不会执行。

但：

```python
return clusters, centroids
```

仍然会执行。

因为：

```python
break
```

只结束 for。

不会结束函数。

---

# 19. 为什么 break 后不需要再替换？

因为：

```python
centroids == new_centroids
```

已经成立。

即：

旧中心点已经是最终结果。

所以：

替不替换都一样。

---

# 20. K-means 完整调用链

```text
k_means
│
├── assign_clusters
│   │
│   └── euclidean_distance
│
├── update_centroids
│   │
│   └── calculate_centroid
│
└── centroids_changed
```

---

# 21. K-means 整体流程

```text
初始化 centroids
        ↓
assign_clusters
        ↓
样本分到不同 clusters
        ↓
update_centroids
        ↓
calculate_centroid
        ↓
得到 new_centroids
        ↓
centroids_changed
        ↓
是否变化？
    ↓          ↓
   是          否
    ↓          ↓
继续迭代      break结束
```

---

# 22. 一个完整例子（强烈建议理解）

## 初始数据

```python
dataset = [
    [1,2],
    [2,3],
    [8,9],
    [9,10]
]
```

初始化：

```python
centroids = [
    [1,2],
    [8,9]
]
```

---

## 第一步：assign_clusters

### point = [1,2]

距离：

```python
distances = [0, 9.9]
```

最近：

```python
nearest_index = 0
```

加入：

```python
clusters[0].append([1,2])
```

---

### point = [2,3]

距离：

```python
distances = [1.4, 8.5]
```

最近：

```python
nearest_index = 0
```

加入：

```python
clusters[0].append([2,3])
```

---

### point = [8,9]

距离：

```python
distances = [9.9, 0]
```

最近：

```python
nearest_index = 1
```

加入：

```python
clusters[1].append([8,9])
```

---

### point = [9,10]

距离：

```python
distances = [11.3, 1.4]
```

最近：

```python
nearest_index = 1
```

加入：

```python
clusters[1].append([9,10])
```

---

最终：

```python
clusters = [
    [[1,2], [2,3]],
    [[8,9], [9,10]]
]
```

---

# 23. update_centroids

第 0 类：

```python
[[1,2], [2,3]]
```

平均：

```python
[(1+2)/2, (2+3)/2]
```

得到：

```python
[1.5, 2.5]
```

---

第 1 类：

```python
[[8,9], [9,10]]
```

平均：

```python
[(8+9)/2, (9+10)/2]
```

得到：

```python
[8.5, 9.5]
```

---

最终：

```python
new_centroids = [
    [1.5,2.5],
    [8.5,9.5]
]
```

---

# 24. 最核心的一句话

K-means 本质：

```text
不断重复：

样本分组
    ↓
重新计算中心
    ↓
再分组
    ↓
再更新中心

直到中心不再变化
```
