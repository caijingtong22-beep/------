# sklearn 决策边界（DecisionBoundaryDisplay）详解

## 原始代码

```python
'''
使用 inspection.DecisionBoundaryDisplay 显示决策边界
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

feature_1, feature_2 = np.meshgrid(
    np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
    np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
)

grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

tree = DecisionTreeClassifier().fit(
    iris.data[:, :2],
    iris.target
)

y_pred = np.reshape(
    tree.predict(grid),
    feature_1.shape
)

display = DecisionBoundaryDisplay(
    xx0=feature_1,
    xx1=feature_2,
    response=y_pred
)

display.plot()

display.ax_.scatter(
    iris.data[:, 0],
    iris.data[:, 1],
    c=iris.target,
    edgecolor="black"
)

plt.show()
```

---

# 1. 为什么使用 `[:,0]` 和 `[:,1]`

鸢尾花数据：

```python
iris.data.shape
# (150, 4)
```

表示：

- 150 个样本
- 4 个特征

四个特征：

| 索引 | 特征 |
|---|---|
| 0 | 花萼长度 |
| 1 | 花萼宽度 |
| 2 | 花瓣长度 |
| 3 | 花瓣宽度 |

---

## `iris.data[:,0]`

```python
iris.data[:,0]
```

含义：

```text
所有行，第0列
```

即：

```text
所有样本的第0个特征
```

---

## `iris.data[:,1]`

```python
iris.data[:,1]
```

含义：

```text
所有样本的第1个特征
```

---

## 为什么只用前两个特征

因为：

```python
iris.data[:, :2]
```

表示：

```text
取前两列特征
```

即：

- 花萼长度
- 花萼宽度

---

### 原因

决策边界图只能直接画二维：

- x轴 一个特征
- y轴 一个特征

所以：

```text
最多直接可视化两个特征
```

---

# 2. 为什么要生成网格（meshgrid）

目的：

```text
想知道整个二维平面上的每个位置
模型会预测成什么类别
```

---

## 平面是连续的

不能对无限多个点预测。

所以：

```text
把空间离散化
```

即：

```text
生成大量网格点
```

例如：

```text
(1,1) (2,1) (3,1)
(1,2) (2,2) (3,2)
(1,3) (2,3) (3,3)
```

---

# 3. `linspace` 的作用

```python
np.linspace(min, max)
```

不是排序原数据。

而是：

```text
在最小值和最大值之间均匀采样
```

例如：

```python
np.linspace(4.3, 7.9)
```

可能生成：

```python
[4.3, 4.37, 4.44, ..., 7.9]
```

---

# 4. meshgrid 是怎么扩展二维网格的

假设：

```python
x = [1,2,3]
y = [10,20]
```

执行：

```python
X, Y = np.meshgrid(x, y)
```

得到：

```python
X =
[[1 2 3]
 [1 2 3]]

Y =
[[10 10 10]
 [20 20 20]]
```

---

## X 表示什么

```python
X =
[[1 2 3]
 [1 2 3]]
```

表示：

```text
每个网格点的 x 坐标
```

每一行都复制 x。

---

## Y 表示什么

```python
Y =
[[10 10 10]
 [20 20 20]]
```

表示：

```text
每个网格点的 y 坐标
```

每一列复制 y。

---

## 真正的二维点

```python
(X[i,j], Y[i,j])
```

例如：

```python
(X[0,0], Y[0,0]) = (1,10)
(X[0,1], Y[0,1]) = (2,10)
```

---

# 5. 为什么要 `ravel()`

```python
feature_1.ravel()
```

作用：

```text
把二维数组展开成一维
```

例如：

```python
[[1,2],
 [3,4]]
```

展开：

```python
[1,2,3,4]
```

---

# 6. 为什么要 `vstack`

```python
np.vstack([
    feature_1.ravel(),
    feature_2.ravel()
])
```

例如：

```python
[
 [1,2,1,2],
 [10,10,20,20]
]
```

shape：

```python
(2,4)
```

表示：

```text
2个特征
4个样本
```

---

# 7. 为什么还要转置 `.T`

sklearn 要求输入格式：

```text
(n_samples, n_features)
```

即：

```python
[
 [样本1特征1, 样本1特征2],
 [样本2特征1, 样本2特征2]
]
```

---

但 `vstack` 后：

```python
[
 [1,2,1,2],
 [10,10,20,20]
]
```

是：

```text
(特征数, 样本数)
```

所以需要：

```python
.T
```

转成：

```python
[
 [1,10],
 [2,10],
 [1,20],
 [2,20]
]
```

shape：

```python
(4,2)
```

即：

```text
4个样本
每个样本2个特征
```

---

# 8. fit 是什么

```python
tree = DecisionTreeClassifier().fit(
    iris.data[:, :2],
    iris.target
)
```

---

## 第一步：创建模型

```python
DecisionTreeClassifier()
```

只是创建一个空模型。

---

## 第二步：fit

```python
.fit(X, y)
```

这里：

```python
X = iris.data[:, :2]
y = iris.target
```

---

## X 是什么

```python
iris.data[:, :2]
```

shape：

```python
(150,2)
```

表示：

```text
150个样本
每个样本2个特征
```

---

## y 是什么

```python
iris.target
```

例如：

```python
[0,0,0,1,1,2,2]
```

表示：

```text
每个样本属于哪个类别
```

---

## fit 本质

```text
根据 X 和 y 学习分类规则
```

训练完成后：

```python
tree
```

变成：

```text
训练好的决策树模型
```

---

# 9. predict 是什么

```python
tree.predict(grid)
```

这里：

```python
grid
```

是：

```python
[
 [x1,y1],
 [x2,y2],
 ...
]
```

即：

```text
二维平面中的大量网格点
```

---

## predict 的输出

```python
[0,0,0,1,1,2,2]
```

表示：

```text
每个网格点预测属于哪个类别
```

---

# 10. 为什么还要 reshape

```python
np.reshape(
    tree.predict(grid),
    feature_1.shape
)
```

---

## 原因

predict 输出：

```python
[0,0,1,1,2,2...]
```

是一维数组。

但：

画图需要二维网格。

---

## feature_1.shape

例如：

```python
(50,50)
```

表示：

```text
50×50 网格
```

---

## reshape 后

```python
[
 [0,0,0,1],
 [0,0,1,1],
 [2,2,2,1]
]
```

于是：

```python
y_pred[i,j]
```

对应：

```python
(feature_1[i,j], feature_2[i,j])
```

这个坐标点的预测类别。

---

# 11. 整个流程总结

```text
原始数据
   ↓
选两个特征
   ↓
生成二维平面
   ↓
均匀采样大量网格点
   ↓
meshgrid生成坐标网格
   ↓
展开成 sklearn 输入格式
   ↓
训练决策树
   ↓
对整个平面预测
   ↓
恢复成二维网格
   ↓
按类别着色
   ↓
得到决策边界图
```