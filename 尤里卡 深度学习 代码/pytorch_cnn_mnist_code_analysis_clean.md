# PyTorch CNN + Pooling 手写数字识别代码解析

## 1. CNN 网络结构整体理解

这一版代码与之前最大的区别：

- 引入卷积层（Convolution Layer）
- 引入池化层（Pooling Layer）
- 不再直接 flatten 输入 MLP
- 开始利用空间结构提取局部特征

注意：

`__init__()` 只定义网络结构。

真正的数据流发生在：

```python
forward()
```

---

## 2. 第一层卷积：Conv2d(1,4,5,padding=2)

### 输入输出 Shape

MNIST 输入：

```python
[batch_size, 1, 28, 28]
```

第一层卷积：

```python
self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
```

输出：

```python
[batch_size, 4, 28, 28]
```

其中：

- 1：输入通道数
- 4：输出通道数
- 5：卷积核大小
- padding=2：边缘补零

### 为什么输出通道是 4

输出通道数：

```text
= 卷积核数量
= feature map 数量
```

### kernel_size=5 的含义

```python
nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=5,
    padding=2
)
```

卷积核大小：

$$
5 \times 5
$$

每个卷积核参数形状：

$$
[1,5,5]
$$

```python
conv1.weight.shape
# [4,1,5,5]
```

---

## 3. 第二层卷积输入通道

第一层输出：

```python
[batch, 4, 28, 28]
```

池化后：

```python
[batch, 4, 14, 14]
```

Pooling：

- 改变空间尺寸
- 不改变 channel 数

因此：

```python
self.conv2 = nn.Conv2d(4, 8, 5, padding=2)
```

输出：

```python
[batch, 8, 14, 14]
```

---

## 4. Pooling 与 Flatten 尺寸变化

MNIST 原图：

$$
28 \times 28
$$

两次池化：

$$
28 \to 14 \to 7
$$

最终：

```python
[batch, 8, 7, 7]
```

flatten 后：

$$
8 \times 7 \times 7 = 392
$$

```python
self.fc1 = nn.Linear(392, 512)
```

---

## 5. 网络整体 Shape 流

```text
输入:
[batch,1,28,28]

conv1:
[batch,4,28,28]

pool:
[batch,4,14,14]

conv2:
[batch,8,14,14]

pool:
[batch,8,7,7]

flatten:
[batch,392]

fc1:
[batch,512]

fc2:
[batch,10]
```

---

## 6. logits、softmax 与 log_softmax

```python
x = self.fc2(x)

x = F.log_softmax(x, dim=0)

return x
```

### logits

```python
self.fc2 = Linear(512, num_classes)
```

输出：

```python
[-2.1, 0.3, 5.7, ...]
```

这些值叫：

$$
logits
$$

### softmax

$$
softmax(x)=
[0.01,0.03,0.90,...]
$$

### log_softmax

$$
\log(softmax(x_i))
=
x_i - \log\sum_j e^{x_j}
$$

### 更规范的写法

```python
x = self.fc2(x)
return x
```

因为：

```python
nn.CrossEntropyLoss()
```

内部已经自动包含：

```text
log_softmax + NLLLoss
```

---

## 7. DataLoader 与 Batch 解包

```python
for batch_idx, (data, target) in enumerate(train_loader):
```

### train_loader 是什么

```text
批量数据迭代器
```

例如：

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
```

### 每次返回什么

```python
(data, target)
```

即：

```text
(图像 batch, 标签 batch)
```

```python
data.shape
# [64,1,28,28]

target.shape
# [64]
```

### enumerate 做了什么

```python
enumerate(train_loader)
```

返回：

```python
(index, item)
```

例如：

```python
(0, first_batch)
(1, second_batch)
```

---

## 8. 数据流

```text
train_dataset
↓
DataLoader
↓
batch
↓
(data,target)
↓
enumerate
↓
(batch_idx,(data,target))
```
