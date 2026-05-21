# CNN 卷积、Pooling 与 PyTorch `eval()` / `no_grad()` 解析

## 1. 为什么卷积后尺寸还是 `28×28`

输入：

```python
[batch, 1, 28, 28]
```

卷积层：

```python
self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
```

输出：

```python
[batch, 4, 28, 28]
```

很多人第一次学 CNN 时会疑惑：

```text
卷积核不是会“扫描图片”吗？
为什么尺寸没变？
```

这里需要区分两个概念：

- spatial size（空间尺寸）
- channel depth（通道深度）

即：

```text
H × W
```

与：

```text
feature map 数量
```

不是同一个东西。

---

## 2. Conv2d 的参数含义

```python
nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=5,
    padding=2,
    stride=1
)
```

含义：

| 参数 | 含义 |
|---|---|
| in_channels | 输入通道数 |
| out_channels | 输出通道数 |
| kernel_size | 卷积核大小 |
| padding | 边缘补零 |
| stride | 步长 |

---

## 3. 为什么 channel 从 1 变成 4

输入：

```python
[batch, 1, 28, 28]
```

输出：

```python
[batch, 4, 28, 28]
```

这里：

```text
1 → 4
```

表示：

```text
feature map 数量增加
```

即：

- 第一层有 4 个卷积核
- 每个卷积核都会扫描整张图
- 每个卷积核都会生成一张 feature map

因此：

```text
1张输入图
↓
4张特征图
```

所以：

```text
channel:
1 → 4
```

---

## 4. 为什么空间尺寸还是 `28×28`

关键在于：

```python
padding=2
kernel_size=5
stride=1
```

卷积输出尺寸公式：

$$
Output
=
\left\lfloor
\frac{W-K+2P}{S}
\right\rfloor
+1
$$

其中：

| 符号 | 含义 |
|---|---|
| $W$ | 输入尺寸 |
| $K$ | kernel size |
| $P$ | padding |
| $S$ | stride |

---

### 代入数值

$$
W=28
$$

$$
K=5
$$

$$
P=2
$$

$$
S=1
$$

得到：

$$
\left\lfloor
\frac{28-5+2\times2}{1}
\right\rfloor
+1
=
28
$$

因此：

$$
28\times28
\rightarrow
28\times28
$$

---

## 5. 为什么 padding 可以保持尺寸

如果：

```python
padding=0
```

则：

$$
28-5+1=24
$$

即：

$$
28\times28
\rightarrow
24\times24
$$

因为：

```text
卷积核会“吃掉边缘”
```

5×5 卷积核无法完整覆盖最边缘区域。

---

### padding 的本质

```python
padding=2
```

等价于：

```text
四周补两圈0
```

即：

$$
28\times28
\rightarrow
32\times32
$$

然后再卷积：

$$
32-5+1=28
$$

因此尺寸保持不变。

---

## 6. Pooling 为什么会变成 `14×14`

Pooling：

```python
self.pool = nn.MaxPool2d(2,2)
```

含义：

| 参数 | 含义 |
|---|---|
| kernel size | 2×2 |
| stride | 2 |

即：

- 每次取一个 `2×2` 区域
- 取最大值
- 每次移动 2 格

---

### Pooling 前

```python
[batch, 4, 28, 28]
```

### Pooling 后

```python
[batch, 4, 14, 14]
```

因为：

$$
28/2=14
$$

---

## 7. 为什么 channel 不变

Pooling：

```text
只对每个 feature map 内部做降采样
```

不会改变：

```text
feature map 数量
```

因此：

```text
4张 feature map 仍然存在
```

只是：

$$
28\times28
\rightarrow
14\times14
$$

---

## 8. Conv 与 Pool 的职责分工

### Conv

负责：

- 提取特征
- 增加语义
- 增加 channel 数

即：

```text
feature extraction
```

---

### Pool

负责：

- 压缩空间尺寸
- 减少计算量
- 增强平移鲁棒性

即：

```text
downsampling
```

---

## 9. 经典 CNN 的变化趋势

经典 CNN 通常：

```text
Conv
↓
Pool
↓
Conv
↓
Pool
```

整体趋势：

| 属性 | 变化 |
|---|---|
| spatial size | 越来越小 |
| semantic strength | 越来越强 |
| channel 数 | 越来越多 |

---

# PyTorch：`eval()` 与 `torch.no_grad()` 区别

很多初学者会混淆：

```python
net.eval()
```

与：

```python
torch.no_grad()
```

它们完全不是一个东西。

---

## 10. `torch.no_grad()` 的作用

```python
with torch.no_grad():
```

作用：

```text
关闭 autograd
```

即：

PyTorch 不再：

- 构建计算图
- 保存梯度
- 记录 backward 信息

---

### 为什么验证阶段通常使用它

验证阶段通常只需要：

```text
forward
→
算准确率
```

而不需要：

```python
loss.backward()
```

因此：

```python
with torch.no_grad():
```

可以：

- 减少显存
- 提升速度

这是标准写法。

---

## 11. `eval()` 的作用

```python
net.eval()
```

不会关闭梯度。

它只会：

```text
切换网络层行为
```

---

### Dropout

训练时：

```text
随机屏蔽神经元
```

测试时：

```text
必须关闭随机屏蔽
```

否则：

每次预测都会不同。

---

### BatchNorm

训练时：

```text
使用当前 batch 的统计量
```

测试时：

```text
使用训练阶段累计均值方差
```

---

## 12. 两者区别总结

| 操作 | 作用 |
|---|---|
| `net.eval()` | 切换网络行为 |
| `torch.no_grad()` | 关闭梯度计算 |

---

## 13. 标准验证写法

```python
net.eval()

with torch.no_grad():

    for data, target in validation_loader:

        output = net(data)

        right = rightness(output, target)

        val_rights.append(right)
```

---

## 14. 为什么 `requires_grad_(True)` 很奇怪

代码：

```python
data.clone().requires_grad_(True)
```

会：

```text
主动开启输入梯度
```

但验证阶段：

```python
没有 loss.backward()
```

因此：

这些梯度没有意义。

只会：

- 浪费显存
- 增加计算图开销

---

## 15. 为什么 `target.detach()` 不需要

因为：

label 默认：

```python
requires_grad=False
```

所以：

```python
target.detach()
```

通常是冗余的。

---

## 16. 为什么不推荐 `net.forward()`

不推荐：

```python
outputs = net.forward(x)
```

推荐：

```python
outputs = net(x)
```

因为：

```python
net(x)
```

除了调用 `forward()`：

还会：

- 注册 hooks
- 处理 module 状态
- 执行框架级逻辑

直接：

```python
net.forward()
```

会绕过这些机制。

