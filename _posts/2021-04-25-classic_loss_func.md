---
layout: post
title: 经典损失函数
date: 2021-04-25
tags: algorithm 机器学习
---

## 回归损失函数

1. 均方差损失(MSE)

   $$
   J_{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
   $$

   **背后的假设**

   实际上在一定的假设下，我们可以使用最大化似然得到均方差损失的形式。假设**模型预测与真实值之间的误差服从标准高斯分布** ($\mu=0, \sigma=1$)，则给定一个$x_i$，模型就输出真实值的概率为：

   $$
   p(y_i|x_i) = \frac{1}{\sqrt{2\pi}}exp(-\frac{(y_i - \hat{y}_i)^2}{2})
   $$

   进一步我们假设数据集中所有样本点之间相互独立，则给定所有 $x$ 输出所有真实值 $y$ 的概率，即似然(Likelihood) 为所有 $p(y_i\|x_i)$ 累乘：

   $$
   L(x, y) = \prod_{i=1}^N\frac{1}{\sqrt{2\pi}}exp(-\frac{(y_i - \hat{y}_i)^2}{2})
   $$

   通常为了计算方便，我们通常最大化对数似然（**Log-Likelihood**）：

   $$
   LL(x, y) = log(L(x, y)) = -\frac{N}{2}log(2\pi) - \frac{1}{2}\sum_{i=1}^N(y_i - \hat{y}_i)^2
   $$

   去掉与$y_i$ 无关的第一项，然后转化为最小化负对数似然(**Negative Log-Likelihood**):

   $$
   NLL(x, y) = \frac{1}{2}\sum_{i=1}^N(y_i - \hat{y}_i)^2
   $$

   可以看到这个实际上就是均方差损失的形式。也就是说**在模型输出与真实值的误差服从高斯分布的假设下，最小化均方差损失函数与极大似然估计本质上是一致的**，因此在这个假设能被满足的场景中（比如回归），均方差损失是一个很好的损失函数选择；当这个假设没能被满足的场景中（比如分类），均方差损失不是一个好的选择

2. 平均绝对损失(MAE)

   $$
   J_{MAE} = \frac{1}{N}\sum_{i=1}^N\left|y_i - \hat{y}_i \right|
   $$

   **背后的假设：**

   同样的我们可以在一定的假设下通过最大化似然得到 MAE 损失的形式，假设**模型预测与真实值之间的误差服从拉普拉斯分布 Laplace distribution** ($\mu=0, b=1$) , 则给定一个$x_i$ 模型输出真实值的 $y_i$ 的概率为：

   $$
   p(y_i | x_i) = \frac{1}{2} exp(-\left|y_i - \hat{y}_i \right|)
   $$

   与上面推导 MSE 时类似，我们可以得到的负对数似然（**Negative Log-Likelihood**）实际上就是 MAE 损失的形式：

   $$
   L(x, y) = \prod_{i=1}^N \frac{1}{2}exp(-\left| y_i - \hat{y}_i \right|) \\
   LL(x, y) = -\frac{N}{2} - \sum_{i=1}^N\left|y_i - \hat{y}_i \right| \\
   NLL(x, y) = \sum_{i=1}^N \left| y_i - \hat{y}_i \right|
   $$

   **MAE 与 MSE 的区别：**

   - MSE 比 MAE 能够更快收敛：当使用梯度下降算法时，MSE 损失的梯度为 $-y_i$，而 MAE 损失的梯度为 $\pm 1$ 。所以，MSE 的梯度会随着误差大小发生变化，而 MAE 的梯度一直保持为 1，这不利于模型的训练
   - MAE 对异常点更加鲁棒：从损失函数上看，MSE 对误差平方化，使得异常点的误差过大；从两个损失函数的假设上看，MSE 假设了误差服从高斯分布，MAE 假设了误差服从拉普拉斯分布，拉普拉斯分布本身对于异常点更加鲁棒.

3. Huber Loss (smooth L1 Loss)
   $$
   J_{Huber} = \sum_{i=1}^N \left( I_{\left|y_i - \hat{y}_i \right|\leqslant \delta} \frac{(y_i - \hat{y}_i)^2}{2} + I_{\left|y_i - \hat{y}_i \right|> \delta} \left(\delta \left|y_i - \hat{y}_i\right| - \frac{1}{2}\delta^2 \right)\right)
   $$
   上式中，$\delta$ 是 Huber Loss 的一个超参数，$\delta$ 的值是 MSE 与 MAE 两个损失连接的位置。

## 分类损失函数

1. 0-1 损失

   $$
   L(y_i, f(x_i)) = \left\{\begin{matrix}1, y_i \neq f(x_i) \\ 0, y_i=f(x_i) \end{matrix}\right.
   $$

   特点： 1. 0-1 损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用. 2. **感知机**就是用的这种损失函数。但是相等这个条件太过严格，因此可以放宽条件，即满足 $\left\|  Y-f(X) \right\| < T$ 时认为相等。

   **由于第一个特点，一般分类损失函数是区寻找 0-1 损失函数的一个相对紧上界凸函数。**

2. Hinge Loss

   $$
   L(y_i, f(x_i)) = max(0, 1-y_if(x_i)), \quad  y_i \in \{-1, 1\}
   $$

3. Logistic 损失

   $$
   L(y_i, f(x_i)) = log_2(1 + exp(-f(x_i)y_i)), \quad y_i \in \{-1, 1\}
   $$

   对所有样本点都有惩罚，对于异常值更加敏感。

4. Cross Entropy

   **4.1 对于二分类**，通常使用 Sigmoid 函数将模型的输出压缩到 $(0, 1)$ 区间内，$\hat{y}\in (0,1)$用来代表给定输入 $x_i$，模型判断为正类的概率。由于只有正负两类，因此同时也得到了负类的概率：

   $$
   p(y_i=1 | x_i) = \hat{y}_i, \quad p(y_i=0 | x_i) = 1 - \hat{y}_i; \quad y_i \in \{0, 1\}
   $$

   将两条式子合并成一条：

   $$
   p(y_i|x_i) = (\hat{y}_i)^{y_i}(1-\hat{y}_i)^{1-y_i}
   $$

   假设数据点之间独立同分布，则似然可以表示为：

   $$
   L(x, y) = \prod_{i=1}^N (\hat{y}_i)^{y_i}(1-\hat{y}_i)^{1-y_i}
   $$

   对似然取对数，然后加负号变成最小化负对数似然，即为交叉熵损失函数的形式：

   $$
   NLL(x, y) = \sum_{i=1}^N\left( y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right)
   $$

   **4.2 在多分类的任务中**，交叉熵损失函数的推导思路和二分类是一样的，变化的地方是真实值$y_i$是一个 One-hot 向量，同时模型输出的压缩由原来的 Sigmoid 函数换成 Softmax 函数。Softmax 函数将每个维度的输出范围都限定在 $(0,1)$ 之间，同时所有维度的输出和为 1，用于表示一个概率分布

   $$
   p(y_i|x_i) = \prod_{k=1}^K (\hat{y}_i^k)^{(y_i^k)}
   $$

   其中，$k \in K$ 表示 K 个类别中的一类，同样的假设数据点之间独立同分布，可得到负对数似然为：

   $$
   NLL(x, y) = J_{CE} = \sum_{i=1}^N \sum_{k=1}^K y_i^k \log(\hat{y}_i^k)
   $$

   由于$y_i$是一个 One-hot 向量，除了目标类为 1 之外其他类别上的输出都为 0，因此上式也可以写为：

   $$
   J_{CE} = \sum_{i=1}^N y_i^{c_i} \log(\hat{y}_i^{\hat{c}_i})
   $$

   其中，$c_i$ 是 $x_i$ 的目标类，通常这个应用于多分类的交叉熵损失函数也被称为 Softmax Loss 或者 Categorical Cross Entropy Loss。

   **4.3 为什么用交叉熵损失 (分类中为什么不用均方差损失？)：**

   1. 一个角度是用最大似然来解释：也就是我们上面的推导

   2. 另一个角度是用信息论来解释交叉熵损失(KL 散度)

   3. 最后一个角度为 BP 过程：当使用平方误差损失函数时，最后一层的误差为 $\delta^{(l)} = -\left(y-a^{(l)}\right)f'(z^{(l)})$，其中最后一项为 $f'(z^{(l)})$，为激活函数的导数。当激活函数为 Sigmoid 函数时，如果 $z^{(l)}$ 的值非常大，函数的梯度趋于饱和，即 $f'(z^{(l)})$ 的绝对值非常小，导致 $\delta^{(l)}$ 的取值也非常小，使得基于梯度的学习速度非常缓慢；

      当使用交叉熵损失函数时，最后一层的误差为 $\delta^{(l)} = f\left(z_k^{(l)} \right) - 1 = a_k^{(l)}-1$，此时导数是线性的，因此不存在学习速度过慢的问题
