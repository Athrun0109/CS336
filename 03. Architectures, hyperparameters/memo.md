### 魔改版Transformer对比原版的区别

- 将归一化模块放到了Attention以及FFN之前
  <img src="1.png" alt="1" style="zoom: 50%;" />



### 归一化函数

- 使用`RMSNorm`代替`LayerNorm`，代替原因是这两个Norm方法效果差不多，但是RMSNorm更加简单，并且在模型训练中，Normalization非常耗时，所以越简单越好。
  **LayerNorm**:
  $$
  y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \cdot \gamma + \beta
  $$
  **RMSNorm**:
  $$
  y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
  $$
  或者
  $$
  y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} \cdot \gamma
  $$

- 去掉bias偏置参数



### 激活函数

- **GLU**
  GLU (Gated Linear Unit) 的核心思想是使用一个门控机制来动态地控制信息流，让网络可以根据输入内容来决定哪些信息应该通过。
  $$
  \text{GLU}(x, W, V, b, c) = (xW + b)\otimes \sigma(xV + c)
  $$
  这个公式的意思是，输入 x 被两个独立的线性变换（xW+b 和 xV+c）分别处理。一个结果作为内容（content），另一个经过 Sigmoid 函数后作为“门”（gate）。门的值在 0 到 1 之间，决定了对应位置的内容有多少可以通过。

- **SwiGLU**
  $$
  \text{SwiGLU}(x, W, V) = (xW) \otimes \text{SiLU}(xV)
  $$
  其中
  $$
  \text{SiLU}(x) = x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
  $$

- **GeGLU**
  $$
  \text{GeGLU}(x, W, V) = (xW) \otimes \text{GELU}(xV)
  $$
  其中
  $$
  \text{GELU}(x)\approx0.5\cdot{x}\cdot(1+\text{tanh}[\sqrt{\frac{2}{\pi}}({x}+0.044715 \cdot{x^3})])
  $$
  GELU(gate)建议直接套用`F.gelu(gate)`实现