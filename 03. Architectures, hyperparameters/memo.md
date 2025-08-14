### 魔改版Transformer对比原版的区别

- 将归一化模块放到了Attention以及FFN之前
  <img src="1.png" alt="1" style="zoom: 50%;" />

- 使用`RMSNorm`代替`LayerNorm`，代替原因是这两个Norm方法效果差不多，但是RMSNorm更加简单，并且在模型训练中，Normalization非常耗时，所以越简单越好。
  LayerNorm:
  $$
  y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \cdot \gamma + \beta
  $$
  RMSNorm:
  $$
  y = \frac{x}{\sqrt{||x||_2^2 + \epsilon}} * \gamma
  $$

- 去掉bias偏置参数
  