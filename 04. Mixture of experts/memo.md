### Top-k Mixture of Experts (MoE) 模型数学表达式

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} \left( g_{i,t} \text{ FFN}_i \left( \mathbf{u}_t^l \right) \right) + \mathbf{u}_t^l
$$

- $\mathbf{h}_t^l$：表示模型在第 l 层、时间步 t（或输入序列的第 t 个 token）的输出。

- $\mathbf{u}_t^l$：表示模型在第 l 层、时间步 t 的输入。

- $\text{FFN}_i$：表示第 i 个**专家（Expert）**，它是一个独立的 Feed-Forward Network（前馈网络）。在 MoE 中，通常有多个这样的专家。

- N：表示总共有 N 个专家。

- $g_{i,t}$：表示一个**门控（Gating）值**，它决定了第 i 个专家的输出对最终结果的贡献程度。这个值是动态计算的，取决于输入 utl。

- 求和项 $\sum_{i=1}^{N} \left( g_{i,t} \text{ FFN}_i \left( \mathbf{u}_t^l \right) \right)$：这部分是所有专家输出的加权和。

- $+\mathbf{u}_t^l$：这是一个**残差连接（Residual Connection）**，将原始输入 $\mathbf{u}_t^l$ 加到专家的加权输出上。这有助于训练更深的网络。

**总结**：这个表达式表明，对于每个输入 utl，模型会计算所有专家的输出，然后根据各自的门控值 gi,t 进行加权求和，最后加上残差连接得到最终输出。
$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t}|1 \leq j \leq N\}, K) \\ 0, & \text{otherwise} \end{cases}
$$

- $s_{i,t}$：表示第 i 个专家在时间步 t 的**分数（Score）**。

- $\text{Topk}(\cdot, K)$：这是一个函数，它从所有专家分数的集合 $\{s_{j,t}|1 \leq j \leq N\}$ 中，选取分数最高的 K 个。

- K：一个超参数，表示要激活的专家数量。通常 K 远小于 N（例如 N=128,K=2）。

**总结**：这个表达式定义了**Top-k**选择机制。它首先计算所有 N 个专家的分数 si,t，然后只选择分数排名前 K 的专家。对于这些被选中的专家，gi,t 等于它们的分数 si,t；对于其他分数较低、未被选中的专家，gi,t 直接被设置为 0。

这正是 MoE 模型高效的关键：**每次只激活少数几个专家（K 个）**，而不是全部专家，从而大大减少了计算量，但同时模型参数量可以非常大。
$$
s_{i,t} = \text{Softmax}_i \left( (\mathbf{u}_t^l)^T \mathbf{e}_i^l \right)
$$

- $\mathbf{u}_t^l$：再次是第 l 层、时间步 t 的输入。
- $\mathbf{e}_i^l$：这是一个**可训练的专家嵌入（Expert Embedding）**向量，用于第 i 个专家。它决定了这个专家在“门控网络”（gating network）中的权重。
- $(\mathbf{u}_t^l)^T \mathbf{e}_i^l$：这是一个点积（或矩阵乘法），用于计算输入 utl 与每个专家嵌入 eil 之间的相似度或相关性。
- $\text{Softmax}_i(⋅)$：这是一个 Softmax 函数，它将所有专家的点积结果归一化为概率分布。si,t 是 Softmax 结果中的第 i 个元素。

**总结**：这个表达式展示了如何通过一个简单的门控网络来计算每个专家的分数。它将当前输入 utl 与每个专家的独有嵌入 eil 进行点积，然后通过 Softmax 归一化，得到每个专家对当前输入的重要程度分数 si,t。这些分数随后被用于 Top-k 门控机制。
