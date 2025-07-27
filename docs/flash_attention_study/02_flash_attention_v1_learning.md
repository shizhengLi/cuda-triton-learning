# 通俗讲解 FlashAttention v1 的分块策略与 Online Softmax 应用

FlashAttention 是一种高效的注意力机制算法，特别设计来减少 GPU 高带宽内存（HBM）的访问，最大化利用快速的静态随机存取存储器（SRAM）。它通过**分块计算**和 **Online Softmax** 结合，解决了传统注意力机制在处理大矩阵时的内存瓶颈问题。下面我们来通俗地讲解代码中的分块策略，以及如何结合 Online Softmax 实现高效计算。

---

## 分块策略：把大问题拆成小块

在 Transformer 的注意力机制中，我们需要计算注意力分数矩阵 $S = QK^T$，然后对其应用 Softmax，最后乘以 $V$ 得到输出 $O$。但当序列长度 $N$ 很大时（比如处理长文本或高分辨率图像），$Q$、$K$、$V$ 矩阵（大小为 $N \times d$）和分数矩阵 $S$（大小为 $N \times N$）会占用大量 HBM。如果直接把这些矩阵加载到 HBM，内存可能不够，计算也会很慢。

FlashAttention 的核心思想是：**把大矩阵切成小块，只把当前需要的块加载到 SRAM 进行计算**。SRAM 比 HBM 快得多，但容量有限（通常只有几 MB），所以我们需要精心设计分块方式，确保每个块的计算都能在 SRAM 中完成。

### 分块方式

- **Q 矩阵的分块**：
  - $Q$ 矩阵的形状是 $[N, d]$，表示 $N$ 个查询向量，每个向量维度是 $d$。
  - 我们按**行**分块，将 $Q$ 切成多个小块，每块大小为 $[B_r, d]$。
  - 比如，$N=1024$，$B_r=256$，那么 $Q$ 会被分成 $1024 / 256 = 4$ 块，每块有 256 行，维度 $d$ 不变。
  - 每次只加载一块 $Q_i$（大小 $[B_r, d]$）到 SRAM。

- **K 和 V 矩阵的分块**：
  - $K$ 和 $V$ 矩阵也是 $[N, d]$ 的形状，同样按**行**分块，每块大小为 $[B_c, d]$。
  - 比如，$N=1024$，$B_c=128$，那么 $K$ 和 $V$ 各被分成 $1024 / 128 = 8$ 块，每块有 128 行。
  - 每次只加载一块 $K_j$ 和 $V_j$（大小 $[B_c, d]$）到 SRAM。

- **注意力分数矩阵 $S$ 的分块**：
  - 计算 $S_{ij} = Q_i K_j^T$，得到一个 $[B_r, B_c]$ 的小矩阵。
  - 这个小矩阵完全在 SRAM 中计算，不需要存到 HBM。
  - 比如，$B_r=256$，$B_c=128$，那么 $S_{ij}$ 是 $[256, 128]$ 的矩阵。

### 为什么这样分块？

- **减少 HBM 访问**：每次只加载 $Q_i$、$K_j$、$V_j$ 这三个小块到 SRAM，计算完一块的注意力后直接更新输出 $O$，无需存储整个 $S$ 矩阵。
- **适合 SRAM 容量**：SRAM 容量有限（比如几 MB），分块后每个小块的内存占用可控。
- **流水线计算**：通过内外循环（外循环遍历 $K$ 和 $V$ 的块，内循环遍历 $Q$ 的块），我们可以逐块计算注意力，逐步累积输出。



---




## Online Softmax 在 FlashAttention 中的应用

FlashAttention 利用 **Online Softmax** 来高效计算注意力分数的归一化，避免存储大规模的注意力分数矩阵 $S = QK^T$，从而最大化 SRAM 利用率，减少 HBM 访问。代码中的 $m[i:i+Br]$ 和 $l[i:i+Br]$ 对应于 $Q$ 的第 $i$ 到 $i+B_r$ 行的最大值和 Softmax 分母，体现了分块处理的核心思想。下面我们结合代码，详细讲解 Online Softmax 如何应用于 $QK^T$ 的归一化，以及最终输出 $O$ 的计算。

### 代码中的分块与 Online Softmax

以下是 FlashAttention 的核心代码片段，展示了分块和 Online Softmax 的实现：



```python
import torch

def flash_attention_v1(Q, K, V, Br, Bc):
    """
    Flash Attention v1 算法实现
    参数:
        Q, K, V: 输入矩阵 [N, d]
        Br, Bc: 分块大小
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)  # 输出矩阵
    l = torch.zeros(N, 1)    # 分母
    m = torch.ones(N, 1) * (-float('inf'))  # 最大值
    
    for j in range(0, N, Bc):
        Kj = K[j:j+Bc, :]  # 加载 K 块
        Vj = V[j:j+Bc, :]  # 加载 V 块
        for i in range(0, N, Br):
            Qi = Q[i:i+Br, :]  # 加载 Q 块
            Sij = Qi @ Kj.T     # 计算 [Br, Bc]
            
            mij = torch.max(Sij, dim=1, keepdim=True).values
            mi_new = torch.maximum(m[i:i+Br], mij)
            
            scale_old = torch.exp(m[i:i+Br] - mi_new)
            scale_new = torch.exp(mij - mi_new)
            
            li_new = l[i:i+Br] * scale_old + torch.sum(
                torch.exp(Sij - mi_new), dim=1, keepdim=True
            )
            
            Pij = torch.exp(Sij - mi_new)
            O[i:i+Br] = O[i:i+Br] * scale_old + Pij @ Vj
            
            l[i:i+Br] = li_new
            m[i:i+Br] = mi_new
    
    O = O / l
    return O
```

### 分块体现

- **$m[i:i+Br]$ 和 $l[i:i+Br]$**：
  - $m[i:i+Br]$ 表示 $Q$ 的第 $i$ 到 $i+B_r$ 行在当前处理阶段的注意力分数最大值。
  - $l[i:i+Br]$ 表示对应行的 Softmax 分母（归一化因子）。
  - 这些统计量按 $Q$ 的行分块（每块 $B_r$ 行）维护，体现了分块计算的核心。

- **$S_{ij}$**：
  - $S_{ij} = Q_i K_j^T$ 是当前块的注意力分数矩阵，形状为 $[B_r, B_c]$。
  - $Q_i$ 是 $Q$ 的第 $i$ 到 $i+B_r$ 行（$[B_r, d]$），$K_j$ 是 $K$ 的第 $j$ 到 $j+B_c$ 行（$[B_c, d]$）。
  - $S_{ij}$ 在 SRAM 中计算，避免存储整个 $[N, N]$ 的 $S$ 矩阵。

- **$O[i:i+Br]$**：
  - 表示输出矩阵 $O$ 的第 $i$ 到 $i+B_r$ 行，形状为 $[B_r, d]$。
  - $O$ 是最终的注意力输出，即 $\text{softmax}(QK^T) \cdot V$，通过分块增量更新。

### Online Softmax 原理

Online Softmax 的核心是对注意力分数矩阵 $S = QK^T$ 进行归一化，得到概率矩阵 $P = \text{softmax}(S)$，但不直接存储 $P$，而是增量式地计算并应用于 $V$。以下是具体步骤：

1. **计算当前块的最大值**：
   - 对于每个小块 $S_{ij}$（$[B_r, B_c]$），计算行最大值 $m_{ij}$：
     ```python
     mij = torch.max(Sij, dim=1, keepdim=True).values
     ```
   - $m_{ij}$ 是当前块每行的最大注意力分数。

2. **更新全局最大值**：
   - 用当前块的最大值 $m_{ij}$ 更新全局最大值 $m[i:i+Br]$：
     ```python
     mi_new = torch.maximum(m[i:i+Br], mij)
     ```
   - $m[i:i+Br]$ 维护 $Q$ 的第 $i$ 到 $i+B_r$ 行在所有已处理 $K$ 块中的最大分数。

3. **缩放因子**：
   - 计算缩放因子以调整旧统计量：
     ```python
     scale_old = torch.exp(m[i:i+Br] - mi_new)
     scale_new = torch.exp(mij - mi_new)
     ```
   - $e^{m_i - m_{i_{\text{new}}}}$ 缩放旧的统计量，适应新的最大值。
   - $e^{m_{ij} - m_{i_{\text{new}}}}$ 是当前块的指数项。

4. **更新分母**：
   - Softmax 的分母 $l$ 是每行的归一化因子，更新公式为：
     $$
     l_{i_{\text{new}}} = l_i \cdot e^{m_i - m_{i_{\text{new}}}} + \sum \exp(S_{ij} - m_{i_{\text{new}}})
     $$
     代码实现：
     ```python
     li_new = l[i:i+Br] * scale_old + torch.sum(
         torch.exp(Sij - mi_new), dim=1, keepdim=True
     )
     ```
   - 第一项 $l_i \cdot e^{m_i - m_{i_{\text{new}}}}$ 缩放之前的分母。
   - 第二项 $\sum \exp(S_{ij} - m_{i_{\text{new}}})$ 是当前块的指数和。

5. **计算未归一化的 Softmax（$P_{ij}$）**：
   - $P_{ij} = \exp(S_{ij} - m_{i_{\text{new}}})$ 是 Softmax 的分子部分：
     ```python
     Pij = torch.exp(Sij - mi_new)
     ```
   - $P_{ij}$（$[B_r, B_c]$）是未归一化的注意力权重，对应于 $\text{softmax}(S_{ij})$ 的分子。

6. **更新输出 $O$**：
   - $O$ 是最终的注意力输出 $\text{softmax}(QK^T) \cdot V$，增量更新公式为：
     $$
     O_i = O_i \cdot e^{m_i - m_{i_{\text{new}}}} + P_{ij} V_j
     $$
     代码实现：
     ```python
     O[i:i+Br] = O[i:i+Br] * scale_old + Pij @ Vj
     ```
   - $O_i \cdot e^{m_i - m_{i_{\text{new}}}}$ 缩放之前的输出。
   - $P_{ij} V_j$ 是当前块的贡献，$P_{ij}$（$[B_r, B_c]$）与 $V_j$（$[B_c, d]$）相乘，得到 $[B_r, d]$ 的输出增量。

7. **最终归一化**：
   - 在所有块处理完成后，用分母 $l$ 归一化 $O$：
     ```python
     O = O / l
     ```
   - 这确保 $O$ 是正确的注意力输出，等价于 $\text{softmax}(QK^T) \cdot V$。

### 为什么高效

- **SRAM 利用率高**：
  - 每次只加载 $Q_i$（$[B_r, d]$）、$K_j$（$[B_c, d]$）、$V_j$（$[B_c, d]$）和 $S_{ij}$（$[B_r, B_c]$），适合 SRAM 容量。
  - 约束公式 $4 B_r d + 8 B_c d \leq M$ 确保数据在 SRAM 中处理。

- **避免存储 $S$ 或 $P$**：
  - Online Softmax 只维护 $m$（$[N, 1]$）和 $l$（$[N, 1]$），无需存储 $[N, N]$ 的 $S$ 或 $P$。

- **流式计算**：
  - 增量更新 $m$、$l$ 和 $O$，适合大规模序列和流式数据。

### 小结

Online Softmax 在 FlashAttention 中通过分块处理 $S = QK^T$，增量计算每行的最大值 $m_i$ 和分母 $l_i$，生成未归一化的 Softmax 权重 $P_{ij}$。这些权重直接与 $V_j$ 相乘，更新输出 $O$，最后通过 $l$ 归一化。针对 $QK^T$ 的部分，Online Softmax 确保高效计算 $\text{softmax}(QK^T)$，而 $O$ 是最终的 $\text{softmax}(QK^T) \cdot V$。这种方法极大减少了内存占用，适合 GPU 上的高效计算。

</xaiArtifact>



## 复杂度分析

### 传统注意力

- **时间复杂度**：$O(N^2 d)$，因为计算 $QK^T$ 需要 $N^2 d$ 次操作，Softmax 和 $SV$ 也需要 $O(N^2)$。
- **空间复杂度**：$O(N^2)$，需要存储整个 $S$ 矩阵。

### FlashAttention

- **时间复杂度**：
  - 每次小块计算 $S_{ij} = Q_i K_j^T$：$O(B_r B_c d)$。
  - 总共 $(N / B_r) \cdot (N / B_c)$ 次小块计算，总时间仍是 $O(N^2 d)$。
  - 但由于 SRAM 的高效访问，实际运行时间大幅减少。
- **空间复杂度**：
  - SRAM 中存储 $Q_i$、$K_j$、$V_j$、$S_{ij}$，总共 $O(B_r d + B_c d + B_r B_c)$。
  - 全局只需要 $O(N d)$ 存储 $O$，$O(N)$ 存储 $m$ 和 $l$。
  - 相比传统注意力的 $O(N^2)$，内存占用大大降低。

---

## 总结

FlashAttention 通过**分块策略**和 **Online Softmax**，将注意力机制的计算拆分成小块，最大化利用 SRAM，减少 HBM 访问。代码中的 $m[i:i+Br]$ 和 $O[i:i+Br]$ 反映了 $Q$ 矩阵的分块处理，而 Online Softmax 确保了每块的 Softmax 计算高效且数值稳定。分块大小 $B_r$ 和 $B_c$ 由 SRAM 容量约束，通过公式 $xB_r d + y B_c d \leq M$ 确定。

希望这篇讲解让你对 FlashAttention 的分块和 Online Softmax 的应用有更清晰的理解！如果有更多问题，欢迎继续讨论～





