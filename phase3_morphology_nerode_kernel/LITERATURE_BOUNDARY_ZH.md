# MORPH-N 最近邻理论边界

## 结论

形态成本自动机就是 min-plus/tropical 加权自动机。规范化代价向量就是加权子集构造中的 configuration/residual。若其可达集合有限，用 BFS 构造确定控制器也是标准加权自动机确定化思路的直接实例。**这些部分不构成 MORPH-N 原创性。**

MORPH-N 仍可能保留的研究空间只在：程序语义隐式生成指数级架构；残差控制状态与可执行程序架构分离；真实中间结果复用定义的迁移几何；无需展开全部程序形态的精确定价；每个执行形态和迁移都附带可检查证书。

## 逐项边界表

| 方向 | 已有理论对象与算法 | TCMK/MORPH-N 的数学对应 | 完全重合 | 可能新增 | 优先权风险 |
|---|---|---|---|---|---|
| 一般 min-plus WFA 确定化 | Almagor、Arbel、Sheinvald 证明一般 min-plus WFA 的 determinisability 可判定，解决长期开放问题；核心使用 gap、baseline-augmented construction、cactus letters 与 zooming。[arXiv:2503.23826](https://arxiv.org/abs/2503.23826) | 全架构是 WFA 状态；任务是字母；动态最优是最小运行权；规范化形态代价向量是 configuration | 形态系统到 WFA 的代数表达、规范化残差、有限残差 BFS 确定化 | 从程序结构隐式产生 WFA 和迁移证书；只保留实际架构子集的 K | **极高**：若只宣传残差确定化，就是重新命名 WFA |
| 确定化复杂度 | 后续工作把纯存在性可判定推进为构造性的 gap 上界，并给出 primitive-recursive 上界；论文明确配置向量、gap 见证和有限 cactus 字母。[arXiv:2602.01221](https://arxiv.org/abs/2602.01221) | “残差是否有界/闭包是否有限”正落入 determinisability/gap 分析 | 对任意显式 WFA 判断是否存在有限确定表示 | 对程序族证明远强于一般上界的结构界、参数化定价 | **极高**：不能把一个特殊充分条件说成首次解决一般终止性 |
| 消歧义与寄存器最小化 | WFA unambiguisability 可判定并归约到 determinisability；tropical CRA counter minimisation 即使固定 7 个寄存器也不可判定。[arXiv:2512.09484](https://arxiv.org/abs/2512.09484) | R 的行为商类似有限确定控制器最小化；若把残差坐标当寄存器，则接近 CRA 表示问题 | 有限显式控制器的行为状态合并 | R 与实际可执行架构 K 的双对象约束 | **高**：不能声称解决一般寄存器最小化；本项目只最小化已完成的有限 Mealy 控制器 |
| 单字母 tropical WFA | 单字母 WFA 可多项式变换为线性多个、每个二次大小的确定 WFA 联；determinisation、register minimisation、boundedness 均 coNP-complete。[arXiv:2606.26038](https://arxiv.org/abs/2606.26038) | 单热点更新实验是 unary alphabet 的特殊情形 | 单字母下的周期/套索、有限确定表示、紧凑联表示 | 单字母字母虽同，但架构空间由程序区间结构隐式给出，K 还要求实际可执行 | **极高**：热点族的小控制器很可能主要由 unary 结构解释，不能单独支持原创性 |
| MTS 与 WFA | 经典 MTS 要求任务无关的度量迁移；工作函数算法给在线竞争保证。树度量上已有 `O(depth·log n)`，HST 上 `O(log n)`，一般度量经嵌入得到 `O(log² n)` 随机算法。[Bubeck 等](https://arxiv.org/abs/1807.04404) | split symmetric-difference 回归是 MTS；真实更新使边权依赖任务，通常非对称 | “服务＋迁移＋离线最优”和度量模型上的 WFA 基线 | 任务依赖的真实复用边、程序结构定价 | **中高**：真实模型不能继承 MTS/WFA 竞争保证；WFA 与 weighted finite automaton 缩写必须区分 |
| 参数化查询优化 | PQO 为参数空间不同区域缓存多个最优计划；线性代价下使用凸多面体区域。[Ganguly, VLDB 1998](https://www.vldb.org/dblp/db/conf/vldb/Ganguly98.html) | PCME 的预算到计划映射是 PQO/多计划缓存的一种 | 静态参数—计划包络 | 跨时刻迁移、持久中间状态与无限历史残差 | **高**：只有“多个计划随参数选择”完全不新 |
| 多目标参数化查询优化 | MPQO 同时处理多参数和多代价指标，并求全部相关计划。[Trummer–Koch](https://www.vldb.org/pvldb/vol8/p221-trummer.pdf) | work/peak/迁移锚点 Pareto 是多目标计划选择 | Pareto 计划集和参数化代价 | 历史依赖的执行核、真实迁移路径 | **高**：静态多维 Pareto 不构成新对象 |
| 动态张量重物化 | DTR 是支持动态计算图的在线贪心逐出/重算框架，按 eviction policy 工作。[arXiv:2006.09616](https://arxiv.org/abs/2006.09616) | 第三阶段 DAG 的在线释放和重算直接重合 | 在线 eviction、重算、动态图支持 | 对重复工作负载编译精确行为控制器和可执行检查点核 | **极高**：DAG 部分若只是另一种 eviction heuristic，原创主张失败 |
| 静态重物化与检查点 | Checkmate 把一般计算图时间—内存权衡写成 MILP，并有近似舍入及硬件 profile；涵盖早期线性网络检查点法。[Checkmate](https://arxiv.org/abs/1910.02653) | 固定预算下的静态架构/检查点最优与 Checkmate 同类 | 静态图、固定预算、MILP 对照和重算计划 | 多任务词、迁移复用、R/K 双核 | **高**：静态最优求解本身已有成熟工作 |
| 等价、包含、最小化边界 | 一般 tropical WFA 的 equivalence/containment 在整数域存在不可判定边界；确定、无歧义及不同权域结论不同。[Almagor–Boker–Kupferman](https://doi.org/10.1016/j.ic.2020.104651)，[Mohri 综述](https://cs.nyu.edu/~mohri/pub/hwa.pdf) | “K 与 H 对所有词等价”一般不能仅靠朴素乘积搜索断言可判定 | 一般 WFA 行为等价问题；确定有限控制器最小化 | 在有限闭包证书下把问题降为有限确定 Mealy 控制器等价 | **极高**：只有在两个残差队列真正清空时，本实现的全词等价结论才成立 |

## 对四篇最新工作的更精确影响

### arXiv:2503.23826

该工作研究“给定 nondeterministic min-plus WFA，是否存在某个等价 deterministic WFA”。TCMK 第二阶段的成对有限 Horizon 残差搜索没有解决该问题；第三阶段显式闭包 BFS 只对实际到达有限不动点的具体自动机给出构造。一般决定过程远比普通 BFS 复杂。因此：

- 有限闭包 BFS 成功是 determinisability 的一个构造见证；
- BFS 持续增长不能直接推出不可确定化；
- MORPH-N 不得声称给出一般 determinisability 算法。

### arXiv:2602.01221

该工作用 gap 的统一上界把 2025 年的可判定性变为 primitive-recursive 算法。MORPH-N 的“所有矩阵条目有限＋有限整数列差”定理是一个非常强的特殊充分条件：它直接给出一步后坐标差界，但不覆盖带 `∞` 不可行边的一般模型。其价值若成立，只能是对持续矩阵维护这一程序族给出小而可计算的界。

### arXiv:2512.09484

行为核 R 不应被描述为一般最小 tropical CRA。当前实现只在已得到有限残差图后运行普通确定 Mealy 分区细化；这避开了论文证明不可判定的通用 counter minimisation。可执行核 K 又是“原 WFA 状态子集”的另一问题，不等于寄存器数最小化。

### arXiv:2606.26038

单热点实验只有一个任务字母，直接落入 unary WFA。论文的二次表示与 coNP 完备结果意味着：单热点下观察到的有限小 R 不能作为新理论证据。第三阶段必须依靠多字母更新、隐式程序空间与真实证书说明额外结构。

## 当前原创性判定

| 主张 | 判定 |
|---|---|
| 动态形态优化可写成 min-plus WFA | 已有对象上的直接归约，不原创 |
| 规范化残差决定未来增量 | 标准确定控制状态性质，不原创 |
| 有限残差闭包可生成精确控制器 | 标准确定化构造的实例，不原创 |
| R 的有限 Mealy 行为商 | 标准自动机最小化，不原创 |
| K 必须是实际可执行程序架构子集 | 与普通确定化不同，可能保留 |
| 从指数级隐式程序空间精确地产生必要 K | 若无需全枚举且有完备证书，可能保留 |
| 真实复用迁移几何与证明携带形态 | 可能保留，但需第二领域和性能证据 |
| R/K 双核共同最小化 | 作为组合问题可能保留；单独两部分均有强最近邻 |

最终优先权风险仍是 **高**。若完整势能定价和第二真实领域没有出现程序结构特有的严格结果，MORPH-N 应判为 `partially_supported`，不能主张新的通用自动机理论。
