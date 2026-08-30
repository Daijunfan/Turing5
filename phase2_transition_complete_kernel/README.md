# Transition-Complete Morphology Kernel (TCMK)

这是独立于第一代 PCME 剪枝实现的第二阶段研究目录。旧目录和旧结果未被修改。

当前实现包含：

- 全 Catalan 括号树枚举，以及显式的“树 + 递归求值顺序”状态枚举；
- 全程 `fractions.Fraction` 的离线 min-plus 动态 oracle；
- 显式初始架构或虚拟 source 构造成本；
- 静态 Pareto、锚点 Pareto、上下文规则、CE-TCMK 和最小核 oracle；
- 成对规范化残差搜索、反例、全状态路径、桥值和删除证书；
- 最佳固定、瞬时最优、迁移贪心、MCS，以及仅在度量模型启用的 WFA；
- 真实“持续维护矩阵链乘积”模型、精确小规模迁移计划、迁移证书和独立 checker；
- 模素数连续矩阵更新与逐步从头左折叠对照；
- n=16/32/64 的局部结合重写候选图实验（与小规模全状态结论严格分栏）。

## 最短复现路径

反例复现只需要 Python 标准库：

```bash
make counterexample
```

完整测试与批量整数残差实验使用 NumPy：

```bash
python3 -m pip install -e '.[experiments]'
make test
```

核心真实实验约需 1–2 分钟：

```bash
make core
```

其他实验：

```bash
make scale
make exhaustive
```

结果写入 `results/`。核心入口和职责：

- `tcmk/morphology.py`：树、求值顺序、工作量和峰值；
- `tcmk/dynamics.py`：显式 source 的全状态精确 DP；
- `tcmk/residual.py`：规范化成对残差分离；
- `tcmk/kernel.py`：桥值、CE-TCMK、最小核；
- `tcmk/real_migration.py`：实际复用、重算、字节和内存模型；
- `tcmk/certificate_checker.py`：不调用 planner 的独立证书检查；
- `tcmk/matrix_runtime.py`：连续真实矩阵执行；
- `THEORY_ZH.md`：定理、反例与相关工作定位；
- `REPORT_ZH.md`：实验结论、成功门槛和失败条件审计。

## 重要范围声明

`SplitDistanceModel` 只用于精确复现第一代 7.395940832473341% 反例和算法网格回归，不作为最终真实性证据。真实结论来自 `PersistentMatrixModel`。

真实迁移边权依赖更新叶，因此通常不是经典 metrical task system。WFA 在该模型中默认标记为不适用，不声称竞争保证。

大规模局部重写候选图不是全 Catalan 状态空间；使用启发式迁移的行只报告上界，不能当作全状态精确 oracle。
