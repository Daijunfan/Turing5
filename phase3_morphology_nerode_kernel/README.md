# MORPH-N：无限时域双核形态编译

第三阶段独立研究目录。第一、第二阶段代码和原始结果保持只读。

最终判定：`partially_supported`。

核心结果：

- 第二阶段 detached 干净 worktree 全量复审通过；
- `2,279,080` 被机器证据分类为 Horizon=6 有限搜索，不是无限闭包；
- 真实 14 架构模型：218 原始残差、56 行为类、4 可执行架构；
- 闭包队列为空，1090 条边由独立 checker 重放；
- n≤9 定价与全 Catalan oracle 一致；n=64 完成，n=128 资源失败；
- 1000 节点实际 NumPy DAG 全部可行输出正确，但无限闭包和性能主张失败。

详见 `REPORT_ZH.md`、`THEORY_ZH.md` 和 `LITERATURE_BOUNDARY_ZH.md`。

复现：

```bash
bash run_all.sh
bash run_sanitizers.sh
```
