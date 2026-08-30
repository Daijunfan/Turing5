# Turing5：架构即状态

公开研究仓库，按阶段保留所有代码、原始结果、反例和负结果。

## 阶段索引

1. `MORPH_Envelope_Research_Prototype/`：第一阶段 PCME/MCS；
2. `phase2_transition_complete_kernel/`：第二阶段 TCMK/CE-TCMK；
3. `phase3_morphology_nerode_kernel/`：第三阶段 MORPH-N 无限时域双核形态编译。

当前状态见：

- `PROJECT_STATUS.md`
- `LATEST_RESULTS.json`
- `phase3_morphology_nerode_kernel/REPORT_ZH.md`

第三阶段一键复现：

```bash
bash phase3_morphology_nerode_kernel/run_all.sh
bash phase3_morphology_nerode_kernel/run_sanitizers.sh
```

第三阶段最终判定为 `partially_supported`：显式有限架构、有限残差闭包下的双核编译成立；完整隐式未来势能定价、n=128 和预算 DAG 的无限闭包未成功。
