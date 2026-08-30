# 证明携带语义形态包络

本项目是“架构即状态”思想的第一代可运行原型。

核心文件：

- `REPORT_ZH.md`：完整中文研究报告；
- `src/morph_envelope.cpp`：C++ 精确包络、在线切换和完整测试；
- `src/tensor_morphology.py`：一般张量/连接超图扩展；
- `tests/independent_runtime_check.py`：独立并发切换验证；
- `results/morph_results_full.json`：完整原始结果；
- `run_all.sh`：一键复现。

```bash
bash run_all.sh
```

当前范围是结合型区间计算和无环张量/连接超图，不是任意程序。
