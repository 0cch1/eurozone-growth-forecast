# 毕设交付范围：最低标准 + 唯一可选 AutoML

本文档明确**必须交付**与**唯一保留的可选内容**，便于按“最低标准先做到最好”，其余留作 Future work。

---

## 一、最低标准（必须完成）

| # | 交付项 | PDD 依据 | 状态 / 行动 |
|---|--------|----------|-------------|
| 1 | **数据** | §3.1, §4.1 | ✅ 已有：EA 年度 panel（fetch + build + 备份） |
| 2 | **数据字典** | §3.1, §3.4 | ✅ 已完成：`docs/data_dictionary.md`（panel_yearly 与 raw 列含义、单位、来源、派生变量说明） |
| 3 | **多模型 + 时序评估** | §3.2–3.3, §4.4 | ✅ 已有：linear, MLP, XGBoost；TimeSeriesSplit；MAE/RMSE |
| 4 | **R² 补充指标** | §4.4 | ✅ 已完成：`evaluation.regression_metrics()` 与 `compare_models` 已增加 R²，表格含 mae/rmse/r2 |
| 5 | **可解释性** | §4.5 | ✅ 已有：SHAP summary、PDP、局部解释、**Permutation importance**（`interpretation.permutation_importance`，`run_interpretation` 会生成图） |
| 6 | **复现说明** | §4.6 | ✅ 已完成：`docs/Reproducibility.md`（环境、种子、推荐运行顺序） |
| 7 | **报告 + 展示** | §3.4 | ⬜ 你做：dissertation-style 报告 + 答辩/展示（方法、结果、performance–interpretability 讨论） |

**目标**：把以上 1–6 在代码与文档里做到“清晰、可复现”；7 由你独立完成。

---

## 二、唯一可选内容：AutoML

- **保留**：使用 **FLAML AutoML** 作为“强基线”，与手动选择的模型对比（同一数据、时间 holdout）。
- **做法**：运行 `python -m src.automl_experiments`，将得到的 MAE/RMSE 写进报告，用 1 段说明“AutoML 与手选模型的对比及意义”。
- **不扩展**：不承诺调参、多组 AutoML 实验或与其他 AutoML 库对比；一个时间预算下一组结果即可。

---

## 三、明确不纳入本次交付（Future work）

以下**不**作为本次毕设的承诺内容，可在报告“局限与未来工作”中一笔带过：

- 国家级 panel、地图可视化
- 符号回归（Py-OPERON）在报告中的系统对比
- 子时段稳健性分析（若时间紧可省略）
- Ridge/Lasso、GAM、多项式回归等额外模型族
- Rolling/expanding window 评估

---

## 四、建议完成顺序

1. ~~数据字典~~ ✅  
2. ~~R² + 复现说明~~ ✅  
3. ~~Permutation importance（并跑一次 SHAP/PDP，确保图可用）~~ ✅  
4. 跑 AutoML，把结果整理进报告  
5. 写报告与展示  

这样在“最低标准先做到最好”的前提下，只多保留 AutoML 这一项可选内容，范围清晰、可交付。
