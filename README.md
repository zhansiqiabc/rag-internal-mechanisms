# RAG Internal Mechanisms

最新科研进度见 [rag.md](rag.md)。E2.1 本轮审计纳入 23 条、暂缓 7 条；完整 30 条候选数据仍保留。2026-11-19 已完成 23 组 × 三条件共 69 条 v2 回答行为复测，结果见 [复测报告](results/e2_v2_pilot_report.md)；审核来源与逐条依据见 [裁决记录](results/e21_conflict_adjudication.md)。

最小可复现的 Qwen2.5-3B-Instruct 隐藏状态实验。当前只比较两段受控检索证据；不微调模型，也不把层间差异解释为因果效应。

## 环境

Windows、NVIDIA GPU、Python 3.12。项目已创建 `.venv`。如需独立安装，先安装与驱动兼容的 PyTorch CUDA 构建，再安装 `requirements.txt`。具体版本记录在 `results/environment.json`。

```powershell
uv venv --python D:\Anaconda3\python.exe .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe experiments\hidden_states_minimal.py
```

当前网络下载未获准；已验证现有 `D:\workspace\resume\qlora-text2sql\.venv` 的 PyTorch 2.11.0+cu130 能在 RTX 5060 Ti 上运行矩阵乘法。可临时用它运行最小实验，不改动该环境：

```powershell
& 'D:\workspace\resume\qlora-text2sql\.venv\Scripts\python.exe' experiments\hidden_states_minimal.py --local-files-only
```

这个现有环境缺少 scikit-learn，但最小隐藏状态脚本不依赖它。模型文件若未缓存，须先取得模型权重才能运行。

已使用本机 `D:\workspace\resume\qlora-text2sql\model`（Qwen3-4B-Instruct-2507）完成脚本烟雾测试，见 `results/qwen3_smoke.json`：两条件各 34 tokens，37 个隐藏状态张量，每个形状 `[1, 34, 2560]`。这是 Qwen3 的程序验证，不能代替目标 Qwen2.5-3B-Instruct 的结果。

首次运行会从 Hugging Face 下载 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。可用 `--model` 指定本地模型路径，或用 `--local-files-only` 禁止联网。实验只保存张量形状与按层汇总的最后提示词 token 状态差异，不保存完整隐藏状态张量。

## 目录

- `data/`：后续实验数据
- `experiments/`：实验脚本
- `results/`：运行结果与环境记录
- `notebooks/`：可视化与分析
- `notes/`：论文笔记

比较使用仅年份不同的两条虚构证据。输出的距离仅是描述性测量；要研究证据利用，还需要答案行为、无证据基线及更严格的对照。

## E1：受控事实关系与回答行为分析（2026-11-17）

30 组虚构人物×机构事实 × A1/A2/B1/B2 四条件（正确答案候选 × 前半句/后半句位置）= 120 条，测 Qwen2.5-3B-Instruct 的回答是否受答案位置影响。

- `experiments/e1_build_dataset.py`：样本构建 + 运行前自动校验（输入 `data/e1_groups_raw.json`，输出 `data/e1_samples_v1.json`）
- `experiments/e1_run_inference.py`：本地推理（贪心，`repetition_penalty=1.0` 覆盖模型默认 1.05；先 `--limit 4` 冒烟再全量）
- `experiments/e1_analyze.py`：保守判定 + 统计报告
- 结果：`results/e1_smoke.json`、`results/e1_responses.json`、`results/e1_scored.json`、`results/e1_rows.csv`、`results/e1_report.md`

结论摘要：总体 117/120 = 97.5%；A1/A2/B2 均 100%，B1 为 90%（3 条错误全是模型退化成「候选A」字母作答）；前半句 95.0% vs 后半句 100.0%（z=-1.754，p=0.079，不显著）；首句机构被选率 47.5%——未发现系统性首置偏差。

## E2：证据状态与回答行为实验（2026-11-17）

复用 E1 的 30 组实体，每组三种证据条件（充分/不足/冲突）共 90 条，无候选字母、开放式作答（可答机构名/「无法确定」/「证据冲突」）。证据状态与模型行为分开判定。

- `experiments/e2_build_dataset.py`：组装 + 校验（输入 `data/e2_craft_input.json` + `data/e2_crafted.json`，输出 `data/e2_samples_v1.json`）
- `experiments/e2_run_inference.py` / `experiments/e2_analyze.py`：推理与行为统计
- 结果：`results/e2_smoke.json`、`e2_responses.json`、`e2_scored.json`、`e2_rows.csv`、`e2_report.md`

原结果摘要：充分 30/30 给金标机构；不足 30/30 回答“无法确定”；原标为冲突的 30 条中，21 条输出“证据冲突”、1 条给金标机构、4 条给另一机构、4 条回答“无法确定”。E2.1 后续审计发现原冲突标签存在问题，因此 21/30 = 70% 只能描述原标签下的输出比例，不能作为严格冲突识别率；标签质量和句式影响目前无法分离。E2 原始数据与结果保留，不回写为 v2 结果。

## E2 v2：23 组配对行为复测（2026-11-19）

使用相同 23 组的充分、不足、冲突材料共 69 条；充分/不足取 E2 v1，冲突取本轮 v2 筛选范围。沿用 E2 提示词和生成设置，输出另存。

- 充分：23/23 输出正确机构；不足：23/23 输出“无法确定”。
- 冲突：0/23 输出“证据冲突”，15/23 选择机构（第一机构 10、第二机构 5），8/23 输出“无法确定”。
- 同样 23 组在旧版中有 17 条输出冲突；其中 21 条冲突材料改过文本，因此不能据此判断模型能力提高或退步。
- 69 条输出完整且未截断；Codex 逐条阅读复核，无独立人工审核。没有提取隐藏状态、训练探针或做干预。
- 详情与复现入口见 [复测报告](results/e2_v2_pilot_report.md)。本轮止于记录行为现象，不能证明模型内部没有表示或未利用冲突信息。
