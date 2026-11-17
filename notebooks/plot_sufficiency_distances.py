"""绘制证据充分性实验的逐层差异图。

纯 CPU 脚本：只读取 results/hidden_states_sufficiency.json，不加载模型。
图含两个面板：(1) 逐层 L2 距离；(2) 逐层余弦相似度。
横轴为隐藏状态下标：0 = embedding，1..35 = 各 Transformer 块，36 = 最终 RMSNorm 之后。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "results" / "hidden_states_sufficiency.json"
OUT = PROJECT_ROOT / "results" / "hidden_states_sufficiency.png"

record = json.loads(RESULT.read_text(encoding="utf-8"))
l2 = record["per_state_comparison"]["l2_distance"]
cos = record["per_state_comparison"]["cosine_similarity"]
n_layers = record["model_structure"]["num_hidden_layers"]
x = list(range(len(l2)))

# 差异加速增长的区间（后段层），仅作视觉提示
accelerate_from = int(n_layers * 0.58)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax1.plot(x, l2, marker="o", markersize=3.5, color="#2563eb", linewidth=1.6)
ax1.axvspan(accelerate_from, n_layers, color="#f59e0b", alpha=0.12,
            label="late layers (divergence accelerates)")
peak_i = max(range(len(l2)), key=lambda i: l2[i])
ax1.annotate(f"max {l2[peak_i]:.1f} @ {peak_i}", xy=(peak_i, l2[peak_i]),
             xytext=(peak_i - 12, l2[peak_i] * 0.92), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="0.3"))
ax1.set_xlabel("hidden-state index (0=embedding, 1-35=blocks, 36=after final RMSNorm)")
ax1.set_ylabel("L2 distance (last prompt token)")
ax1.set_title("Per-layer L2 distance")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(alpha=0.3)

ax2.plot(x, cos, marker="o", markersize=3.5, color="#dc2626", linewidth=1.6)
ax2.axvspan(accelerate_from, n_layers, color="#f59e0b", alpha=0.12)
min_i = min(range(len(cos)), key=lambda i: cos[i])
ax2.annotate(f"min {cos[min_i]:.3f} @ {min_i}", xy=(min_i, cos[min_i]),
             xytext=(min_i + 3, cos[min_i] - 0.012), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="0.3"))
ax2.set_xlabel("hidden-state index")
ax2.set_ylabel("cosine similarity")
ax2.set_title("Per-layer cosine similarity")
ax2.set_ylim(0.82, 1.004)
ax2.grid(alpha=0.3)

fig.suptitle(
    "Qwen2.5-3B-Instruct: sufficient vs insufficient evidence "
    "(single pair, descriptive only)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
