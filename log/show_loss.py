import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 路径设置
# =========================
exp_name = "traffic_gnn_exp60"
history_dir = f"/HUBU-AI095/xhx/log/{exp_name}/history"

stage_files = {
    "Stage 1": os.path.join(history_dir, "stage1_history.csv"),
    "Stage 2": os.path.join(history_dir, "stage2_history.csv"),
    "Stage 3": os.path.join(history_dir, "stage3_history.csv"),
}

# =========================
# 2. 读取数据
# =========================
history = {}
for stage_name, file_path in stage_files.items():
    df = pd.read_csv(file_path)
    history[stage_name] = df

# =========================
# 3. 绘图参数
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

plot_items = [
    ("train_loss", "Training Loss", axes[0, 0], "train_loss"),
    ("val_loss",   "Validation Loss", axes[0, 1], "val_loss"),
    ("train_mae",  "Training MAE", axes[1, 0], "train_mae"),
    ("val_mae",    "Validation MAE", axes[1, 1], "val_mae"),
]

markers = ["o", "s", "^"]

for metric, title, ax, ylabel in plot_items:
    for i, (stage_name, df) in enumerate(history.items()):
        if metric in df.columns:
            ax.plot(
                df["epoch"],
                df[metric],
                marker=markers[i],
                linewidth=2.0,
                markersize=5,
                label=stage_name
            )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=10, loc="best", frameon=True)

    # 更像科研图
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# 总标题和子图间距拉开，避免重叠
fig.suptitle(
    "Training History Comparison Across Three Stages",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# =========================
# 4. 只保存 PNG
# =========================
save_path = os.path.join(history_dir, f"{exp_name}_show.png")
plt.savefig(save_path, bbox_inches="tight")
plt.show()

print(f"PNG saved to: {save_path}")