import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 路径设置
# =========================
exp_name = "traffic_gnn_exp60"
history_dir = f"/HUBU-AI095/xhx/log/{exp_name}/history"

joint_file = os.path.join(history_dir, "joint_history.csv")

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(joint_file)

# =========================
# 3. 绘图参数
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

# 顶部：train 各loss
train_metrics = [
    ("train_loss", "Total Loss"),
    ("train_base_loss", "Base Loss"),
    ("train_recent_loss", "Recent Loss"),
    ("train_event_loss", "Event Loss"),
]

markers = ["o", "s", "^", "d"]

for i, (metric, label) in enumerate(train_metrics):
    if metric in df.columns:
        axes[0].plot(
            df["epoch"],
            df[metric],
            marker=markers[i],
            linewidth=2.0,
            markersize=5,
            label=label
        )

axes[0].set_title("Train Loss Curves", fontsize=15, fontweight="bold", pad=8)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].grid(True, linestyle="--", alpha=0.35)
axes[0].legend(fontsize=10, loc="upper right", frameon=True)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# 底部：val 各loss
val_metrics = [
    ("val_loss", "Total Loss"),
    ("val_base_loss", "Base Loss"),
    ("val_recent_loss", "Recent Loss"),
    ("val_event_loss", "Event Loss"),
]

for i, (metric, label) in enumerate(val_metrics):
    if metric in df.columns:
        axes[1].plot(
            df["epoch"],
            df[metric],
            marker=markers[i],
            linewidth=2.0,
            markersize=5,
            label=label
        )

axes[1].set_title("Validation Loss Curves", fontsize=15, fontweight="bold", pad=8)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Loss", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.35)
axes[1].legend(fontsize=10, loc="upper right", frameon=True)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# 总标题
fig.suptitle(
    "Joint Stage Loss Curves",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# =========================
# 4. 保存 PNG
# =========================
save_path = os.path.join(history_dir, f"{exp_name}_joint_loss_show.png")
plt.savefig(save_path, bbox_inches="tight")
plt.show()

print(f"PNG saved to: {save_path}")