# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.patches as mpatches
#
# # 读取数据
# df = pd.read_csv(r"D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\头测-an\data.csv", encoding="gbk")
# df.columns = df.columns.str.strip()
#
# mpl.rcParams['font.family'] = 'Times New Roman'
#
# # 抖动
# df["age_jittered"] = df["age"] + np.random.uniform(0, 0, size=len(df))
#
# palette = {"Female": "#ed7d31", "Male": "#4472c4"}
#
# sns.set(style="whitegrid")
# plt.figure(figsize=(7.16, 4.5))  # IEEE双栏
#
# # === 小提琴图（底层，手动设置透明度） ===
# violin = sns.violinplot(
#     data=df,
#     x="cervical_stage",
#     y="age_jittered",
#     hue="sex",
#     split=True,
#     inner=None,
#     palette=palette,
#     saturation=0.9
# )
#
# # 手动设置小提琴透明度
# for patch in violin.collections:
#     patch.set_alpha(0.9)
#
# # === 箱线图（白底黑边，最上层） ===
# sns.boxplot(
#     data=df,
#     x="cervical_stage",
#     y="age_jittered",
#     hue="sex",
#     dodge=True,
#     linewidth=1.5,
#     fliersize=0,
#     width=0.25,
#     color='white',  # 设置白色箱体
#     boxprops=dict(facecolor='white', edgecolor='black'),
#     capprops=dict(color='black'),
#     whiskerprops=dict(color='black'),
#     medianprops=dict(color='black')
# )
#
# # === strip 数据点 ===
# sns.stripplot(
#     data=df,
#     x="cervical_stage",
#     y="age_jittered",
#     hue="sex",
#     hue_order=["Male", "Female"],
#     dodge=True,
#     palette=palette,
#     size=3,
#     alpha=1,
#     jitter=True
# )
#
# # === 图例 ===
# legend_handles = [
#     mpatches.Patch(color=palette["Male"], label="Male"),
#     mpatches.Patch(color=palette["Female"], label="Female")
# ]
# plt.legend(
#     handles=legend_handles,
#     title="Sex",
#     loc="upper left",
#     frameon=True,
#     framealpha=1
# )
#
# plt.title("Age Distribution Across Cervical Stages by Sex", fontsize=12)
# plt.xlabel("Cervical Stages", fontsize=10)
# plt.ylabel("Age (Years)", fontsize=10)
# plt.xticks(fontsize=9)
# plt.yticks(fontsize=9)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
#
# plt.savefig("violin_boxplot_whitebox.pdf", format="pdf", bbox_inches='tight')
# plt.show()
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置主题和样式（可选）
# sns.set_theme(style="whitegrid")
#
# # 示例数据集（使用Seaborn内置数据集）
# tips = sns.load_dataset("tips")
#
# # 创建图形和坐标轴
# plt.figure(figsize=(10, 6))
#
# # 1. 绘制小提琴图（内部不填充颜色）
# sns.violinplot(
#     x="day",
#     y="total_bill",
#     data=tips,
#     inner=None,  # 不显示内部图形
#     color="lightblue"  # 小提琴颜色
# )
#
# # 2. 在同一个坐标轴上添加箱线图
# sns.boxplot(
#     x="day",
#     y="total_bill",
#     data=tips,
#     width=0.15,  # 控制箱线宽度
#     color="white",  # 填充白色
#     flierprops={"marker": "o", "markerfacecolor": "black"},  # 异常值黑点
#     boxprops={"edgecolor": "black", "linewidth": 1.5},  # 箱体黑边
#     whiskerprops={"color": "black", "linewidth": 1.5},  # 须线黑色
#     capprops={"color": "black", "linewidth": 1.5},  # 横线黑色
#     medianprops={"color": "black", "linewidth": 1.5}  # 中位线黑色
# )
#
# # 添加标题和标签
# plt.title("Violin Plot with White Boxplot", fontsize=14)
# plt.xlabel("Day of Week", fontsize=12)
# plt.ylabel("Total Bill ($)", fontsize=12)
#
# # 显示图形
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置全局字体为 Times New Roman，字号为14
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14
#
# # 原始混淆矩阵
# conf_mat = np.array([
#     [103, 13, 6, 0, 0, 0],
#     [10, 64, 15, 2, 0, 0],
#     [5, 12, 46, 2, 2, 0],
#     [0, 4, 8, 20, 9, 2],
#     [0, 0, 1, 3, 26, 1],
#     [0, 0, 0, 1, 3, 2]
# ])
#
# # 计算每一行的百分比
# conf_mat_percent = conf_mat / conf_mat.sum(axis=1, keepdims=True) * 100
#
# # 类别标签
# labels = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']
#
# # 绘制热图
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(
#     conf_mat_percent,
#     annot=True,
#     fmt=".1f",
#     cmap="Blues",
#     xticklabels=labels,
#     yticklabels=labels,
#     cbar_kws={'label': 'Percentage (%)'},  # 添加 colorbar 标签
#     annot_kws={"size": 14}  # 单独设置注释字号
# )
#
# # 设置坐标轴标签
# ax.set_xlabel("Predicted Label", fontsize=14)
# ax.set_ylabel("True Label", fontsize=14)
#
# # 可选：设置标题
# # ax.set_title("Normalized Confusion Matrix (%)", fontsize=14)
#
# # 显示并保存为PDF
# plt.tight_layout()
# plt.savefig("Confusion_Matrix.pdf", format="pdf", bbox_inches='tight')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为 Times New Roman，字号为14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# 原始混淆矩阵
conf_mat = np.array([
    [103, 13, 6, 0, 0, 0],
    [10, 64, 15, 2, 0, 0],
    [5, 12, 46, 2, 2, 0],
    [0, 4, 8, 20, 9, 2],
    [0, 0, 1, 3, 26, 1],
    [0, 0, 0, 1, 3, 2]
])

# 类别标签
labels = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']

# 绘制热图（使用原始计数）
plt.figure(figsize=(7.5, 4.5))
ax = sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",  # 显示整数
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={'label': 'Number of Samples'},  # colorbar 标签
    annot_kws={"size": 14}
)

# 设置坐标轴标签
ax.set_xlabel("Predicted Label", fontsize=14)
ax.set_ylabel("True Label", fontsize=14)

# 可选：设置标题
# ax.set_title("Confusion Matrix (Counts)", fontsize=14)

# 显示并保存为PDF
plt.tight_layout()
plt.savefig("Confusion_Matrix_Counts.pdf", format="pdf", bbox_inches='tight')
plt.show()