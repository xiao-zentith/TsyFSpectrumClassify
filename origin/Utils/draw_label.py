import matplotlib.pyplot as plt
import numpy as np

# ====================
# 学术图表样式设置
# ====================
plt.rcParams['font.family'] = 'Arial'  # 全局字体设置为Arial
plt.rcParams['font.size'] = 10         # 基础字号
plt.rcParams['axes.linewidth'] = 1.2   # 坐标轴线宽
plt.rcParams['axes.spines.right'] = False  # 关闭右侧脊柱
plt.rcParams['axes.spines.top'] = False    # 关闭顶部脊柱

# ====================
# 数据准备
# ====================
categories = ['UNet 1', 'UNet 2', 'UNet 3']
values_abs = [21.67, 18.35, 16.48]    # 绝对值数据（范围10-30）
values_pct = [99.53, 99.75, 99.83]    # 百分比数据（范围99%+）

# ====================
# 创建画布和坐标轴
# ====================
fig, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)
ax2 = ax1.twinx()

# 设置柱状图位置参数
x = np.arange(len(categories))  # 横坐标位置
width = 0.35  # 柱子宽度

# ====================
# 绘制柱状图
# ====================
# 绝对值柱子（左轴）
rects1 = ax1.bar(x - width/2, values_abs, width,
                color='#2c7bb6', edgecolor='black', linewidth=0.8,
                label='RMSE')

# 百分比柱子（右轴）
rects2 = ax2.bar(x + width/2, values_pct, width,
                color='#d7191c', edgecolor='black', linewidth=0.8,
                label='Cosine similarity (%)')

# ====================
# 坐标轴设置
# ====================
# 左轴设置（绝对值）
ax1.set_ylabel('RMSE', fontweight='bold')
ax1.set_ylim(0, 35)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontweight='bold')
ax1.tick_params(axis='y', which='major', length=5, width=1.2)

# 右轴设置（百分比）
ax2.set_ylabel('Cosine similarity (%)', fontweight='bold')
ax2.set_ylim(99, 100.5)
ax2.tick_params(axis='y', which='major', length=5, width=1.2)

# ====================
# 添加数据标签
# ====================
def add_labels(ax, rects, precision=1):
    """自动添加数据标签"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.{precision}f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

add_labels(ax1, rects1, 2)
add_labels(ax2, rects2, 2)

# ====================
# 图例和网格设置
# ====================
# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2,
          loc='upper left', frameon=False)

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# ====================
# 保存输出
# ====================
plt.tight_layout()
plt.savefig('academic_bar_chart.png', bbox_inches='tight', dpi=300)
plt.show()