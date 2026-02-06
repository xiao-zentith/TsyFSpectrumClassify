import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件中的二维矩阵数据
file_path = r'C:\Users\xiao\Desktop\academic_papers\data\open_dataset\fish\xlsx\sample_69.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'    # 替换为你的工作表名称

# 使用pandas读取Excel文件
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 将DataFrame转换为numpy数组以便绘图
matrix_data = df.values

# 创建等高线图
plt.figure(figsize=(8, 6))
contour = plt.contourf(matrix_data, cmap='viridis')
plt.colorbar(contour)
plt.title('等高线图')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图形
plt.show()