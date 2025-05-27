import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 加载.mat文件
mat_file_path = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\CMA1_fluo\CMA1_fluo.mat'  # 替换为你的.mat文件路径
data = loadmat(mat_file_path)

# 提取所需的数据
all_sample_x = data['AllSamples_X']
axis1 = data['Axis1'].flatten()
axis2 = data['Axis2'].flatten()

# 检查形状是否符合预期
if all_sample_x.shape != (105, 4303):
    raise ValueError("AllSample_X does not have the expected shape of (105, 4303)")

# 创建输出目录（如果不存在）
output_dir = r'C:\Users\xiao\Desktop\Draw-flatbread\data\open_dataset\fish\xlsx'
os.makedirs(output_dir, exist_ok=True)

# 将每个样本转换为13x331矩阵并保存为单独的.xlsx文件
for i in range(all_sample_x.shape[0]):
    sample_vector = all_sample_x[i, :]
    sample_matrix = sample_vector.reshape(13, 331)
    sample_matrix = np.transpose(sample_matrix)

    # 创建DataFrame并将索引设置为Axis1和Axis2的值
    df = pd.DataFrame(sample_matrix, index=axis1, columns=axis2)

    # 保存为.xlsx文件
    output_file_path = os.path.join(output_dir, f'sample_{i + 1}.xlsx')
    df.to_excel(output_file_path, engine='openpyxl')

print(f"Successfully saved {all_sample_x.shape[0]} samples to {output_dir}")



