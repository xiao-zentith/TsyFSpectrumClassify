"""
路径管理器使用示例
"""

from src.utils.path_manager import PathManager

# 初始化路径管理器
pm = PathManager()

# 获取各种路径
data_raw = pm.get_path('data', 'raw')
data_processed = pm.get_path('data', 'processed')
models_classification = pm.get_path('models', 'classification')
results_regression = pm.get_path('results', 'regression')

print(f"原始数据路径: {data_raw}")
print(f"处理后数据路径: {data_processed}")
print(f"分类模型路径: {models_classification}")
print(f"回归结果路径: {results_regression}")

# 在代码中使用
import json

# 读取配置文件
config_path = pm.get_path('configs', 'main') / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 保存模型
model_path = pm.get_path('models', 'classification') / 'best_model.pth'
# torch.save(model.state_dict(), model_path)
