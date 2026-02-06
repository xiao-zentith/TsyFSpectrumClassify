# 模型目录

此目录用于存储训练好的模型文件。

## 目录结构
- `classification/` - 分类模型
- `regression/` - 回归模型

## 使用方法
```python
from src.utils.path_manager import PathManager

pm = PathManager()
model_path = pm.get_path('models', 'classification')
```
