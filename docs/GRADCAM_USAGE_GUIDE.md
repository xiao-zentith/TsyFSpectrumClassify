# Grad-CAM 使用指南

## 概述

本项目实现了专门针对光谱数据的Grad-CAM（Gradient-weighted Class Activation Mapping）可解释性分析工具。该工具可以帮助理解深度学习模型在处理光谱数据时关注的重要区域和特征。

## 功能特性

- **多模型支持**: 支持UNET、DualUNet等回归模型
- **光谱特化**: 专门针对光谱数据的分析优化
- **批量处理**: 支持批量样本分析
- **层级分析**: 可分析模型不同层的重要性
- **真实数据测试**: 提供基于真实光谱数据的测试案例

## 核心组件

### SpectralGradCAM 类

主要的Grad-CAM分析器，位于 `src/utils/interpretability/gradcam.py`

#### 主要参数

- `model`: 要分析的PyTorch模型
- `target_layers`: 目标分析层（可选，自动检测）
- `model_type`: 模型类型（'unet', 'dualunet'等）
- `use_cuda`: 是否使用GPU加速

#### 主要方法

- `generate_cam(input_tensor, target_class=None)`: 生成CAM热力图
- `get_important_wavelengths(cam, threshold=0.5)`: 提取重要波长
- `analyze_spectral_regions(cam, input_shape)`: 分析光谱区域

## 快速开始

### 1. 基础使用示例

```python
import torch
from src.utils.interpretability.gradcam import SpectralGradCAM
from src.regression.models.unet import UNET

# 创建模型
model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128])
model.eval()

# 准备输入数据 (batch_size, channels, height, width)
input_tensor = torch.randn(1, 1, 63, 21)

# 创建Grad-CAM分析器
grad_cam = SpectralGradCAM(model, model_type='unet')

# 生成CAM
results = grad_cam.generate_cam(input_tensor, target_class=None)

# 查看结果
for layer_name, result in results.items():
    print(f"层 {layer_name}:")
    print(f"  - CAM形状: {result['cam'].shape}")
    print(f"  - 峰值强度: {result['peak_intensity']:.4f}")
    print(f"  - 平均强度: {result['mean_intensity']:.4f}")
```

### 2. 真实数据分析

```python
import pandas as pd
from src.utils.interpretability.gradcam import SpectralGradCAM

# 加载真实光谱数据
input_df = pd.read_excel('data.xlsx', header=None, usecols=lambda x: x != 0, skiprows=1)
input_tensor = torch.from_numpy(input_df.to_numpy()).float().unsqueeze(0).unsqueeze(0)

# 进行分析
grad_cam = SpectralGradCAM(model, model_type='unet')
results = grad_cam.generate_cam(input_tensor)

# 获取第一个层的结果
first_layer = list(results.keys())[0]
result = results[first_layer]
print(f"峰值强度: {result['peak_intensity']:.4f}")
```

## 示例脚本

### 1. 基础演示脚本

运行 `examples/gradcam_usage_examples.py` 查看各种使用场景：

```bash
cd /home/asus515/PycharmProjects/TsyFSpectrumClassify_remote
python examples/gradcam_usage_examples.py
```

该脚本包含以下示例：
- 基础Grad-CAM分析
- 模型对比分析
- 批量数据处理
- 2D光谱分析
- 层重要性分析
- 训练集成示例

### 2. 真实数据测试脚本

运行 `examples/gradcam_real_data_test.py` 使用真实光谱数据：

```bash
python examples/gradcam_real_data_test.py
```

该脚本包含：
- 真实数据基础分析
- 多模型对比
- 批量样本分析
- 层级重要性分析

## 输出结果说明

### CAM结果字典

每个层的分析结果包含以下字段：

```python
{
    'cam': torch.Tensor,              # CAM热力图张量
    'cam_shape': tuple,               # CAM形状
    'peak_intensity': float,          # 峰值强度 (0-1)
    'mean_intensity': float,          # 平均强度 (0-1)
    'important_wavelengths': list,    # 重要波长列表
    'spectral_regions': dict,         # 光谱区域分析
    'activation_stats': dict          # 激活统计信息
}
```

### 强度指标含义

- **峰值强度**: CAM中最大激活值，表示模型最关注的区域强度
- **平均强度**: CAM的平均激活值，表示整体关注度
- **重要波长**: 激活值超过阈值的波长位置
- **光谱区域**: 不同光谱区域的激活统计

## 高级用法

### 1. 自定义目标层

```python
from src.utils.interpretability.gradcam import get_model_target_layers

# 获取所有可用层
target_layers = get_model_target_layers(model, 'unet')

# 选择特定层进行分析
selected_layers = target_layers[:3]  # 前3层
grad_cam = SpectralGradCAM(model, target_layers=selected_layers, model_type='unet')
```

### 2. 批量分析

```python
batch_results = []
for data_file in data_files:
    # 加载数据
    input_tensor = load_spectral_data(data_file)
    
    # 分析
    results = grad_cam.generate_cam(input_tensor)
    first_layer = list(results.keys())[0]
    
    batch_results.append({
        'file': data_file,
        'peak_intensity': results[first_layer]['peak_intensity'],
        'mean_intensity': results[first_layer]['mean_intensity']
    })

# 统计分析
peak_intensities = [r['peak_intensity'] for r in batch_results]
print(f"平均峰值强度: {np.mean(peak_intensities):.4f}")
```

### 3. 模型对比

```python
models = {
    'UNET_Small': UNET(is_norm=False, in_channels=1, out_channels=1, features=[16, 32, 64]),
    'UNET_Large': UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128, 256])
}

comparison_results = {}
for model_name, model in models.items():
    grad_cam = SpectralGradCAM(model, model_type='unet')
    results = grad_cam.generate_cam(input_tensor)
    
    first_layer = list(results.keys())[0]
    comparison_results[model_name] = results[first_layer]['peak_intensity']

# 找出最佳模型
best_model = max(comparison_results.keys(), key=lambda x: comparison_results[x])
print(f"最佳模型: {best_model}")
```

## 数据格式要求

### 输入数据格式

- **张量形状**: `(batch_size, channels, height, width)`
- **数据类型**: `torch.float32`
- **数值范围**: 建议归一化到 [0, 1] 或标准化

### Excel文件格式

对于真实数据测试，Excel文件应满足：
- 第一行为标题行（会被跳过）
- 第一列为索引列（会被排除）
- 数据部分为纯数值矩阵

## 性能优化建议

### 1. GPU加速

```python
# 启用CUDA加速
grad_cam = SpectralGradCAM(model, model_type='unet', use_cuda=True)
```

### 2. 批处理优化

```python
# 对于大量样本，建议分批处理
batch_size = 10
for i in range(0, len(data_files), batch_size):
    batch_files = data_files[i:i+batch_size]
    # 处理批次
```

### 3. 内存管理

```python
# 及时释放不需要的变量
del grad_cam
torch.cuda.empty_cache()  # 如果使用GPU
```

## 故障排除

### 常见错误及解决方案

1. **维度错误**: 确保输入张量是4D格式
   ```python
   # 错误: 3D张量
   input_tensor = torch.randn(63, 21)
   
   # 正确: 4D张量
   input_tensor = torch.randn(1, 1, 63, 21)
   ```

2. **模型参数错误**: 检查模型构造函数参数
   ```python
   # 确保包含所有必需参数
   model = UNET(is_norm=False, in_channels=1, out_channels=1, features=[32, 64, 128])
   ```

3. **CUDA内存不足**: 减少批次大小或使用CPU
   ```python
   grad_cam = SpectralGradCAM(model, model_type='unet', use_cuda=False)
   ```

## 扩展开发

### 添加新模型支持

1. 在 `get_model_target_layers` 函数中添加新模型类型
2. 实现模型特定的层检测逻辑
3. 测试新模型的CAM生成效果

### 自定义分析指标

```python
def custom_analysis(cam, input_tensor):
    """自定义CAM分析函数"""
    # 实现自定义分析逻辑
    custom_metric = torch.sum(cam * some_weight_matrix)
    return {'custom_metric': custom_metric.item()}

# 在结果中添加自定义分析
results = grad_cam.generate_cam(input_tensor)
for layer_name, result in results.items():
    custom_result = custom_analysis(result['cam'], input_tensor)
    result.update(custom_result)
```

## 参考资料

- [Grad-CAM原论文](https://arxiv.org/abs/1610.02391)
- [PyTorch官方文档](https://pytorch.org/docs/)
- 项目相关模型文档: `src/regression/models/`

## 更新日志

- **v1.0**: 初始版本，支持基础Grad-CAM功能
- **v1.1**: 添加光谱特化分析
- **v1.2**: 支持真实数据测试和批量处理
- **v1.3**: 优化性能和内存使用

## 联系方式

如有问题或建议，请通过项目Issue或Pull Request联系开发团队。