# Grad-CAM 可解释性模块

本模块为光谱数据分析模型提供Grad-CAM（Gradient-weighted Class Activation Mapping）可解释性功能。

## 功能特性

- 🔍 **多模型支持**: 支持UNet、DualUNet、CNN等模型架构
- 📊 **1D/2D光谱数据**: 同时支持一维光谱和二维高光谱图像数据
- 🎯 **自动层检测**: 自动识别模型中适合进行Grad-CAM分析的层
- 📈 **可视化工具**: 提供丰富的可视化功能，包括热力图、重要性排序等
- 🔬 **光谱分析**: 专门针对光谱数据的波长重要性分析
- 📦 **批量处理**: 支持批量样本分析和模型对比

## 模块结构

```
src/utils/interpretability/
├── __init__.py              # 模块初始化
├── gradcam.py              # 核心Grad-CAM算法实现
├── visualization.py        # 可视化工具
├── model_wrapper.py        # 模型集成包装器
└── README.md              # 使用说明（本文件）
```

## 快速开始

### 1. 基本使用

```python
from src.utils.interpretability import SpectralGradCAM, GradCAMVisualizer
from src.regression.models.unet import UNET
import numpy as np

# 加载模型
model = UNET(is_norm=True, in_channels=1, out_channels=2)
model.load_state_dict(torch.load('your_model.pth'))

# 创建Grad-CAM分析器
gradcam = SpectralGradCAM(model, model_type='unet')

# 准备光谱数据
spectrum = np.random.randn(1024)  # 1024个波长点
wavelengths = np.linspace(400, 800, 1024)  # 400-800 nm

# 生成CAM
cam_results = gradcam.generate_cam(spectrum, target_output_idx=0)

# 可视化结果
visualizer = GradCAMVisualizer()
visualizer.plot_1d_gradcam(spectrum, cam_results, wavelengths)
```

### 2. 使用模型包装器

```python
from src.utils.interpretability import SpectralModelAnalyzer

# 创建分析器
analyzer = SpectralModelAnalyzer(model, model_type='unet')

# 分析单个样本
results = analyzer.analyze_sample(
    spectrum, 
    target_output_idx=0,
    wavelengths=wavelengths
)

# 可视化分析结果
analyzer.visualize_analysis(spectrum, results, wavelengths=wavelengths)
```

### 3. 批量分析

```python
# 批量数据
batch_data = np.random.randn(10, 1024)  # 10个样本
sample_names = [f'Sample_{i}' for i in range(10)]

# 批量分析
batch_results = analyzer.batch_analysis(
    batch_data,
    sample_names=sample_names,
    wavelengths=wavelengths
)
```

## 详细功能说明

### GradCAM 类

核心Grad-CAM实现，支持通用PyTorch模型。

**主要方法:**
- `generate_cam()`: 生成类激活映射
- `get_target_layers()`: 获取目标层
- `resize_cam()`: 调整CAM尺寸

### SpectralGradCAM 类

专门为光谱数据优化的Grad-CAM实现。

**主要方法:**
- `generate_cam()`: 生成光谱数据的CAM
- `get_important_wavelengths()`: 获取重要波长
- `analyze_spectral_regions()`: 分析光谱区域
- `rank_layer_importance()`: 层重要性排序

### GradCAMVisualizer 类

提供丰富的可视化功能。

**主要方法:**
- `plot_1d_gradcam()`: 一维光谱Grad-CAM可视化
- `plot_2d_gradcam()`: 二维图像Grad-CAM可视化
- `compare_models()`: 模型对比可视化
- `plot_layer_importance()`: 层重要性可视化
- `generate_summary_report()`: 生成分析报告

### SpectralModelAnalyzer 类

高级模型分析包装器，整合所有功能。

**主要方法:**
- `analyze_sample()`: 单样本分析
- `batch_analysis()`: 批量分析
- `compare_models()`: 模型对比
- `visualize_analysis()`: 可视化分析结果

## 支持的模型类型

| 模型类型 | 类名 | 说明 |
|---------|------|------|
| UNet | `UNET` | U-Net架构，适用于光谱回归 |
| DualUNet | `DualUNet` | 双分支U-Net |
| DualUNetSharedEncoder | `DualUNetSharedEncoder` | 共享编码器的双U-Net |
| CNN | `DualSimpleCNN` | 双通道简单CNN |
| ResNet | `ResNet*` | ResNet系列模型 |
| VGG | `VGG*` | VGG系列模型 |

## 使用示例

### 示例1: 训练后模型分析

```python
# 加载训练好的模型
model = torch.load('trained_model.pth')
analyzer = SpectralModelAnalyzer(model, model_type='unet')

# 加载测试数据
test_data = np.load('test_spectrum.npy')
wavelengths = np.load('wavelengths.npy')

# 分析
results = analyzer.analyze_sample(test_data, wavelengths=wavelengths)

# 查看重要波长
for layer_name, cam_data in results['cam_results'].items():
    if 'important_wavelengths' in cam_data:
        print(f"{layer_name}: {cam_data['important_wavelengths'][:5]}")
```

### 示例2: 模型对比

```python
models = {
    'UNet': unet_model,
    'DualUNet': dual_unet_model,
    'CNN': cnn_model
}

# 对比分析
comparison_results = {}
for name, model in models.items():
    analyzer = SpectralModelAnalyzer(model, model_type=name.lower())
    results = analyzer.analyze_sample(test_spectrum)
    comparison_results[name] = results

# 可视化对比
visualizer = GradCAMVisualizer()
visualizer.compare_models(test_spectrum, comparison_results, wavelengths)
```

### 示例3: 集成到训练流程

```python
def validate_with_interpretability(model, val_loader, epoch):
    """在验证过程中添加可解释性分析"""
    analyzer = SpectralModelAnalyzer(model, model_type='unet')
    
    # 选择几个样本进行分析
    for i, (data, target) in enumerate(val_loader):
        if i >= 3:  # 只分析前3个batch
            break
            
        sample = data[0].numpy()  # 取第一个样本
        results = analyzer.analyze_sample(sample)
        
        # 保存分析结果
        save_path = f'interpretability/epoch_{epoch}_sample_{i}'
        analyzer.visualize_analysis(
            sample, results, 
            save_dir=save_path,
            sample_name=f'Epoch{epoch}_Sample{i}'
        )
```

## 命令行工具

使用提供的演示脚本：

```bash
# 基本使用
python scripts/interpretability/gradcam_demo.py \
    --model_path models/trained_unet.pth \
    --data_path data/test_spectrum.npz \
    --output_dir results/gradcam

# 使用配置文件
python scripts/interpretability/gradcam_demo.py \
    --config_path config/model_config.json \
    --model_path models/checkpoint.pth \
    --data_path data/validation_data.npz \
    --save_figures

# 分析特定层
python scripts/interpretability/gradcam_demo.py \
    --model_path models/model.pth \
    --data_path data/sample.npz \
    --layers encoder.conv1 decoder.up1 \
    --target_output 1
```

## 输出说明

### CAM结果结构

```python
cam_results = {
    'layer_name': {
        'cam': np.ndarray,              # CAM数组
        'cam_shape': tuple,             # CAM形状
        'peak_intensity': float,        # 峰值强度
        'mean_intensity': float,        # 平均强度
        'important_wavelengths': list,  # 重要波长（1D数据）
        'dominant_wavelength': float,   # 主导波长（1D数据）
        'spectral_regions': dict,       # 光谱区域分析
    }
}
```

### 分析结果结构

```python
analysis_results = {
    'cam_results': dict,           # CAM结果
    'importance_ranking': list,    # 层重要性排序
    'model_info': dict,           # 模型信息
    'input_shape': tuple,         # 输入形状
    'target_output_idx': int,     # 目标输出索引
}
```

## 注意事项

1. **内存使用**: 大模型和高分辨率数据可能消耗大量内存
2. **计算时间**: Grad-CAM计算需要前向和反向传播，较为耗时
3. **模型兼容性**: 确保模型支持梯度计算（`requires_grad=True`）
4. **数据格式**: 输入数据应为numpy数组或PyTorch张量
5. **设备兼容**: 支持CPU和GPU计算，自动检测可用设备

## 故障排除

### 常见问题

1. **"No gradients found"错误**
   - 确保模型处于训练模式或支持梯度计算
   - 检查目标层是否正确

2. **内存不足**
   - 减少批量大小
   - 使用CPU而非GPU
   - 分析较少的层

3. **可视化问题**
   - 确保安装了matplotlib
   - 检查数据维度是否正确

4. **模型不兼容**
   - 检查模型类型参数
   - 确认模型架构支持

### 调试技巧

```python
# 启用调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型层
analyzer = SpectralModelAnalyzer(model, model_type='unet')
print("Available layers:", analyzer.get_model_info()['layer_names'])

# 测试单层
gradcam = SpectralGradCAM(model, target_layers=['specific_layer'])
```

## 扩展开发

### 添加新模型支持

1. 在`get_model_target_layers()`函数中添加新模型类型
2. 实现模型特定的层选择逻辑
3. 测试兼容性

### 自定义可视化

```python
class CustomVisualizer(GradCAMVisualizer):
    def custom_plot(self, data, cam_results):
        # 实现自定义可视化
        pass
```

### 性能优化

- 使用`torch.no_grad()`减少内存使用
- 实现批量CAM计算
- 添加缓存机制

## 参考文献

1. Selvaraju, R. R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." ICCV 2017.
2. Chattopadhay, A., et al. "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks." WACV 2018.