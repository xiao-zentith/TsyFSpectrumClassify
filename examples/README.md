# 示例脚本说明

本目录包含了项目的各种示例脚本，帮助用户快速上手和理解项目功能。

## Grad-CAM 可解释性分析示例

### 📁 gradcam_usage_examples.py
**基础演示脚本** - 展示Grad-CAM功能的各种使用场景

**包含示例:**
- ✅ 基础Grad-CAM分析
- ✅ 模型对比分析  
- ✅ 批量数据处理
- ✅ 2D光谱分析
- ✅ 层重要性分析
- ✅ 训练集成示例

**运行方式:**
```bash
python examples/gradcam_usage_examples.py
```

### 📁 gradcam_real_data_test.py
**真实数据测试脚本** - 使用项目中的真实光谱数据进行测试

**包含示例:**
- 🔬 真实数据基础分析
- 📊 多模型性能对比
- 📈 批量样本分析统计
- 🎯 层级重要性排序

**运行方式:**
```bash
python examples/gradcam_real_data_test.py
```

**数据要求:**
- 使用FITC_HPTS数据集
- Excel格式光谱数据
- 自动处理数据维度

### 📁 path_manager_usage.py
**路径管理示例** - 展示项目路径管理工具的使用

## 快速开始

### 1. 运行基础示例
```bash
# 进入项目根目录
cd /home/asus515/PycharmProjects/TsyFSpectrumClassify_remote

# 运行基础Grad-CAM演示
python examples/gradcam_usage_examples.py
```

### 2. 测试真实数据
```bash
# 运行真实数据测试（需要数据集）
python examples/gradcam_real_data_test.py
```

## 输出示例

### 基础分析输出
```
=== 示例1: 基础Grad-CAM分析 ===
分析完成，共分析了 8 个层
层 downs.0.double_conv.0:
  - CAM形状: torch.Size([32, 16])
  - 峰值强度: 0.8234
  - 平均强度: 0.0456
```

### 真实数据分析输出
```
=== 示例1: 真实数据基础分析 ===
真实数据形状: (63, 21)
输入张量形状: torch.Size([1, 1, 63, 21])
分析完成，共分析了 16 个层
层 downs.0.double_conv.0:
  - CAM形状: torch.Size([1, 32, 32, 11])
  - 峰值强度: 0.7234
  - 平均强度: 0.0623
```

### 批量分析统计
```
批量分析统计:
  - 成功分析样本数: 5
  - 平均峰值强度: 0.7357
  - 峰值强度标准差: 0.3687
  - 最高峰值强度样本: F2 + H8_extracted.xlsx
```

## 功能特性

### ✨ 核心功能
- **多模型支持**: UNET、DualUNet等
- **光谱优化**: 专门针对光谱数据设计
- **批量处理**: 支持大规模数据分析
- **实时分析**: 快速生成可解释性结果

### 🔧 技术特点
- **自动层检测**: 智能识别模型关键层
- **GPU加速**: 支持CUDA加速计算
- **内存优化**: 高效的内存使用策略
- **错误处理**: 完善的异常处理机制

## 依赖要求

```python
torch >= 1.8.0
pandas >= 1.3.0
numpy >= 1.21.0
openpyxl >= 3.0.0  # Excel文件读取
```

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在项目根目录运行
   cd /home/asus515/PycharmProjects/TsyFSpectrumClassify_remote
   python examples/gradcam_usage_examples.py
   ```

2. **数据维度错误**
   - 确保输入数据是4D张量: `(batch, channel, height, width)`
   - 使用 `.unsqueeze()` 添加缺失维度

3. **内存不足**
   ```python
   # 使用CPU模式
   grad_cam = SpectralGradCAM(model, model_type='unet', use_cuda=False)
   ```

4. **Excel文件读取错误**
   - 检查文件路径是否正确
   - 确保Excel文件格式符合要求

## 更多信息

- 📖 **详细文档**: 查看 `docs/GRADCAM_USAGE_GUIDE.md`
- 🔬 **核心代码**: `src/utils/interpretability/gradcam.py`
- 🏗️ **模型定义**: `src/regression/models/`

## 贡献指南

欢迎提交Issue和Pull Request来改进示例脚本！

### 添加新示例
1. 在 `examples/` 目录创建新脚本
2. 遵循现有代码风格
3. 添加详细注释和错误处理
4. 更新此README文件