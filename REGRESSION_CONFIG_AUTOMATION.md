# Regression Configuration Automation System

这个自动化系统可以自动生成回归任务所需的配置文件和数据集信息文件，无需手动配置。

## 🚀 快速开始

### 方法1: 使用便捷脚本（推荐）

```bash
# 生成所有配置文件（config + dataset_info）
./generate_regression_configs.sh

# 或者使用具体命令
./generate_regression_configs.sh all          # 生成所有文件
./generate_regression_configs.sh config       # 只生成config文件
./generate_regression_configs.sh dataset-info # 只生成dataset_info文件
./generate_regression_configs.sh clean        # 清理并重新生成所有文件
./generate_regression_configs.sh help         # 显示帮助信息
```

### 方法2: 使用Python脚本

```bash
# 生成所有配置文件
python regression_automation_pipeline.py

# 只生成config文件
python regression_automation_pipeline.py --config-only

# 只生成dataset_info文件
python regression_automation_pipeline.py --dataset-info-only

# 清理并重新生成所有文件
python regression_automation_pipeline.py --clean

# 显示帮助
python regression_automation_pipeline.py --help
```

## 📁 系统组成

### 核心脚本

1. **<mcfile name="regression_config_generator.py" path="/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression_config_generator.py"></mcfile>**
   - 自动发现数据集文件夹
   - 检测每个数据集的组件数量
   - 生成 `regression_config_xxx.json` 文件

2. **<mcfile name="regression_dataset_info_generator.py" path="/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression_dataset_info_generator.py"></mcfile>**
   - 基于config文件和预处理数据集结构
   - 使用预处理数据集 (`dataset_preprocess`) 而不是原始数据集 (`dataset_raw`)
   - 创建交叉验证数据分割
   - 生成 `regression_dataset_info_xxx.json` 文件

3. **<mcfile name="regression_automation_pipeline.py" path="/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/regression_automation_pipeline.py"></mcfile>**
   - 整合上述两个生成器
   - 提供完整的自动化管道
   - 包含验证和错误处理

4. **<mcfile name="generate_regression_configs.sh" path="/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/generate_regression_configs.sh"></mcfile>**
   - 便捷的Shell脚本包装器
   - 提供彩色输出和友好的命令行界面

## 🔧 工作原理

### 自动发现机制

系统会自动扫描以下目录结构：

### 数据集信息生成流程
1. **读取配置**: 从 `regression_config_xxx.json` 文件中提取数据集名称
2. **扫描预处理数据**: 在 `data/dataset/dataset_preprocess/{dataset_name}/` 目录中查找 `*_extracted.xlsx` 文件
3. **匹配目标文件**: 在 `data/dataset/dataset_target/{dataset_name}/Component{N}/` 目录中查找对应的目标文件
4. **创建交叉验证分割**: 使用 5-fold 交叉验证，每个 fold 内部再进行 80/20 训练/验证分割
5. **生成文件路径映射**: 为每个数据样本创建输入文件到目标文件的完整路径映射
6. **保存配置**: 将所有信息保存到 `regression_dataset_info_xxx.json` 文件

**重要说明**: 系统现在使用预处理数据集 (`dataset_preprocess`) 作为输入，这些文件通常以 `_extracted.xlsx` 结尾，包含经过预处理的光谱数据。

```
data/dataset/
├── dataset_raw/           # 输入数据
│   ├── C6_FITC/          # 数据集1
│   ├── C6_HPTS/          # 数据集2
│   ├── FITC_HPTS/        # 数据集3
│   └── Fish/             # 数据集4
└── dataset_target/        # 目标数据
    ├── C6_FITC/
    │   ├── Component1/    # 组件1
    │   └── Component2/    # 组件2
    ├── Fish/
    │   ├── Component1/    # 组件1
    │   ├── Component2/    # 组件2
    │   ├── Component3/    # 组件3
    │   └── Component4/    # 组件4
    └── ...
```

### 生成的文件

对于每个发现的数据集，系统会生成：

1. **Config文件**: `configs/regression/regression_config_<dataset>.json`
   - 包含数据集路径配置
   - 动态生成多个 `dataset_target` 路径
   - 应用特殊配置（如Fish数据集的特殊设置）

2. **Dataset Info文件**: `configs/regression/regression_dataset_info_<dataset>.json`
   - 包含训练/验证/测试数据分割
   - 每个样本的输入-目标文件对应关系
   - 5折交叉验证结构

## 📊 生成的文件示例

### Config文件示例 (regression_config_C6_FITC.json)

```json
{
  "dataset_raw": "/path/to/data/dataset/dataset_raw/C6_FITC",
  "dataset_target1": "/path/to/data/dataset/dataset_target/C6_FITC/Component1",
  "dataset_target2": "/path/to/data/dataset/dataset_target/C6_FITC/Component2",
  "dataset_preprocess": "/path/to/data/dataset/dataset_preprocess/C6_FITC",
  "dataset_resized": "/path/to/data/dataset/dataset_resized/C6_FITC",
  "is_cross_validation": true,
  "is_mixup": true,
  "model_save_path": "/path/to/models/regression/C6_FITC",
  "result_save_path": "/path/to/results/regression/C6_FITC"
}
```

### Dataset Info文件示例 (regression_dataset_info_C6_FITC.json)

```json
[
  {
    "fold": 0,
    "inner_fold": 0,
    "train": [
      {
        "input": "/path/to/input.xlsx",
        "targets": [
          "/path/to/target1.xlsx",
          "/path/to/target2.xlsx"
        ]
      }
    ],
    "validation": [...],
    "test": [...]
  }
]
```

## ⚙️ 配置说明

### 路径配置 (configs/paths.json)

系统使用 <mcfile name="paths.json" path="/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/paths.json"></mcfile> 中的路径配置：

```json
{
  "data": {
    "dataset": {
      "raw": "data/dataset/dataset_raw",
      "target": "data/dataset/dataset_target",
      "processed": "data/dataset/dataset_preprocess",
      "resized": "data/dataset/dataset_resized"
    }
  },
  "regression": {
    "special_configs": {
      "Fish": {
        "is_cross_validation": false,
        "is_mixup": false
      }
    }
  }
}
```

### 特殊配置

- **Fish数据集**: 应用特殊配置 `is_cross_validation: false, is_mixup: false`
- **其他数据集**: 使用默认配置 `is_cross_validation: true, is_mixup: true`

## 🔍 验证和错误处理

系统包含完整的验证机制：

1. **依赖检查**: 确保所需脚本存在
2. **路径验证**: 检查数据集路径是否存在
3. **组件检测**: 自动检测每个数据集的组件数量
4. **文件匹配**: 验证生成的config和dataset_info文件对应关系
5. **错误报告**: 详细的错误信息和警告

## 📈 使用场景

### 新增数据集

当添加新的数据集时：

1. 将数据放入相应的 `dataset_raw` 和 `dataset_target` 目录
2. 运行 `./generate_regression_configs.sh clean`
3. 系统会自动发现并生成相应的配置文件

### 更新现有数据集

当数据集结构发生变化时：

1. 运行 `./generate_regression_configs.sh clean`
2. 系统会重新扫描并更新所有配置文件

### 部分更新

如果只需要更新特定类型的文件：

```bash
# 只更新config文件
./generate_regression_configs.sh config

# 只更新dataset_info文件
./generate_regression_configs.sh dataset-info
```

## 🎯 优势

1. **完全自动化**: 无需手动编写配置文件
2. **动态适应**: 自动适应数据集结构变化
3. **错误处理**: 完善的验证和错误报告机制
4. **灵活使用**: 支持部分生成和清理重建
5. **易于扩展**: 可轻松添加新的数据集类型和配置选项

## 🚨 注意事项

1. 确保数据集目录结构符合预期格式
2. Component文件夹必须以 "Component" 开头并包含数字
3. 输入和目标文件必须是 `.xlsx` 格式
4. 运行前确保已安装 `scikit-learn` 依赖：`pip install scikit-learn`

## 🔧 故障排除

### 常见问题

1. **"No module named 'sklearn'"**
   ```bash
   pip install scikit-learn
   ```

2. **"目标路径不存在"**
   - 检查 `data/dataset/dataset_target/` 目录结构
   - 确保数据集文件夹存在且包含Component子文件夹

3. **"没有找到Component文件夹"**
   - 确保目标目录中有以 "Component" 开头的子文件夹
   - 检查文件夹命名格式（如 Component1, Component2 等）

通过这个自动化系统，您可以轻松管理和生成回归任务所需的所有配置文件，大大提高工作效率！