# Origin Files - 原始文件备份

这个目录包含了项目重构前的原始文件和目录结构。

## 目录说明

### 原始代码目录
- `Utils/` - 原始工具函数目录（已重构到 `src/utils/`）
- `regression/` - 原始回归模块（已重构到 `src/regression/`）
- `classfication/` - 原始分类模块（已重构到 `src/classification/`）
- `augmentation/` - 原始数据增强模块（已重构到 `src/augmentation/`）
- `preprocess/` - 原始预处理模块（已重构到 `src/preprocessing/`）
- `UI_version/` - 原始UI模块（已重构到 `src/ui/`）
- `model_demo/` - 原始模型演示（已重构到 `notebooks/demos/`）

### 原始文件
- `demo.py` - 原始演示文件（已重构到 `notebooks/exploration/demo.py`）
- `add_noise.py` - 原始噪声添加文件（已重构到 `src/preprocessing/add_noise.py`）

### 重构脚本
`restructure_scripts/` 目录包含了所有用于项目重构的脚本文件：
- 各种重构脚本（`*_restructure.py`）
- 重构指南和文档
- 重构过程中生成的报告文件

## 重构时间
2025-10-30 15:33:20

## 注意事项
- 这些文件仅作为备份保存，新的项目结构位于根目录
- 如需恢复某个文件，可以从这里复制到相应的新位置
- 数据集目录（dataset/, dataset_classify/ 等）未被移动，仍在原位置
