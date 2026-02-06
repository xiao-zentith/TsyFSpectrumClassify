# Data Directory - 数据目录

这个目录包含了项目的所有数据集和相关文件。

## 目录结构

### 数据集目录
- `dataset/` - 主要数据集目录
  - `dataset_raw/` - 原始数据
  - `dataset_resized/` - 调整大小后的数据
  - `dataset_preprocess/` - 预处理后的数据
  - `dataset_target/` - 目标数据
- `dataset_classify/` - 分类数据集目录
  - `dataset_raw/` - 原始分类数据
  - `dataset_preprocess/` - 预处理后的分类数据
  - `dataset_noise/` - 噪声数据
  - `dataset_preprocess2/` - 二次预处理数据

### 数据文件
- `dataset.zip` - 数据集压缩包

## 符号链接

为了保持与现有代码的兼容性，在项目根目录创建了以下符号链接：
- `dataset` → `data/dataset`
- `dataset_classify` → `data/dataset_classify`

## 配置文件

相关的配置文件已移动到 `configs/` 目录：
- `configs/config.json` - 主配置文件
- `configs/dataset_classify_config.json` - 分类数据集配置
- `configs/dataset_info.json` - 数据集信息

## 整理时间
2025-10-30 15:40:20

## 注意事项
- 数据集目录已从根目录移动到此处以保持项目结构清晰
- 通过符号链接保持了向后兼容性
- 配置文件中的路径已相应更新
