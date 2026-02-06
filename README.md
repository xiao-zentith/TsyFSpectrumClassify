# TsyF Spectrum Classification Project

## 项目结构

```
├── src/                    # 源代码
│   ├── utils/             # 工具函数
│   ├── classification/    # 分类模块
│   ├── regression/        # 回归模块
│   ├── augmentation/      # 数据增强
│   ├── preprocessing/     # 数据预处理
│   └── ui/               # 用户界面
├── notebooks/             # Jupyter notebooks
├── tests/                # 测试代码
├── scripts/              # 脚本文件
├── configs/              # 配置文件
└── docs/                 # 文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

1. 数据预处理：使用 `src/preprocessing/` 中的脚本
2. 模型训练：使用 `scripts/training/` 中的脚本
3. 模型评估：使用 `scripts/evaluation/` 中的脚本

## 重构说明

本项目已完成安全重构，主要改进：
- 模块化的代码组织
- 标准化的文件命名
- 清晰的目录结构
- 完整的测试框架
- 避免递归备份问题
