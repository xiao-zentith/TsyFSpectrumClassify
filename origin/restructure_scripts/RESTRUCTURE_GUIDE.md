# 项目重构执行指南

## 概述
本指南提供了完整的项目重构方案，包括目录结构优化、文件命名规范化、导入路径更新等。

## 重构目标
- 🗂️ 优化项目目录结构
- 📝 统一文件命名规范
- 🔗 更新导入路径
- 🧪 分离分类和回归功能
- 📦 创建标准Python包结构

## 新的项目结构
```
TsyFSpectrumClassify/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   ├── augmented/                 # 增强数据
│   └── results/                   # 结果数据
├── src/                           # 源代码
│   ├── utils/                     # 通用工具
│   │   ├── data_io/              # 数据输入输出
│   │   ├── visualization/        # 可视化工具
│   │   ├── metrics/              # 评估指标
│   │   └── file_operations/      # 文件操作
│   ├── classification/           # 分类模块
│   │   ├── models/              # 分类模型
│   │   │   └── demo/            # 演示模型
│   │   └── utils/               # 分类工具
│   ├── regression/              # 回归模块
│   │   ├── models/              # 回归模型
│   │   ├── training/            # 训练脚本
│   │   └── utils/               # 回归工具
│   ├── augmentation/            # 数据增强
│   ├── preprocessing/           # 数据预处理
│   └── ui/                      # 用户界面
├── notebooks/                    # Jupyter notebooks
│   ├── exploration/             # 数据探索
│   ├── experiments/             # 实验记录
│   └── demos/                   # 演示notebook
├── tests/                        # 测试代码
├── configs/                      # 配置文件
├── scripts/                      # 脚本文件
└── docs/                         # 文档
```

## 执行步骤

### 第一步：准备工作
1. **确保在正确的分支**
   ```bash
   git checkout refactor  # 或创建新分支
   ```

2. **检查当前状态**
   ```bash
   git status
   ```

### 第二步：执行重构
运行完整重构脚本：
```bash
python complete_restructure.py
```

这个脚本会：
- ✅ 创建代码备份（排除数据集）
- ✅ 创建新的目录结构
- ✅ 移动和重命名文件
- ✅ 创建 `__init__.py` 文件
- ✅ 生成项目配置文件（requirements.txt, .gitignore, README.md）

### 第三步：更新导入路径
运行导入更新脚本：
```bash
python update_imports.py
```

这个脚本会：
- 🔄 扫描所有Python文件
- 🔄 更新import语句
- 📝 生成更新报告

### 第四步：验证重构结果
运行测试脚本：
```bash
python test_restructure.py
```

这个脚本会验证：
- 📁 目录结构完整性
- 📄 关键文件存在性
- 🔗 导入路径正确性
- 🐍 Python语法正确性

### 第五步：手动检查和调整
1. **检查生成的文件**
   - `restructure.log` - 重构日志
   - `import_update_report.txt` - 导入更新报告
   - `test_report.txt` - 测试报告

2. **手动调整（如需要）**
   - 检查特殊的导入语句
   - 调整相对导入路径
   - 更新配置文件中的路径

3. **运行功能测试**
   ```bash
   # 测试关键功能
   python -c "from src.utils.data_io.mat_reader import *"
   python -c "from src.classification.models.cnn_2d_v1 import *"
   ```

## 文件命名规范

### Python文件
- 使用小写字母和下划线：`data_loader.py`
- 避免数字开头：`cnn_2d_v1.py`（而不是 `2D_CNN1.py`）

### 类名
- 使用驼峰命名：`DataLoader`, `CNNModel`

### 函数和变量
- 使用小写字母和下划线：`load_data()`, `model_config`

### 常量
- 使用大写字母和下划线：`MAX_EPOCHS`, `DEFAULT_BATCH_SIZE`

## 导入路径映射

### 主要映射关系
```python
# 旧路径 -> 新路径
from Utils.read_mat import -> from src.utils.data_io.mat_reader import
from Utils.draw_2D_spectrum import -> from src.utils.visualization.spectrum_plotter import
from classfication.Utils.ImageDataset import -> from src.classification.utils.image_dataset import
from regression.model.UNet import -> from src.regression.models.unet import
```

## 注意事项

### ⚠️ 重要提醒
1. **备份重要**：脚本会自动创建备份，但建议额外手动备份
2. **分支管理**：在专门的分支上进行重构
3. **逐步验证**：每个步骤完成后都要验证结果
4. **依赖检查**：重构后检查所有依赖是否正常

### 🔧 故障排除
1. **导入错误**：检查 `import_update_report.txt`
2. **文件缺失**：检查 `restructure.log`
3. **语法错误**：检查 `test_report.txt`

### 📝 后续工作
1. **更新文档**：更新项目文档中的路径引用
2. **配置文件**：更新IDE配置、部署脚本等
3. **团队同步**：通知团队成员新的项目结构

## 回滚方案
如果重构出现问题，可以：
1. 从备份目录恢复文件
2. 使用Git回滚到重构前的提交
3. 手动恢复关键文件

## 联系支持
如果在重构过程中遇到问题，请：
1. 查看生成的日志文件
2. 检查错误信息
3. 参考本指南的故障排除部分

---
**重构完成后，请删除旧的脚本文件和临时文件，保持项目整洁。**