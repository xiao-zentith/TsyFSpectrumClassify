# 🚀 项目重构快速开始

## 一键执行（推荐）

```bash
python run_restructure.py
```

这个命令会自动执行：
1. ✅ 创建代码备份
2. ✅ 重构项目结构  
3. ✅ 更新导入路径
4. ✅ 验证重构结果

## 分步执行

如果需要分步控制，可以单独运行：

### 1. 完整重构
```bash
python complete_restructure.py
```

### 2. 更新导入
```bash
python update_imports.py
```

### 3. 验证结果
```bash
python test_restructure.py
```

## 重要文件

| 文件 | 说明 |
|------|------|
| `run_restructure.py` | 一键执行脚本 |
| `complete_restructure.py` | 完整重构脚本 |
| `update_imports.py` | 导入路径更新 |
| `test_restructure.py` | 重构验证测试 |
| `RESTRUCTURE_GUIDE.md` | 详细执行指南 |

## 生成的报告

执行后会生成以下报告文件：
- `restructure.log` - 重构操作日志
- `import_update_report.txt` - 导入更新报告  
- `test_report.txt` - 验证测试报告

## 新项目结构预览

```
src/
├── utils/                 # 通用工具
│   ├── data_io/          # 数据IO
│   ├── visualization/    # 可视化
│   ├── metrics/          # 评估指标
│   └── file_operations/  # 文件操作
├── classification/       # 分类模块
│   ├── models/          # 分类模型
│   └── utils/           # 分类工具
├── regression/          # 回归模块
│   ├── models/          # 回归模型
│   ├── training/        # 训练脚本
│   └── utils/           # 回归工具
├── augmentation/        # 数据增强
├── preprocessing/       # 数据预处理
└── ui/                  # 用户界面
```

## ⚠️ 注意事项

1. **备份**：脚本会自动创建备份，但建议手动备份重要文件
2. **分支**：建议在 `refactor` 分支上执行
3. **依赖**：重构后需要更新相关配置和文档

## 🆘 遇到问题？

1. 查看生成的日志文件
2. 参考 `RESTRUCTURE_GUIDE.md` 详细指南
3. 检查错误信息并手动调整

---
**准备好了吗？运行 `python run_restructure.py` 开始重构！** 🎯