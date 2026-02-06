# TsyFSpectrumClassify 项目重构方案

## 1. 优化后的目录结构

```
TsyFSpectrumClassify_remote/
├── data/                           # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 预处理后的数据
│   ├── augmented/                 # 数据增强后的数据
│   └── results/                   # 结果数据
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── classification/            # 分类相关代码
│   │   ├── __init__.py
│   │   ├── models/               # 分类模型
│   │   │   ├── __init__.py
│   │   │   ├── cnn_2d_v1.py     # 2D CNN模型 v1
│   │   │   ├── cnn_2d_v2.py     # 2D CNN模型 v2
│   │   │   ├── knn_v1.py        # KNN模型 v1
│   │   │   ├── knn_v2.py        # KNN模型 v2
│   │   │   ├── lstm_v1.py       # LSTM模型 v1
│   │   │   ├── random_forest_v1.py # 随机森林模型 v1
│   │   │   ├── transformer_v1.py   # Transformer模型 v1
│   │   │   ├── transformer_v2.py   # Transformer模型 v2
│   │   │   ├── simple_cnn.py       # 简单CNN
│   │   │   ├── simple_lstm.py      # 简单LSTM
│   │   │   ├── simple_transformer.py # 简单Transformer
│   │   │   ├── gate_network.py     # 门控网络
│   │   │   ├── vote_model.py       # 投票模型
│   │   │   └── demo/              # 模型演示
│   │   │       ├── __init__.py
│   │   │       ├── cnn_2d_demo.py
│   │   │       ├── knn_demo.py
│   │   │       └── moe_demo.py
│   │   └── utils/                # 分类工具类
│   │       ├── __init__.py
│   │       ├── image_dataset.py  # 图像数据集处理
│   │       ├── category_generator.py # 类别生成器
│   │       ├── plot_utils.py     # 绘图工具
│   │       └── matrix_reader.py  # 矩阵读取工具
│   ├── regression/               # 回归相关代码
│   │   ├── __init__.py
│   │   ├── models/              # 回归模型
│   │   │   ├── __init__.py
│   │   │   ├── dual_simple_cnn.py
│   │   │   ├── dual_unet.py
│   │   │   ├── dual_unet_co_encoder.py
│   │   │   ├── fvgg11.py
│   │   │   ├── resnet18.py
│   │   │   └── unet.py
│   │   ├── training/            # 训练相关
│   │   │   ├── __init__.py
│   │   │   ├── custom_dataset.py
│   │   │   ├── train_model.py
│   │   │   └── test_model.py
│   │   └── utils/               # 回归工具类
│   │       ├── __init__.py
│   │       └── regression_utils.py
│   ├── preprocessing/           # 数据预处理
│   │   ├── __init__.py
│   │   ├── normalization.py     # 标准化处理
│   │   ├── noise_addition.py    # 噪声添加
│   │   └── data_augmentation.py # 数据增强
│   ├── augmentation/           # 数据增强
│   │   ├── __init__.py
│   │   ├── gmm.py             # GMM增强
│   │   ├── mixup.py           # MixUp增强
│   │   ├── vae.py             # VAE增强
│   │   └── contour_drawer.py  # 轮廓绘制
│   ├── utils/                 # 通用工具
│   │   ├── __init__.py
│   │   ├── data_io/          # 数据输入输出
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py    # 数据加载
│   │   │   ├── mat_reader.py     # MAT文件读取
│   │   │   ├── npz_reader.py     # NPZ文件读取
│   │   │   ├── matrix_reader.py  # 矩阵读取
│   │   │   └── json_generator.py # JSON生成
│   │   ├── visualization/    # 可视化工具
│   │   │   ├── __init__.py
│   │   │   ├── spectrum_plotter.py  # 光谱绘制
│   │   │   ├── contour_plotter.py   # 轮廓绘制
│   │   │   ├── radar_plotter.py     # 雷达图绘制
│   │   │   ├── label_drawer.py      # 标签绘制
│   │   │   └── result_plotter.py    # 结果绘制
│   │   ├── metrics/          # 评估指标
│   │   │   ├── __init__.py
│   │   │   ├── pearson_calculator.py    # 皮尔逊相关系数
│   │   │   ├── similarity_calculator.py # 相似度计算
│   │   │   ├── cosine_similarity.py     # 余弦相似度
│   │   │   └── relative_error.py        # 相对误差
│   │   └── file_operations/  # 文件操作
│   │       ├── __init__.py
│   │       ├── batch_resizer.py     # 批量调整大小
│   │       ├── file_merger.py       # 文件合并
│   │       ├── format_converter.py  # 格式转换
│   │       └── name_processor.py    # 文件名处理
│   └── ui/                   # 用户界面
│       ├── __init__.py
│       └── contour_ui.py     # 轮廓绘制UI
├── notebooks/               # Jupyter notebooks
│   ├── exploration/        # 数据探索
│   ├── experiments/        # 实验记录
│   └── demos/             # 演示notebook
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_classification/
│   ├── test_regression/
│   └── test_utils/
├── configs/               # 配置文件
│   ├── model_configs/    # 模型配置
│   ├── data_configs/     # 数据配置
│   └── training_configs/ # 训练配置
├── scripts/              # 脚本文件
│   ├── run_training.py   # 训练脚本
│   ├── batch_test.py     # 批量测试
│   ├── demo.py          # 演示脚本
│   └── data_extraction.py # 数据提取
├── docs/                # 文档
│   ├── README.md
│   ├── API.md
│   └── CHANGELOG.md
├── requirements.txt     # 依赖包
├── setup.py            # 安装脚本
└── .gitignore         # Git忽略文件
```

## 2. 文件命名规范

### Python文件命名
- 使用小写字母和下划线：`my_module.py`
- 类名使用驼峰命名：`class MyClass`
- 函数名使用小写和下划线：`def my_function()`
- 常量使用大写和下划线：`MY_CONSTANT = 10`

### 目录命名
- 使用小写字母和下划线
- 避免使用空格和特殊字符
- 使用复数形式表示包含多个项目的目录：`models/`, `utils/`

## 3. 重构步骤

### 步骤1：创建备份
```bash
# 创建代码备份（排除数据集）
rsync -av --exclude='dataset/' --exclude='dataset_classify/' \
         --exclude='.venv/' --exclude='__pycache__/' \
         --exclude='*.pyc' --exclude='logs/' \
         --exclude='results/' --exclude='dataset_result/' \
         . ../TsyFSpectrumClassify_code_backup/
```

### 步骤2：执行重构脚本
```bash
python restructure_project.py
```

### 步骤3：更新导入路径
- 更新所有Python文件中的import语句
- 使用相对导入或绝对导入
- 确保所有模块路径正确

### 步骤4：测试重构结果
- 运行现有的测试用例
- 检查主要功能是否正常工作
- 验证数据加载和模型训练

## 4. 重构后的优势

1. **清晰的模块分离**：分类和回归功能完全分离
2. **统一的命名规范**：所有文件和目录遵循Python标准
3. **更好的可维护性**：相关功能集中在同一目录
4. **便于扩展**：新功能可以轻松添加到对应模块
5. **减少重复代码**：通用工具集中管理

## 5. 注意事项

1. **备份重要**：重构前务必创建完整备份
2. **逐步测试**：每完成一个模块的重构就进行测试
3. **更新文档**：重构后更新相关文档和README
4. **依赖管理**：检查并更新requirements.txt
5. **版本控制**：使用Git跟踪重构过程