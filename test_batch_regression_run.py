import subprocess
import os
import sys
from itertools import product
from pathlib import Path

# 解析项目根目录，确保可导入 src
current_script_path = Path(__file__).resolve()
project_root = current_script_path
sys.path.append(str(project_root))

from src.utils.path_manager import PathManager

def main():
    # 小规模测试：只测试一个配置组合
    configs = ["C6_FITC"]  # 只测试一个数据集
    models = ["DualSimpleCNN"]  # 只测试一个模型
    is_norm_values = [True]  # 只测试一个标准化选项
    loss_types = ["rmse + mae"]  # 只测试一个损失函数

    pm = PathManager()
    logs_root = Path(pm.get_path('logs')) / 'test_training_runs'
    logs_root.mkdir(parents=True, exist_ok=True)

    training_script = Path("scripts/training/run_regression_training.py")

    for config, model, is_norm, loss_type in product(configs, models, is_norm_values, loss_types):
        output_dir = logs_root / f"{config}_{model}_norm_{is_norm}_loss_{loss_type.replace(' ', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = output_dir / "training.log"

        with open(log_file_path, "w") as log_file:
            print(f"Starting training for config={config}, model={model}, is_norm={is_norm}, loss_type={loss_type}")
            try:
                result = subprocess.run(
                    [sys.executable, str(training_script),
                     "--config", config,
                     "--model", model,
                     "--loss_type", loss_type,
                     "--fold_number", str(2),  # 减少fold数量以加快测试
                     "--is_norm", str(is_norm)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                log_file.write(result.stdout)
                print(f"Training completed for config={config}, model={model}, is_norm={is_norm}, loss_type={loss_type}. Log saved to {log_file_path}")
            except subprocess.CalledProcessError as e:
                log_file.write(e.output)
                print(f"Training failed for config={config}, model={model}, is_norm={is_norm}, loss_type={loss_type}. Error logged to {log_file_path}")

if __name__ == "__main__":
    main()