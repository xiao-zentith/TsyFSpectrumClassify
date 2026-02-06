import subprocess
import os
import sys
from itertools import product

def main():
    configs = ["C6_FITC", "C6_HPTS", "FITC_HPTS", "ALL"]
    # configs = ["Fish"]
    # configs = ["ALL"]
    # models = ["DualSimpleCNN", "VGG11", "ResNet18", "DualUNet", "DualUNetSharedEncoder"]
    # models = ["DualUNet", "DualUNetSharedEncoder"]
    models = ["DualSimpleCNN", "VGG11", "ResNet18"]
    is_norm_values = [True, False]
    # is_norm_values = [False]
    # loss_types = ["rmse", "mae", "rmse + mae"]
    loss_types = ["rmse + mae"]

    for config, model, is_norm, loss_type in product(configs, models, is_norm_values, loss_types):
        output_dir = f"logs/{config}_{model}_norm_{is_norm}_loss_{loss_type}"
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, "training.log")


        with open(log_file_path, "w") as log_file:
            print(f"Starting training for config={config}, model={model}, is_norm={is_norm}, loss_type={loss_type}")
            try:
                result = subprocess.run(
                    ["/home/asus515/anaconda3/envs/EEM/bin/python", "run_training.py",
                     "--config", config,
                     "--model", model,
                     "--loss_type", loss_type,
                     "--fold_number", str(10),
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



