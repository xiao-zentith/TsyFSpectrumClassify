import subprocess
import os
import sys
from itertools import product

def main():
    configs = ["C6_FITC", "C6_HPTS", "FITC_HPTS"]
    models = ["DualSimpleCNN", "VGG11", "ResNet18", "DualUNet", "DualUNetSharedEncoder"]
    is_norm_values = [True, False]

    for config, model, is_norm in product(configs, models, is_norm_values):
        output_dir = f"logs/{config}_{model}_norm_{is_norm}"
        os.makedirs(output_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, "training.log")


        with open(log_file_path, "w") as log_file:
            print(f"Starting training for config={config}, model={model}, is_norm={is_norm}")
            try:
                result = subprocess.run(
                    ["/home/asus515/anaconda3/envs/EEM/bin/python", "run_training.py",
                     "--config", config,
                     "--model", model,
                     "--is_norm", str(is_norm)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                log_file.write(result.stdout)
                print(f"Training completed for config={config}, model={model}, is_norm={is_norm}. Log saved to {log_file_path}")
            except subprocess.CalledProcessError as e:
                log_file.write(e.output)
                print(f"Training failed for config={config}, model={model}, is_norm={is_norm}. Error logged to {log_file_path}")

if __name__ == "__main__":
    main()



