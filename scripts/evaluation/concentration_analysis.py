import json
import os
import re
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def parse_concentration_ratio(filename):
    """
    Parses the filename to extract the concentration ratio of C6 to FITC.
    Example filenames:
    - 'C6_3_FITC_7.xlsx' -> 3/7
    - 'FITC_7_HPTS_3' -> None (or handle as needed)
    """
    match = re.search(r'C6_(\d+)_FITC_(\d+)', Path(filename).stem)
    if match:
        c6_concentration = int(match.group(1))
        fitc_concentration = int(match.group(2))
        return c6_concentration / fitc_concentration if fitc_concentration != 0 else float('inf')
    return None


def calculate_integrated_intensity_ratio(predicted_maps):
    """
    Calculates the ratio of integrated intensities from the predicted component maps.
    """
    # Sum over the spatial dimensions (height and width)
    integrated_intensities = np.sum(predicted_maps, axis=(2, 3))
    # Ratio of component 1 (C6) to component 2 (FITC)
    # Add a small epsilon to avoid division by zero
    ratios = integrated_intensities[:, 0] / (integrated_intensities[:, 1] + 1e-9)
    return ratios


def main():
    # 1. Define paths and parameters
    dataset_info_path = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/regression_dataset_info_C6_FITC.json"
    results_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/regression/DualUNetSharedEncoder/training_20251104_004300/fold_0"
    
    # 2. Load dataset_info to get test files and true ratios
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    fold_0_data = next((item for item in dataset_info if item['fold'] == 0), None)
    if not fold_0_data:
        print("No data found for fold 0.")
        return

    test_items = fold_0_data.get('test', [])
    if not test_items:
        print("No test files found in fold 0.")
        return

    true_ratios = [parse_concentration_ratio(item['input']) for item in test_items]
    valid_test_data = [(item, ratio) for item, ratio in zip(test_items, true_ratios) if ratio is not None]
    
    if not valid_test_data:
        print("No valid test files found for C6/FITC ratio analysis.")
        return
        
    test_items, true_ratios = zip(*valid_test_data)

    # 3. Load pre-computed predictions and calculate predicted ratios
    predicted_ratios = []
    valid_true_ratios = []
    for i, item in enumerate(test_items):
        try:
            target_0_path = os.path.join(results_dir, f"test_sample_{i}_target_0.npz")
            target_1_path = os.path.join(results_dir, f"test_sample_{i}_target_1.npz")

            map_0 = np.load(target_0_path)['pred']
            map_1 = np.load(target_1_path)['pred']
            
            # Stack the two maps to create the expected input for the ratio calculation
            # The shape should be (N, C, H, W), so we add a batch dimension
            predicted_map = np.stack([map_0, map_1], axis=0)[np.newaxis, ...] # Shape: (1, 2, H, W)
            
            ratio = calculate_integrated_intensity_ratio(predicted_map)
            predicted_ratios.append(ratio[0])
            valid_true_ratios.append(true_ratios[i])

        except FileNotFoundError:
            print(f"Warning: Prediction file for sample {i} not found. Skipping.")
            continue

    # 4. Perform linear regression
    if len(valid_true_ratios) < 2 or len(predicted_ratios) < 2:
        print("Not enough data to perform linear regression.")
        return

    true_ratios_np = np.array(valid_true_ratios).reshape(-1, 1)
    predicted_ratios_np = np.array(predicted_ratios).reshape(-1, 1)

    reg = LinearRegression().fit(true_ratios_np, predicted_ratios_np)
    r_squared = reg.score(true_ratios_np, predicted_ratios_np)

    print(f"Linear Regression R-squared: {r_squared:.4f}")

    # 5. Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ratios_np, predicted_ratios_np, alpha=0.7, label='Predicted vs. True Ratios')
    plt.plot(true_ratios_np, reg.predict(true_ratios_np), color='red', linewidth=2, label=f'Linear Fit (R²={r_squared:.4f})')
    plt.xlabel("True Concentration Ratio (C6 / FITC)")
    plt.ylabel("Predicted Integrated Intensity Ratio (from .npz files)")
    plt.title("Concentration Ratio Analysis from Pre-computed Results")
    plt.legend()
    plt.grid(True)
    
    output_path = "concentration_analysis_C6_FITC_from_npz.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    main()