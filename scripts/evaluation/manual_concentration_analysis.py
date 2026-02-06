import json
import os
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_json_robustly(file_path):
    """
    Loads a JSON file containing multiple JSON objects that are not in a list
    and may be missing commas between them.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        return []

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    json_list = []
    pos = 0
    content = content.strip()
    while pos < len(content):
        try:
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos == len(content):
                break
            
            obj, end_pos = decoder.raw_decode(content[pos:])
            json_list.append(obj)
            pos += end_pos
        except json.JSONDecodeError as e:
            next_brace = content.find('{', pos + 1)
            if next_brace == -1:
                print(f"Warning: JSON decoding error at position {pos}: {e}. Halting parse.")
                break 
            else:
                print(f"Warning: JSON decoding error at position {pos}: {e}. Skipping to next '{{'.")
                pos = next_brace

    return json_list

def parse_concentration_ratio(filename, component_pair):
    """
    Parses the filename to extract the concentration ratio for a given component pair.
    e.g., component_pair = 'C6_FITC'
    """
    c1, c2 = component_pair.split('_')
    match = re.search(rf'{c1}_(\d+)_{c2}_(\d+)', Path(filename).stem)
    if match:
        concentration1 = int(match.group(1))
        concentration2 = int(match.group(2))
        return concentration1 / concentration2 if concentration2 != 0 else float('inf')
    return None

def calculate_integrated_intensity_ratio(predicted_maps):
    """
    Calculates the ratio of integrated intensities from the predicted component maps.
    """
    integrated_intensities = np.sum(predicted_maps, axis=(2, 3))
    ratios = integrated_intensities[:, 0] / (integrated_intensities[:, 1] + 1e-9)
    return ratios

def analyze_fold(results_dir, dataset_info_path, component_pair, fold_id, output_dir):
    """
    Performs concentration analysis for a single fold.
    """
    print(f"Analyzing {component_pair} - Fold {fold_id}...")

    # 1. Load dataset_info to get test files and true ratios
    dataset_info = load_json_robustly(dataset_info_path)
    if not dataset_info:
        print(f"  Error: Could not load or parse dataset_info from {dataset_info_path}")
        return

    if not (0 <= fold_id < len(dataset_info)):
        print(f"  Error: fold_id {fold_id} is out of bounds for dataset_info with length {len(dataset_info)}.")
        return
        
    fold_data = dataset_info[fold_id]

    if not fold_data:
        print(f"  Warning: No data found for fold {fold_id} in {dataset_info_path}. Skipping.")
        return

    test_items = fold_data.get('test', [])
    if not test_items:
        print(f"  Warning: No test files found in fold {fold_id}. Skipping.")
        return

    true_ratios = [parse_concentration_ratio(item['input'], component_pair) for item in test_items]
    valid_test_data = [(item, ratio) for item, ratio in zip(test_items, true_ratios) if ratio is not None]
    
    if not valid_test_data:
        print(f"  Warning: No valid test files found for {component_pair} ratio analysis in fold {fold_id}. Skipping.")
        return
        
    test_items, true_ratios = zip(*valid_test_data)

    # 2. Load pre-computed predictions and calculate predicted ratios
    predicted_ratios = []
    valid_true_ratios = []
    for i, item in enumerate(test_items):
        try:
            target_0_path = os.path.join(results_dir, f"test_sample_{i}_target_0.npz")
            target_1_path = os.path.join(results_dir, f"test_sample_{i}_target_1.npz")

            map_0 = np.load(target_0_path)['pred']
            map_1 = np.load(target_1_path)['pred']
            
            predicted_map = np.stack([map_0, map_1], axis=0)[np.newaxis, ...]
            
            ratio = calculate_integrated_intensity_ratio(predicted_map)
            predicted_ratios.append(ratio[0])
            valid_true_ratios.append(true_ratios[i])

        except FileNotFoundError:
            print(f"  Warning: Prediction file for sample {i} in fold {fold_id} not found. Skipping.")
            continue

    # 3. Perform linear regression
    if len(valid_true_ratios) < 2 or len(predicted_ratios) < 2:
        print(f"  Warning: Not enough data for linear regression in fold {fold_id}. Skipping.")
        return

    true_ratios_np = np.array(valid_true_ratios).reshape(-1, 1)
    predicted_ratios_np = np.array(predicted_ratios).reshape(-1, 1)

    reg = LinearRegression().fit(true_ratios_np, predicted_ratios_np)
    r_squared = reg.score(true_ratios_np, predicted_ratios_np)

    print(f"  R-squared: {r_squared:.4f}")

    # 4. Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ratios_np, predicted_ratios_np, alpha=0.7, label='Predicted vs. True Ratios')
    plt.plot(true_ratios_np, reg.predict(true_ratios_np), color='red', linewidth=2, label=f'Linear Fit (R²={r_squared:.4f})')
    plt.xlabel(f"True Concentration Ratio ({component_pair.replace('_', ' / ')})")
    plt.ylabel("Predicted Integrated Intensity Ratio")
    plt.title(f"Concentration Analysis: {component_pair} - Fold {fold_id}")
    plt.legend()
    plt.grid(True)
    
    output_filename = f"analysis_{component_pair}_fold_{fold_id}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"  Plot saved to {output_path}")
    plt.close()


def main():
    # --- Manual Configuration ---
    MODEL_NAME = "DualUNetSharedEncoder"  # e.g., "DualUNetSharedEncoder", "DualSimpleCNN"
    TRAINING_RUN = "training_20251105_022308"  # e.g., "training_20251103_231910"
    COMPONENT_PAIR = "FITC_HPTS"  # e.g., "C6_FITC", "FITC_HPTS"
    FOLD_ID = 11  # e.g., 0, 1, 2, 3, 4
    # --------------------------

    base_results_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/regression/"
    base_config_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/"
    output_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/concentration_analysis_plots/"
    
    os.makedirs(output_dir, exist_ok=True)

    # Construct paths based on manual configuration
    fold_results_dir = os.path.join(base_results_dir, MODEL_NAME, TRAINING_RUN, f"fold_{FOLD_ID}")
    dataset_info_filename = f"regression_dataset_info_{COMPONENT_PAIR}.json"
    dataset_info_path = os.path.join(base_config_dir, dataset_info_filename)

    # Check if necessary paths and files exist
    if not os.path.isdir(fold_results_dir):
        print(f"Error: Fold results directory not found at {fold_results_dir}")
        return

    if not os.path.isfile(dataset_info_path):
        print(f"Error: Dataset info file not found at {dataset_info_path}")
        return

    # Run the analysis for the specified configuration
    analyze_fold(fold_results_dir, dataset_info_path, COMPONENT_PAIR, FOLD_ID, output_dir)

if __name__ == '__main__':
    main()