import json
import os
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add project root to sys.path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_json_robustly(file_path):
    """
    Loads a JSON file that may contain multiple concatenated JSON objects.
    This handles cases where JSON objects are not in a valid list (e.g., missing commas).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # If content is empty, return an empty list
    if not content:
        return []

    # First, try to load as a standard JSON array
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If it fails, proceed with robust parsing
        pass

    decoder = json.JSONDecoder()
    json_list = []
    pos = 0
    while pos < len(content):
        # Skip whitespace
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos == len(content):
            break

        # Find the start of a JSON object
        if content[pos] == '{':
            try:
                obj, end_pos = decoder.raw_decode(content[pos:])
                json_list.append(obj)
                pos += end_pos
            except json.JSONDecodeError:
                # In case of an error, find the next potential start of an object
                next_brace = content.find('{', pos + 1)
                if next_brace == -1:
                    break  # No more objects
                pos = next_brace
        else:
            # If the character is not a brace, we might have a malformed file
            # For simplicity, we just advance to the next character
            pos += 1
            
    return json_list

def parse_concentration_ratio(filename, component_pair):
    """
    Parses the filename to extract the concentration ratio for a given component pair.
    e.g., component_pair = 'C6_FITC', filename = 'C6_7_FITC_3.xlsx' -> 7/3
    """
    c1, c2 = component_pair.split('_')
    # Regex to find concentrations, allowing for floating point numbers as well
    match = re.search(rf'{c1}_([\d\.]+)_{c2}_([\d\.]+)', Path(filename).stem)
    if match:
        try:
            concentration1 = float(match.group(1))
            concentration2 = float(match.group(2))
            return concentration1 / concentration2 if concentration2 != 0 else float('inf')
        except (ValueError, ZeroDivisionError):
            return None
    return None

def calculate_predicted_ratio(pred_map_1, pred_map_2):
    """
    Calculates the ratio of integrated intensities from two predicted component maps.
    """
    intensity1 = np.sum(pred_map_1)
    intensity2 = np.sum(pred_map_2)
    return intensity1 / (intensity2 + 1e-9) # Add epsilon to avoid division by zero

def analyze_fold(model_name, training_run, component_pair, fold_id):
    """
    Main analysis function for a single fold.
    """
    print(f"--- Starting Analysis ---")
    print(f"Component Pair: {component_pair}")
    print(f"Model: {model_name}")
    print(f"Training Run: {training_run}")
    print(f"Fold ID: {fold_id}")
    print("-------------------------")

    # --- 1. Construct Paths ---
    base_results_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/regression/"
    base_config_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/"
    output_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/concentration_analysis_plots/"
    
    os.makedirs(output_dir, exist_ok=True)

    fold_results_dir = Path(base_results_dir) / model_name / training_run / f"fold_{fold_id}"
    dataset_info_path = Path(base_config_dir) / f"regression_dataset_info_{component_pair}.json"

    if not fold_results_dir.is_dir():
        print(f"\nERROR: Fold results directory not found at: {fold_results_dir}")
        return
    if not dataset_info_path.is_file():
        print(f"\nERROR: Dataset info file not found at: {dataset_info_path}")
        return

    # --- 2. Load Dataset Info and True Ratios ---
    dataset_info = load_json_robustly(dataset_info_path)
    if not dataset_info:
        print(f"\nERROR: Could not load or parse dataset_info from {dataset_info_path}")
        return

    if not (0 <= fold_id < len(dataset_info)):
        print(f"\nERROR: Fold ID {fold_id} is out of bounds. File contains {len(dataset_info)} folds.")
        return
        
    fold_data = dataset_info[fold_id]
    test_items = fold_data.get('test', [])

    if not test_items:
        print(f"\nWARNING: No test files found in fold {fold_id}. Skipping.")
        return

    # --- 3. Extract True and Predicted Ratios ---
    true_ratios = []
    predicted_ratios = []

    for i, item in enumerate(test_items):
        input_filename = item['input']
        
        # Get true ratio
        true_ratio = parse_concentration_ratio(input_filename, component_pair)
        if true_ratio is None:
            print(f"  - Skipping sample {i}: Could not parse true ratio from '{input_filename}'")
            continue

        # Get predicted ratio
        try:
            # Paths to the predicted .npz files for each component
            pred_path_0 = fold_results_dir / f"test_sample_{i}_target_0.npz"
            pred_path_1 = fold_results_dir / f"test_sample_{i}_target_1.npz"

            pred_map_0 = np.load(pred_path_0)['pred']
            pred_map_1 = np.load(pred_path_1)['pred']
            
            predicted_ratio = calculate_predicted_ratio(pred_map_0, pred_map_1)
            
            true_ratios.append(true_ratio)
            predicted_ratios.append(predicted_ratio)

        except FileNotFoundError:
            print(f"  - Skipping sample {i}: Prediction file not found.")
            continue
        except Exception as e:
            print(f"  - Skipping sample {i}: Error loading prediction - {e}")
            continue

    if len(true_ratios) < 2:
        print(f"\nWARNING: Not enough valid data points ({len(true_ratios)}) for linear regression. Skipping.")
        return

    # --- 4. Perform Linear Regression ---
    true_ratios_np = np.array(true_ratios).reshape(-1, 1)
    predicted_ratios_np = np.array(predicted_ratios).reshape(-1, 1)

    reg = LinearRegression().fit(true_ratios_np, predicted_ratios_np)
    r_squared = reg.score(true_ratios_np, predicted_ratios_np)

    print(f"\nAnalysis Complete for Fold {fold_id}:")
    print(f"  - R-squared: {r_squared:.4f}")

    # --- 5. Plot the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ratios_np, predicted_ratios_np, alpha=0.8, edgecolors='k', label='Data Points')
    plt.plot(true_ratios_np, reg.predict(true_ratios_np), color='#FF5733', linewidth=2, label=f'Linear Fit (R² = {r_squared:.4f})')
    
    plt.xlabel(f"True Concentration Ratio ({component_pair.replace('_', ' / ')})", fontsize=12)
    plt.ylabel("Predicted Integrated Intensity Ratio", fontsize=12)
    plt.title(f"Concentration Analysis: {model_name} - Fold {fold_id}", fontsize=14, weight='bold')
    plt.legend(fontsize=10)
    
    # Save the plot
    output_filename = f"analysis_{component_pair}_{model_name}_fold_{fold_id}.png"
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"  - Plot saved to: {output_path}")
    plt.close()


def main():
    # =================================================================================
    # --- MANUAL CONFIGURATION ---
    # =================================================================================
    # Component pair to analyze (e.g., "C6_FITC", "FITC_HPTS")
    # This determines which regression_dataset_info_XXX.json to use.
    COMPONENT_PAIR = "C6_FITC"

    # Model and training run to evaluate
    MODEL_NAME = "DualUNetSharedEncoder"
    TRAINING_RUN = "training_20251104_004300"
    
    # Fold to analyze
    FOLD_ID = 8
    # =================================================================================

    analyze_fold(MODEL_NAME, TRAINING_RUN, COMPONENT_PAIR, FOLD_ID)


if __name__ == '__main__':
    main()