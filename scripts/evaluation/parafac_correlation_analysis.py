import json
import os
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add project root to sys.path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_json_robustly(file_path):
    """
    Loads a JSON file that may contain multiple concatenated JSON objects.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    json_list = []
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos == len(content):
            break
        if content[pos] == '{':
            try:
                obj, end_pos = decoder.raw_decode(content[pos:])
                json_list.append(obj)
                pos += end_pos
            except json.JSONDecodeError:
                next_brace = content.find('{', pos + 1)
                if next_brace == -1:
                    break
                pos = next_brace
        else:
            pos += 1
    return json_list

def parse_concentration_ratio(filename, component_pair):
    """
    Parses the filename to extract the true concentration ratio.
    """
    c1, c2 = component_pair.split('_')
    match = re.search(rf'{c1}_([\d\.]+)_{c2}_([\d\.]+)', Path(filename).stem)
    if match:
        try:
            concentration1 = float(match.group(1))
            concentration2 = float(match.group(2))
            return concentration1 / concentration2 if concentration2 != 0 else float('inf')
        except (ValueError, ZeroDivisionError):
            return None
    return None

def read_xlsx_to_numpy(file_path):
    """
    Reads an xlsx file, skipping the header, and converts it to a numpy array of floats.
    """
    try:
        # Skip the first row (header) and read the rest of the data
        df = pd.read_excel(file_path, header=None, skiprows=1)
        return df.to_numpy(dtype=float)
    except FileNotFoundError:
        print(f"  - ERROR: XLSX file not found at {file_path}")
        return None
    except ValueError:
        print(f"  - ERROR: Non-numeric data found in {file_path} (even after skipping header). Could not convert to matrix.")
        return None
    except Exception as e:
        print(f"  - ERROR: Failed to read {file_path}: {e}")
        return None

def calculate_parafac_ratio(target_files):
    """
    Calculates the ratio of integrated intensities from PARAFAC target files (.xlsx).
    """
    if len(target_files) != 2:
        return None

    matrix1 = read_xlsx_to_numpy(target_files[1])
    matrix2 = read_xlsx_to_numpy(target_files[0])

    if matrix1 is None or matrix2 is None:
        return None

    intensity1 = np.sum(matrix1)
    intensity2 = np.sum(matrix2)
    
    return intensity1 / (intensity2 + 1e-9) # Add epsilon to avoid division by zero

def analyze_parafac_correlation(component_pair, fold_id):
    """
    Analyzes the correlation between true ratios and PARAFAC-derived ratios.
    """
    print(f"--- Starting PARAFAC Correlation Analysis ---")
    print(f"Component Pair: {component_pair}")
    print(f"Fold ID: {fold_id}")
    print("---------------------------------------------")

    # --- 1. Construct Paths ---
    base_config_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression/"
    output_dir = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/parafac_analysis_plots/"
    
    os.makedirs(output_dir, exist_ok=True)

    dataset_info_path = Path(base_config_dir) / f"regression_dataset_info_{component_pair}.json"

    if not dataset_info_path.is_file():
        print(f"\nERROR: Dataset info file not found at: {dataset_info_path}")
        return

    # --- 2. Load Dataset Info ---
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

    # --- 3. Extract True and PARAFAC Ratios ---
    true_ratios = []
    parafac_ratios = []

    print("Processing test samples...")
    for i, item in enumerate(test_items):
        # Get true ratio from input filename
        true_ratio = parse_concentration_ratio(item['input'], component_pair)
        if true_ratio is None:
            print(f"  - Sample {i}: Skipping, could not parse true ratio from '{item['input']}'")
            continue

        # Get PARAFAC ratio from target files
        parafac_ratio = calculate_parafac_ratio(item['targets'])
        if parafac_ratio is None:
            print(f"  - Sample {i}: Skipping, could not calculate PARAFAC ratio.")
            continue
            
        true_ratios.append(true_ratio)
        parafac_ratios.append(parafac_ratio)

    if len(true_ratios) < 2:
        print(f"\nWARNING: Not enough valid data points ({len(true_ratios)}) for linear regression. Skipping.")
        return

    # --- 4. Perform Linear Regression ---
    true_ratios_np = np.array(true_ratios).reshape(-1, 1)
    parafac_ratios_np = np.array(parafac_ratios).reshape(-1, 1)

    reg = LinearRegression().fit(true_ratios_np, parafac_ratios_np)
    r_squared = reg.score(true_ratios_np, parafac_ratios_np)

    print(f"\nAnalysis Complete for Fold {fold_id}:")
    print(f"  - R-squared: {r_squared:.4f}")

    # --- 5. Plot the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    plt.scatter(true_ratios_np, parafac_ratios_np, alpha=0.8, edgecolors='k', label='Data Points')
    plt.plot(true_ratios_np, reg.predict(true_ratios_np), color='#007ACC', linewidth=2, label=f'Linear Fit (R² = {r_squared:.4f})')
    
    plt.xlabel(f"True Concentration Ratio ({component_pair.replace('_', ' / ')})", fontsize=12)
    plt.ylabel("PARAFAC Integrated Intensity Ratio", fontsize=12)
    plt.title(f"PARAFAC vs. True Ratio Correlation - {component_pair} - Fold {fold_id}", fontsize=14, weight='bold')
    plt.legend(fontsize=10)
    
    # Save the plot
    output_filename = f"parafac_analysis_{component_pair}_fold_{fold_id}.png"
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"  - Plot saved to: {output_path}")
    plt.close()


def main():
    # =================================================================================
    # --- MANUAL CONFIGURATION ---
    # =================================================================================
    # Component pair to analyze (e.g., "C6_FITC", "FITC_HPTS")
    COMPONENT_PAIR = "FITC_HPTS"

    # Fold to analyze
    FOLD_ID = 0
    # =================================================================================

    analyze_parafac_correlation(COMPONENT_PAIR, FOLD_ID)


if __name__ == '__main__':
    main()