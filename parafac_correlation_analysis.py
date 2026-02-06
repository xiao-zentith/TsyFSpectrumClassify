import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Configuration ---
# 【!!】确保这里的 COMPONENT_PAIR 是你想要分析的那个
COMPONENT_PAIR = "C_HPTS" 
CONFIG_DIR = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/configs/regression"
PLOTS_DIR = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/results/parafac_analysis_plots"
DATA_ROOT = "/home/asus515/PycharmProjects/TsyFSpectrumClassify_remote/data/dataset_regression"

# --- Function Definitions ---

def load_json_robustly(filepath):
    """
    Loads a JSON file that may contain multiple concatenated JSON objects.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    decoder = json.JSONDecoder()
    objects = []
    pos = 0
    # Handle cases where the file is a list of objects or just concatenated objects
    content = content.strip()
    if content.startswith('['):
        return json.loads(content)
        
    while pos < len(content):
        try:
            obj, pos = decoder.raw_decode(content, pos)
            objects.append(obj)
            # Skip whitespace and commas between objects
            match = re.match(r'[\s,]*', content[pos:])
            if match:
                pos += match.end()
        except json.JSONDecodeError:
            # If there's trailing whitespace or garbage, we might be done
            break
            
    return objects

def parse_concentration_ratio(filename, component_pair):
    """
    Parses the concentration ratio from the filename.
    e.g., C6_7_FITC_3.xlsx -> 7 / (7 + 3) = 0.7
    
    【!!】重要：请确保你想要的比率是第一个组分 (c1) 的比率
    """
    components = component_pair.split('_')
    if len(components) != 2:
        raise ValueError(f"Invalid COMPONENT_PAIR: {component_pair}")
    c1, c2 = components

    # 尝试匹配 c1_val_c2_val (例如 FITC_7_HPTS_3)
    match = re.search(rf'{c1}_(\d+)_{c2}_(\d+)', os.path.basename(filename))
    if match:
        c1_val = int(match.group(1))
        c2_val = int(match.group(2))
        if c1_val + c2_val == 10:
            return c1_val / 10.0 # 返回 c1 的比率
            
    # 尝试匹配 c2_val_c1_val (例如 HPTS_3_FITC_7)
    match = re.search(rf'{c2}_(\d+)_{c1}_(\d+)', os.path.basename(filename))
    if match:
        c1_val = int(match.group(2)) # 注意 group 索引
        c2_val = int(match.group(1))
        if c1_val + c2_val == 10:
            return c1_val / 10.0 # 仍然返回 c1 的比率
            
    return None

def read_xlsx_to_numpy(filepath):
    """
    Reads an XLSX file into a NumPy array, skipping the header.
    """
    try:
        # Skip the first row (header)
        df = pd.read_excel(filepath, header=None, skiprows=1)
        return df.to_numpy(dtype=float)
    except ValueError:
        print(f"Warning: Non-numeric data found in {filepath} after skipping header. Skipping file.")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}. Skipping file.")
        return None

def calculate_parafac_ratio(matrix1, matrix2):
    """
    Calculates the ratio of the sums of two matrices.
    返回 matrix1 / (matrix1 + matrix2) 的比率
    """
    # 截断负值，因为强度不能为负
    sum1 = np.sum(np.maximum(matrix1, 0))
    sum2 = np.sum(np.maximum(matrix2, 0))
    total = sum1 + sum2
    if total < 1e-9: # 避免除以零
        return 0.5 # 如果两个图谱都是0，则无法判断
    return sum1 / total

def analyze_parafac_correlation(component_pair):
    """
    Main function to perform the PARAFAC correlation analysis across all folds.
    """
    # 1. Locate and load the dataset info file
    dataset_info_filename = f"regression_dataset_info_{component_pair}.json"
    dataset_info_path = os.path.join(CONFIG_DIR, dataset_info_filename)

    if not os.path.exists(dataset_info_path):
        print(f"Error: Dataset info file not found at {dataset_info_path}")
        return

    all_folds_data = load_json_robustly(dataset_info_path)
    
    # 2. Aggregate test data from all folds and remove duplicates
    all_input_files = set()
    for fold_data in all_folds_data:
        for item in fold_data.get('test', []):
            if isinstance(item, dict) and 'input' in item:
                all_input_files.add(item['input'])
            elif isinstance(item, list): # 兼容旧格式
                all_input_files.add(item[0])
                
    unique_input_files = sorted(list(all_input_files))
    print(f"Found {len(all_folds_data)} folds. Aggregated {len(unique_input_files)} unique test samples.")

    true_ratios = []
    parafac_ratios = []
    
    components = component_pair.split('_')
    c1_name, c2_name = components[0], components[1]
    print(f"Analyzing... X-axis = True Ratio of {c1_name}, Y-axis = PARAFAC Ratio of {c1_name}")

    # 3. Process each unique test file
    for input_file_path in unique_input_files:
        base_filename = os.path.basename(input_file_path)
        
        # 4. Calculate true ratio from filename
        true_ratio = parse_concentration_ratio(base_filename, component_pair)
        if true_ratio is None:
            print(f"Warning: Could not parse true ratio from {base_filename}. Skipping.")
            continue

        # 5. Get target file paths directly from the data
        target_files = []
        for fold_data in all_folds_data:
            for item in fold_data.get('test', []):
                # 兼容新旧JSON格式
                item_input = item.get('input') if isinstance(item, dict) else item[0]
                if item_input == input_file_path:
                    target_files = item.get('targets') if isinstance(item, dict) else item[1:]
                    break
            if target_files:
                break

        if not target_files or len(target_files) < 2:
            print(f"Warning: Target files for {base_filename} not found in JSON data. Skipping.")
            continue

        # 确保 target_file_1 对应 c1, target_file_2 对应 c2
        # (这假设你的 'targets' 列表是按 [target_c1, target_c2] 顺序排列的)
        # 【!!】请根据你的JSON结构检查此处的索引
        # 根据你的图 (FITC_HPTS)，当X(True Ratio)小时, Y(PARAFAC Ratio)也小，
        # 这意味着 parse_concentration_ratio(c1/c1+c2) 和 calculate_parafac_ratio(m1/m1+m2)
        # 对应的是同一种物质。
        
        # 【!!】重要：原代码是 target_files[1], target_files[0]
        # 【!!】我将其改为 [0], [1] 以匹配 c1, c2。请你确认！
        target_file_1, target_file_2 = target_files[0], target_files[1] 

        # 6. Read PARAFAC matrices
        matrix1 = read_xlsx_to_numpy(target_file_1)
        matrix2 = read_xlsx_to_numpy(target_file_2)

        if matrix1 is None or matrix2 is None:
            print(f"Warning: Skipping sample {base_filename} due to data reading errors.")
            continue
            
        parafac_ratio = calculate_parafac_ratio(matrix1, matrix2)

        true_ratios.append(true_ratio)
        parafac_ratios.append(parafac_ratio)

    if len(true_ratios) < 2:
        print(f"Error: Not enough valid data points ({len(true_ratios)}) for linear regression.")
        return

    # 7. Perform R-squared validation
    true_ratios_np = np.array(true_ratios).reshape(-1, 1)
    parafac_ratios_np = np.array(parafac_ratios)

    reg = LinearRegression().fit(true_ratios_np, parafac_ratios_np)
    r2 = r2_score(parafac_ratios_np, reg.predict(true_ratios_np))

    print(f"\n--- PARAFAC Correlation Analysis Summary ---")
    print(f"Component Pair: {component_pair}")
    print(f"Total Unique Samples Analyzed: {len(true_ratios)}")
    print(f"R-squared: {r2:.4f}")
    print("-----------------------------------------")

    # ----------------------------------------------------
    # --- 8. Plotting (SCI Publication Quality) ---
    # ----------------------------------------------------
    
    # 设置更适合出版的字体和字号
    plt.rcParams.update({'font.size': 14, 'font.family': 'Arial', 'font.weight': 'normal'})

    plt.figure(figsize=(8, 8)) # 创建一个正方形画布
    ax = plt.gca()

    # 绘制散点图 (优化：增加描边和大小)
    ax.scatter(true_ratios, parafac_ratios, 
               s=80,  # 增大点的大小
               alpha=0.7, 
               label='PARAFAC Results', 
               edgecolor='black', # 增加黑色描边
               linewidth=0.5,
               zorder=3) # zorder=3 确保点在网格线之上

    # 绘制线性拟合线 (优化：加粗)
    ax.plot(true_ratios_np, reg.predict(true_ratios_np), 
            color='red', 
            linewidth=2.5,  # 加粗线条
            label='Linear Fit', # R^2值将作为文本标注
            zorder=2)

    # 绘制理想的y=x线 (优化：加粗)
    ax.plot([0, 1], [0, 1], 
            color='grey', 
            linestyle='--', 
            linewidth=2.5,  # 加粗线条
            label='Ideal y=x Line',
            zorder=1)

    # --- 优化图表元素 ---
    
    # 优化：设置坐标轴标签（加粗）
    # 【!!】请根据需要修改这里的标签
    ax.set_xlabel(f'True Concentration Ratio ({c1_name})', fontweight='bold', fontsize=16)
    ax.set_ylabel(f'PARAFAC Intensity Ratio ({c1_name})', fontweight='bold', fontsize=16)

    # 优化：设置坐标轴范围和刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14) # 增大刻度数字

    # 优化：将 R^2 值作为文本直接添加到图上
    ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', 
            transform=ax.transAxes, # 使用相对坐标
            fontsize=16, 
            fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加图例
    ax.legend(loc='lower right', fontsize=14, frameon=True, shadow=False) # 移到右下角
    
    # 添加网格
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 强制X,Y轴等比例
    ax.set_aspect('equal', adjustable='box')

    # 优化：确保所有元素都显示完整
    plt.tight_layout()

    # --- 保存图表 ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 保存为高分辨率PNG，并额外保存一份PDF（矢量图）
    plot_filename_png = f"parafac_analysis_{component_pair}_optimized.png"
    plot_filename_pdf = f"parafac_analysis_{component_pair}_optimized.pdf"
    
    plot_path_png = os.path.join(PLOTS_DIR, plot_filename_png)
    plot_path_pdf = os.path.join(PLOTS_DIR, plot_filename_pdf)
    
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf) # 矢量图，期刊可能更喜欢
    
    print(f"Optimized plot saved to {plot_path_png} (and .pdf)")
    plt.close()


if __name__ == "__main__":
    analyze_parafac_correlation(COMPONENT_PAIR)