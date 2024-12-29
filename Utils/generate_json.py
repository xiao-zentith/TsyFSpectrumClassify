import os
import json
from itertools import combinations
from sklearn.model_selection import KFold
from sympy.simplify.cse_main import preprocess_for_cse


def list_xlsx_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]


def generate_mixup_filenames(file_pairs, output_folder):
    augmented_names = []
    for file1, file2 in file_pairs:
        # base_name1 = os.path.splitext(file1)[0]
        # base_name2 = os.path.splitext(file2)[0]
        base_name1 = file1
        base_name2 = file2
        mixed_name = f"mixed_{base_name1}_with_{base_name2}.xlsx"
        noise_mixed_name = f"noise_mixed_{base_name1}_with_{base_name2}.xlsx"
        mixed_path = os.path.join(output_folder, mixed_name)
        noise_mixed_path = os.path.join(output_folder, noise_mixed_name)
        augmented_names.append(mixed_path)
        augmented_names.append(noise_mixed_path)
    return augmented_names


def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    input_folder = config["dataset_raw"]
    processed_folder = config["dataset_processed"]
    target_folders = [config["dataset_target1"], config["dataset_target2"]]
    output_json = "../dataset_info.json"

    xlsx_files = list_xlsx_files(input_folder)

    # Initialize the outer cross-validator
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    dataset_info_list = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(xlsx_files)):
        train_val_files = [xlsx_files[i] for i in train_val_indices]
        test_files = [xlsx_files[i] for i in test_indices]

        # Initialize the inner cross-validator
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_files)):
            train_files = [train_val_files[i] for i in train_indices]
            val_files = [train_val_files[i] for i in val_indices]

            # Generate mixup pairs for training set
            train_file_pairs = list(combinations(train_files, 2))
            augmented_files = generate_mixup_filenames(train_file_pairs, processed_folder)

            # Record the paths along with their corresponding target files
            dataset_info = {
                "fold": fold_idx,
                "inner_fold": inner_fold_idx,
                "train": [{"input": os.path.join(processed_folder, os.path.basename(tf)),
                           "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                       tfolder in target_folders]} for tf in augmented_files],
                "validation": [{"input": os.path.join(processed_folder, os.path.basename(vf)),
                                "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                               for vf in val_files],
                "test": [{"input": os.path.join(processed_folder, os.path.basename(tef)),
                          "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]} for
                         tef in test_files]
            }

            dataset_info_list.append(dataset_info)

    # Write to JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset_info_list, f, indent=4)


if __name__ == "__main__":
    config_file = r'C:\Users\xiao\PycharmProjects\TsyFSpectrumClassify\config.json'  # Update this path if necessary
    main(config_file)



