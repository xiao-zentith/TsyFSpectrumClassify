import os
import json
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold

def list_xlsx_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

def generate_mixup_filenames(file_pairs):
    augmented_names = []
    for file1, file2 in file_pairs:
        base_name1 = os.path.splitext(file1)[0]
        base_name2 = os.path.splitext(file2)[0]
        mixed_name = f"mixed_{base_name1}_with_{base_name2}.xlsx"
        noise_mixed_name = f"noise_mixed_{base_name1}_with_{base_name2}.xlsx"
        mixed_path = mixed_name
        noise_mixed_path = noise_mixed_name
        augmented_names.append(mixed_path)
        augmented_names.append(noise_mixed_path)
    return augmented_names


def simple_data_split(xlsx_files, data_split):
    train_val_files, test_files = train_test_split(xlsx_files, test_size=data_split["test"], random_state=42)
    train_files, val_files = train_test_split(train_val_files,
                                              test_size=(data_split["val"] / (data_split["train"] + data_split["val"])),
                                              random_state=42)
    return train_files, val_files, test_files


def nested_cross_validation(is_mixup, xlsx_files, data_split, processed_folder, target_folders):
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)
    dataset_info_list = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(xlsx_files)):
        train_val_files = [xlsx_files[i] for i in train_val_indices]
        test_files = [xlsx_files[i] for i in test_indices]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)

        for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_files)):
            train_files = [train_val_files[i] for i in train_indices]
            val_files = [train_val_files[i] for i in val_indices]

            # Generate mixup pairs for training set
            train_file_pairs = list(combinations(train_files, 2))
            augmented_files = generate_mixup_filenames(train_file_pairs)

            # Include original train files in the augmented files
            if is_mixup:
                all_train_files = train_files + augmented_files
            else:
                all_train_files = train_files

            # Record the paths along with their corresponding target files
            dataset_info = {
                "fold": fold_idx,
                "inner_fold": inner_fold_idx,
                "train": [{"input": os.path.join(processed_folder, tf),
                           "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                       tfolder in target_folders]} for tf in all_train_files],
                "validation": [{"input": os.path.join(processed_folder, vf),
                                "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                               for vf in val_files],
                "test": [{"input": os.path.join(processed_folder, tef),
                          "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]} for
                         tef in test_files]
            }

            dataset_info_list.append(dataset_info)

    return dataset_info_list


def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    input_folder = config["dataset_raw"]
    processed_folder = config["dataset_processed"]
    target_folders = [config["dataset_target1"], config["dataset_target2"], config["dataset_target3"], config["dataset_target4"]]
    data_split = config["data_split"]
    is_cross_validation = config["is_cross_validation"]
    output_json =  "../dataset_info.json"
    is_mixup = config["is_mixup"]
    xlsx_files = list_xlsx_files(input_folder)

    if is_cross_validation:
        dataset_info_list = nested_cross_validation(is_mixup, xlsx_files, data_split, processed_folder, target_folders)
    else:
        train_files, val_files, test_files = simple_data_split(xlsx_files, data_split)

        # Generate mixup pairs for training set
        train_file_pairs = list(combinations(train_files, 2))
        augmented_files = generate_mixup_filenames(train_file_pairs)

        # Include original train files in the augmented files
        if is_mixup:
            all_train_files = train_files + augmented_files
        else:
            all_train_files = train_files

        # Record the paths along with their corresponding target files
        dataset_info = {
            "train": [{"input": os.path.join(processed_folder, tf),
                       "targets": [os.path.join(tfolder, os.path.basename(os.path.splitext(tf)[0] + ".xlsx")) for
                                   tfolder in target_folders]} for tf in all_train_files],
            "validation": [
                {"input": os.path.join(processed_folder, vf),
                 "targets": [os.path.join(tfolder, os.path.basename(vf)) for tfolder in target_folders]}
                for vf in val_files],
            "test": [
                {"input": os.path.join(processed_folder, tef),
                 "targets": [os.path.join(tfolder, os.path.basename(tef)) for tfolder in target_folders]}
                for tef in test_files]
        }

        dataset_info_list = [dataset_info]

    # Write to JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset_info_list, f, indent=4)


if __name__ == "__main__":
    config_file = '../config.json'  # Update this path if necessary
    main(config_file)



