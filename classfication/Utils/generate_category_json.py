import os
import json
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold

def list_txt_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def generate_mixup_filenames(file_pairs, category, processed_folder):
    augmented_names = []
    for file1, file2 in file_pairs:
        base_name1 = os.path.splitext(file1)[0]
        base_name2 = os.path.splitext(file2)[0]
        mixed_name = f"mixed_{base_name1}.txt_with_{base_name2}.txt.txt"
        noise_mixed_name = f"noise_mixed_{base_name1}.txt_with_{base_name2}.txt.txt"
        mixed_path = os.path.join(processed_folder, category, mixed_name)
        noise_mixed_path = os.path.join(processed_folder, category, noise_mixed_name)
        augmented_names.append(mixed_path)
        augmented_names.append(noise_mixed_path)
    return augmented_names

def simple_data_split(txt_files, categories, data_split):
    train_val_files, test_files, train_val_categories, test_categories = train_test_split(
        txt_files, categories, test_size=data_split["test"], stratify=categories, random_state=42)
    train_files, val_files, train_categories, val_categories = train_test_split(
        train_val_files, train_val_categories,
        test_size=(data_split["val"] / (data_split["train"] + data_split["val"])), stratify=train_val_categories, random_state=42)
    return train_files, val_files, test_files, train_categories, val_categories, test_categories

def nested_cross_validation(all_txt_files, all_categories, data_split, processed_folder):
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    dataset_info_list = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(all_txt_files, all_categories)):
        train_val_files = [all_txt_files[i] for i in train_val_indices]
        test_files = [all_txt_files[i] for i in test_indices]
        train_val_categories = [all_categories[i] for i in train_val_indices]
        test_categories = [all_categories[i] for i in test_indices]

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for inner_fold_idx, (train_indices, val_indices) in enumerate(inner_cv.split(train_val_files, train_val_categories)):
            train_files = [train_val_files[i].split('\\')[-1] for i in train_indices]
            val_files = [train_val_files[i].split('\\')[-1] for i in val_indices]
            train_categories = [train_val_categories[i] for i in train_indices]
            val_categories = [train_val_categories[i] for i in val_indices]

            whole_train_files = [os.path.join(processed_folder, train_val_files[i]) for i in train_indices]
            # Generate mixup pairs for training set within each category
            augmented_files = []
            unique_categories = set(train_categories)
            for cat in unique_categories:
                cat_train_files = [tf for tf, tc in zip(train_files, train_categories) if tc == cat]
                train_file_pairs = list(combinations(cat_train_files, 2))
                augmented_files.extend(generate_mixup_filenames(train_file_pairs, cat, processed_folder))

            # Include original train files in the augmented files
            all_train_files = whole_train_files + augmented_files

            # Record the paths along with their corresponding category labels
            dataset_info = {
                "fold": fold_idx,
                "inner_fold": inner_fold_idx,
                "train": [{"input": tf,
                           "category": tf.split('\\')[-2]} for tf in all_train_files],  # Each pair generates two augmented files
                "validation": [{"input": os.path.join(processed_folder, cat, vf),
                                "category": cat} for vf, cat in zip(val_files, val_categories)],
                "test": [{"input": os.path.join(processed_folder, tef),
                          "category": cat} for tef, cat in zip(test_files, test_categories)]
            }

            dataset_info_list.append(dataset_info)

    return dataset_info_list

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    input_folder = config["dataset_raw"]
    processed_folder = config["dataset_processed"]
    data_split = config["data_split"]
    is_cross_validation = config["is_cross_validation"]
    output_json = "dataset_info.json"

    subfolders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    all_txt_files = []
    all_categories = []

    for subfolder in subfolders:
        txt_files = list_txt_files(os.path.join(input_folder, subfolder))
        all_txt_files.extend([os.path.join(subfolder, tf) for tf in txt_files])
        all_categories.extend([subfolder] * len(txt_files))

    if is_cross_validation:
        dataset_info_list = nested_cross_validation(all_txt_files, all_categories, data_split, processed_folder)
    else:
        train_files, val_files, test_files, train_categories, val_categories, test_categories = simple_data_split(all_txt_files, all_categories, data_split)

        # Generate mixup pairs for training set within each category
        augmented_files = []
        unique_categories = set(train_categories)
        for cat in unique_categories:
            cat_train_files = [tf for tf, tc in zip(train_files, train_categories) if tc == cat]
            train_file_pairs = list(combinations(cat_train_files, 2))
            augmented_files.extend(generate_mixup_filenames(train_file_pairs, cat, processed_folder))

        # Include original train files in the augmented files
        all_train_files = train_files

        # Record the paths along with their corresponding category labels
        dataset_info = {
            "train": [{"input": os.path.join(processed_folder, cat, tf),
                       "category": cat} for tf, cat in zip(all_train_files, train_categories * 2)],  # Each pair generates two augmented files
            "validation": [
                {"input": os.path.join(processed_folder, cat, vf),
                 "category": cat}
                for vf, cat in zip(val_files, val_categories)],
            "test": [
                {"input": os.path.join(processed_folder, cat, tef),
                 "category": cat}
                for tef, cat in zip(test_files, test_categories)]
        }

        dataset_info_list = [dataset_info]

    # Write to JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset_info_list, f, indent=4)

if __name__ == "__main__":
    config_file = '../../dataset_classify/config.json'  # Update this path if necessary
    main(config_file)



