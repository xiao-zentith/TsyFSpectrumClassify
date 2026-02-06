import json
import os

import numpy as np


def load_data(input_folder):
    features = []
    labels = []
    label_map = {}

    for label_idx, label in enumerate(os.listdir(input_folder)):
        label_map[label] = label_idx
        label_folder = os.path.join(input_folder, label)

        if not os.path.isdir(label_folder):
            continue

        for txt_file in os.listdir(label_folder):
            if not txt_file.endswith('.txt'):
                continue

            file_path = os.path.join(label_folder, txt_file)
            x_coords, y_coords, matrix = read_matrix_from_file(file_path)

            # Normalize the matrix values between 0 and 1
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

            features.append(matrix.flatten())  # Flatten the matrix to 1D
            labels.append(label_idx)

    X = np.array(features)
    y = np.array(labels)

    return X, y, label_map


def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    x_coords = list(map(float, lines[0].strip().split()))
    y_coords = []
    data_matrix = []
    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        y_coords.append(parts[0])
        data_matrix.append(parts[1:])

    return np.array(x_coords), np.array(y_coords), np.array(data_matrix)

def load_data_from_json(json_path, label_map):
    with open(json_path, 'r') as file:
        dataset_info = json.load(file)

    all_data_paths = []
    all_labels = []

    for fold_info in dataset_info:
        for category_idx, category in enumerate(label_map.keys()):
            train_files = [item['input'] for item in fold_info['train'] if item['category'] == category]
            val_files = [item['input'] for item in fold_info['validation'] if item['category'] == category]
            test_files = [item['input'] for item in fold_info['test'] if item['category'] == category]

            all_data_paths.extend(train_files + val_files + test_files)
            all_labels.extend([label_map[category]] * (len(train_files) + len(val_files) + len(test_files)))

    return all_data_paths, all_labels