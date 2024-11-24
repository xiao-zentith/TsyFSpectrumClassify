import os
import numpy as np


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract x coordinates from the first line
    x_coords = list(map(float, lines[0].strip().split()))

    # Extract y coordinates and matrix values
    data = []
    for line in lines[1:]:
        row = list(map(float, line.strip().split()))
        data.append(row)

    y_coords = [row[0] for row in data]
    matrix_values = [row[1:] for row in data]

    return x_coords, y_coords, np.array(matrix_values)


def add_noise_to_matrix(matrix, sigma=0.02):
    noise = np.random.normal(0, sigma, matrix.shape)
    noisy_matrix = matrix + (matrix * noise)
    return noisy_matrix


def save_noisy_matrix_to_file(x_coords, y_coords, noisy_matrix, output_file_path):
    with open(output_file_path, 'w') as file:
        # Write x coordinates to the first line
        file.write(' '.join(map(str, x_coords)) + '\n')

        # Write y coordinates and noisy matrix values with 4 decimal places
        for y_coord, row in zip(y_coords, noisy_matrix):
            formatted_row = ' '.join(f"{val:.4f}" for val in row)
            file.write(f"{y_coord} {formatted_row}\n")


def process_files_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)

        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        for file_name in files:
            if file_name.endswith('.txt'):
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(current_output_dir, file_name)

                x_coords, y_coords, matrix_values = read_data_from_file(input_file_path)

                print(f"Processing file: {input_file_path}")

                noisy_matrix = add_noise_to_matrix(matrix_values)

                save_noisy_matrix_to_file(x_coords, y_coords, noisy_matrix, output_file_path)
                print(f"Noisy matrix saved to {output_file_path}")


input_dir = r'C:\Users\xiao\Desktop\论文汇总\data\dataset_to_TsyF'
output_dir = r'C:\Users\xiao\Desktop\论文汇总\data\dataset_after_noise'

process_files_in_directory(input_dir, output_dir)






