import os
import numpy as np
import shutil


class NoiseProcessor:
    def __init__(self, input_folder, output_folder, seed=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if seed is not None:
            np.random.seed(seed)

    def read_data_from_file(self, file_path):
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

    def add_noise_to_matrix(self, matrix, sigma=0.02):
        noise = np.random.normal(0, sigma, matrix.shape)
        noisy_matrix = matrix + (matrix * noise)
        return noisy_matrix

    def save_noisy_matrix_to_file(self, x_coords, y_coords, noisy_matrix, output_file_path):
        with open(output_file_path, 'w') as file:
            # Write x coordinates to the first line
            file.write(' '.join(map(str, x_coords)) + '\n')

            # Write y coordinates and noisy matrix values with 4 decimal places
            for y_coord, row in zip(y_coords, noisy_matrix):
                formatted_row = ' '.join(f"{val:.4f}" for val in row)
                file.write(f"{y_coord} {formatted_row}\n")

    def copy_original_file(self, input_file_path, current_output_dir):
        file_name = os.path.basename(input_file_path)
        destination_path = os.path.join(current_output_dir, file_name)
        shutil.copy(input_file_path, destination_path)

    def process_files_in_directory(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for root, dirs, files in os.walk(self.input_folder):
            relative_path = os.path.relpath(root, self.input_folder)
            current_output_dir = os.path.join(self.output_folder, relative_path)

            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

            for file_name in files:
                if file_name.endswith('.txt'):
                    input_file_path = os.path.join(root, file_name)
                    output_file_path = os.path.join(current_output_dir, 'noise_' + file_name)

                    x_coords, y_coords, matrix_values = self.read_data_from_file(input_file_path)

                    # print(f"Processing file: {input_file_path}")

                    noisy_matrix = self.add_noise_to_matrix(matrix_values)

                    self.save_noisy_matrix_to_file(x_coords, y_coords, noisy_matrix, output_file_path)
                    # print(f"Noisy matrix saved to {output_file_path}")

                    # Copy the original file to the output directory
                    self.copy_original_file(input_file_path, current_output_dir)


# Example usage:
input_path = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\EEM_mixup'
output_path = r'C:\Users\xiao\Desktop\画大饼环节\data\dataset_EEM\EEM_noise'
processor = NoiseProcessor(input_path, output_path, seed=42)
processor.process_files_in_directory()



