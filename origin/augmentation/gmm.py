import os
import numpy as np
# from keras.src.backend import standardize_data_format
from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_tsf_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    x_coords = lines[0].strip().split()[1:]
    y_coords = [line.strip().split()[0] for line in lines[1:]]
    matrix_data = np.array([list(map(float, line.strip().split()[1:])) for line in lines[1:]])

    return np.array(x_coords).astype(float), np.array(y_coords).astype(float), matrix_data


def standardize_data(matrix_data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(matrix_data.reshape(-1, 1)).reshape(matrix_data.shape)
    return standardized_data, scaler


def train_gmm(standardized_data, n_components, max_iter, tol, init_params):
    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=max_iter, tol=tol,
                          init_params=init_params)
    gmm.fit(standardized_data.reshape(-1, 1))
    return gmm


def generate_new_spectra(gmm, shape, n_samples=1):
    new_samples = gmm.sample(n_samples * shape[0] * shape[1])[0].reshape(shape)
    # new_samples = scaler.inverse_transform(new_samples.reshape(-1, 1)).reshape(shape)
    return np.round(new_samples, 2)


def save_generated_data(output_path, x_coords, y_coords, generated_data, original_file_name):
    base_name = os.path.splitext(original_file_name)[0]
    output_file_path = os.path.join(output_path, f"{base_name}_generated.txt")

    with open(output_file_path, 'w') as file:
        header = [''] + list(map(str, x_coords))
        file.write(' '.join(header) + '\n')

        for i, y_coord in enumerate(y_coords):
            row = [str(y_coord)] + list(map(str, generated_data[i]))
            file.write(' '.join(row) + '\n')


def calculate_cosine_similarity(matrix1, matrix2):
    return cosine_similarity(matrix1.flatten().reshape(1, -1), matrix2.flatten().reshape(1, -1))[0][0]


def visualize_data(original_data, generated_data, x_coords, y_coords, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original data
    axes[0].imshow(original_data, aspect='auto',
                   extent=[x_coords.min(), x_coords.max(), y_coords.max(), y_coords.min()], cmap='viridis')
    axes[0].set_title(f'Original Data ({filename})')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')

    # Plot generated data
    axes[1].imshow(generated_data, aspect='auto',
                   extent=[x_coords.min(), x_coords.max(), y_coords.max(), y_coords.min()], cmap='viridis')
    axes[1].set_title(f'Generated Data ({filename})')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')

    plt.tight_layout()
    plt.show()


def process_folder(folder_path, output_path, n_components, max_iter, tol, init_params):
    total_cosine_similarity = 0
    num_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            x_coords, y_coords, matrix_data = read_tsf_matrix(file_path)

            # Standardize the data
            standardized_data, scaler = standardize_data(matrix_data)
            # standardized_data = matrix_data

            # Train GMM model_demo using EM algorithm with specified parameters
            gmm = train_gmm(standardized_data, n_components, max_iter, tol, init_params)

            # Generate new spectra data and round to 2 decimal places
            generated_data = generate_new_spectra(gmm, matrix_data.shape)

            # Save the generated data
            save_generated_data(output_path, x_coords, y_coords, generated_data, filename)

            # Calculate cosine similarity
            cos_sim = calculate_cosine_similarity(matrix_data, generated_data)
            total_cosine_similarity += cos_sim
            num_files += 1

            print(f"Processed {filename}, Cosine Similarity: {cos_sim}")

            # Visualize original and generated data
            visualize_data(matrix_data, generated_data, x_coords, y_coords, filename)

    average_cosine_similarity = total_cosine_similarity / num_files if num_files > 0 else 0
    print(f"Average Cosine Similarity: {average_cosine_similarity}")


# Example usage
folder_path = r'C:\Users\xiao\Desktop\画大饼环节\data\GMM_test\dataset_extract\C6 + hpts'
export_path = r'C:\Users\xiao\Desktop\画大饼环节\data\GMM_test\dataset_output'
process_folder(folder_path, export_path, n_components=6, max_iter=200, tol=1e-4, init_params='kmeans')
