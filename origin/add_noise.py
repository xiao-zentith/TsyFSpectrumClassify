
import numpy as np

def read_matrix_from_file(file_path):
    """Read a matrix from a txt file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrix = [list(map(float, line.strip().split())) for line in lines]
    return np.array(matrix)

def add_noise_to_matrix(matrix, sigma=0.02):
    """Add noise to the matrix using the formula y = (1 + norm(0, σ))x."""
    noise = np.random.normal(0, sigma, matrix.shape)
    noisy_matrix = (1 + noise) * matrix
    return noisy_matrix

# Example usage
file_path = 'matrix.txt'  # Replace with your file path
matrix = read_matrix_from_file(file_path)
noisy_matrix = add_noise_to_matrix(matrix)

print("Original Matrix:")
print(matrix)
print("\nNoisy Matrix:")
print(noisy_matrix)