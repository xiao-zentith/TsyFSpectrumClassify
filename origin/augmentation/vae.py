import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TSFDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, *self.output_shape), mu, logvar


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


def inverse_standardize_data(scaled_data, scaler):
    inversed_data = scaler.inverse_transform(scaled_data.reshape(-1, 1)).reshape(scaled_data.shape)
    return inversed_data


def build_vae(input_dim, hidden_dim1, hidden_dim2, latent_dim):
    vae = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
    return vae


def train_vae(vae, dataloader, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            mse = mse_loss(recon_batch, data)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader.dataset)}')


def generate_new_spectra(vae, shape, n_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.eval()
    noise = torch.randn(n_samples, vae.latent_dim).to(device)
    with torch.no_grad():
        generated_data = vae.decode(noise).cpu().numpy().reshape(shape)
    return np.round(generated_data, 2)


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


def process_folder(folder_path, output_path, hidden_dim1, hidden_dim2, latent_dim, epochs, batch_size):
    total_cosine_similarity = 0
    num_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            x_coords, y_coords, matrix_data = read_tsf_matrix(file_path)

            # Standardize the data
            standardized_data, scaler = standardize_data(matrix_data)

            # Build and train VAE model_demo
            input_dim = matrix_data.size
            vae = build_vae(input_dim, hidden_dim1, hidden_dim2, latent_dim)
            dataset = TSFDataset(standardized_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_vae(vae, dataloader, epochs=epochs, lr=1e-3)

            # Generate new spectra data and round to 2 decimal places
            generated_scaled_data = generate_new_spectra(vae, matrix_data.shape, n_samples=1)[0]
            generated_data = inverse_standardize_data(generated_scaled_data, scaler)

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
process_folder(folder_path, export_path, hidden_dim1=64, hidden_dim2=32, latent_dim=8, epochs=100, batch_size=32)



