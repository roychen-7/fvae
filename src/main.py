import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from model import FactorVAE
from loss import FactorVAELoss
from discriminator import Discriminator

# Parse command-line arguments
parser = argparse.ArgumentParser(description='FactorVAE Training')
parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to config file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Hyperparameters
input_channels = config['model']['input_channels']
latent_dim = config['model']['latent_dim']
output_channels = config['model']['output_channels']
gamma = config['training']['gamma']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
epochs = config['training']['num_epochs']

# Initialize model, loss, and discriminator
model = FactorVAE(input_channels, latent_dim, output_channels)
loss_fn = FactorVAELoss(gamma)
discriminator = Discriminator(latent_dim)

# Optimizers
optimizer_vae = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Create a simple synthetic dataset for demonstration
# This generates random images with the specified dimensions
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, channels=3, height=64, width=64):
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image data
        return torch.randn(self.channels, self.height, self.width)

# Create dataset and dataloader
# Model expects 28x28 images (to match the encoder architecture)
dataset = SyntheticDataset(num_samples=1000, channels=input_channels, height=28, width=28)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        # Forward pass
        x = batch  # Placeholder for input data
        recon_x, mu, logvar, z = model(x)

        # Compute loss
        loss, recon_loss, kl_div, tc_loss = loss_fn.compute_loss(recon_x, x, mu, logvar, z, discriminator)

        # Backward pass and optimization for VAE
        optimizer_vae.zero_grad()
        optimizer_disc.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_vae.step()

        # Update discriminator separately
        # The discriminator loss should be computed separately
        z_detached = z.detach()
        z_perm = z_detached[torch.randperm(z_detached.size(0))]
        logits_real = discriminator(z_detached)
        logits_perm = discriminator(z_perm)

        # Discriminator tries to distinguish real from permuted
        disc_loss = -torch.mean(logits_real - logits_perm)
        disc_loss.backward()
        optimizer_disc.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}, TC Loss: {tc_loss.item():.4f}')

# Save model checkpoint
import os
results_dir = '../results' if os.path.exists('../results') else 'results'
os.makedirs(results_dir, exist_ok=True)
checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pth')
torch.save(model.state_dict(), checkpoint_path)
print(f'\nModel checkpoint saved to: {checkpoint_path}')

# Note: This script is a template and requires actual dataset and configuration values to run properly.
# Ensure the dataset is loaded correctly and the config.yaml file is set up with the necessary parameters.
