import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from model_financial import FinancialFactorVAE
from loss import FactorVAELoss
from discriminator import Discriminator
from qlib_data_loader import get_qlib_dataloader


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FactorVAE Training with Qlib Data')
    parser.add_argument('--config', type=str, default='config/config_qlib.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='alpha158', choices=['alpha158', 'alpha360'],
                       help='Dataset to use (alpha158 or alpha360)')
    parser.add_argument('--start_time', type=str, default='2015-01-01', help='Start date for training data')
    parser.add_argument('--end_time', type=str, default='2020-12-31', help='End date for training data')
    parser.add_argument('--instruments', type=str, default='csi300', help='Market instruments')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Hyperparameters
    latent_dim = config['model']['latent_dim']
    hidden_dims = config['model'].get('hidden_dims', [128, 64])
    gamma = config['training']['gamma']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']

    # Get qlib dataloader
    print(f"Loading {args.dataset} dataset from {args.start_time} to {args.end_time}...")
    num_stocks = config.get('dataset', {}).get('num_stocks', 100)
    train_loader, num_features = get_qlib_dataloader(
        dataset_name=args.dataset,
        start_time=args.start_time,
        end_time=args.end_time,
        batch_size=batch_size,
        num_stocks=num_stocks
    )

    print(f"Number of features: {num_features}")

    # Initialize model, loss, and discriminator
    model = FinancialFactorVAE(input_dim=num_features, latent_dim=latent_dim, hidden_dims=hidden_dims)
    loss_fn = FactorVAELoss(gamma)
    discriminator = Discriminator(latent_dim)

    # Optimizers
    optimizer_vae = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_div = 0
        epoch_tc_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Forward pass
            x = batch

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

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_div += kl_div.item()
            epoch_tc_loss += tc_loss.item()
            num_batches += 1

        # Print epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_div = epoch_kl_div / num_batches
        avg_tc_loss = epoch_tc_loss / num_batches

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, KL Div: {avg_kl_div:.4f}, TC Loss: {avg_tc_loss:.4f}')

    # Save model checkpoint and training results
    results_dir = 'results/fin_res'
    os.makedirs(results_dir, exist_ok=True)

    # Save model checkpoint
    checkpoint_path = os.path.join(results_dir, f'model_checkpoint_{args.dataset}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'num_features': num_features,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'config': config,
        'dataset': args.dataset,
        'start_time': args.start_time,
        'end_time': args.end_time
    }, checkpoint_path)
    print(f'\nModel checkpoint saved to: {checkpoint_path}')

    # Save model architecture summary
    summary_path = os.path.join(results_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"FactorVAE Model Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Time Range: {args.start_time} to {args.end_time}\n")
        f.write(f"Number of Features: {num_features}\n")
        f.write(f"Latent Dimension: {latent_dim}\n")
        f.write(f"Hidden Dimensions: {hidden_dims}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Gamma (TC weight): {gamma}\n\n")
        f.write(f"Model:\n{model}\n")
    print(f'Model summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
