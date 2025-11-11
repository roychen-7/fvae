import torch
import torch.nn as nn
import torch.nn.functional as F


class FinancialEncoder(nn.Module):
    """Encoder for 1D financial data (alpha158/360 features)"""
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64]):
        super(FinancialEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent space projection
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            mu, logvar: Mean and log variance of the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class FinancialDecoder(nn.Module):
    """Decoder for 1D financial data"""
    def __init__(self, latent_dim, output_dim, hidden_dims=[64, 128]):
        super(FinancialDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Output layer
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: Latent representation of shape (batch_size, latent_dim)

        Returns:
            recon_x: Reconstructed features of shape (batch_size, output_dim)
        """
        h = self.decoder(z)
        recon_x = self.fc_out(h)
        return recon_x


class FinancialFactorVAE(nn.Module):
    """
    FactorVAE model for financial data (alpha158/360 features).
    This model learns disentangled representations of financial factors.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64]):
        super(FinancialFactorVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = FinancialEncoder(input_dim, latent_dim, hidden_dims)
        decoder_hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        self.decoder = FinancialDecoder(latent_dim, input_dim, decoder_hidden_dims)

    def forward(self, x):
        """
        Forward pass of the FactorVAE.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            recon_x: Reconstructed features
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent representation
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var).

        Args:
            mu: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian

        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encode input to latent representation.

        Args:
            x: Input features

        Returns:
            mu: Mean of latent distribution
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        """
        Decode latent representation to reconstructed features.

        Args:
            z: Latent representation

        Returns:
            Reconstructed features
        """
        return self.decoder(z)
