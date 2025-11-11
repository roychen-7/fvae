import torch
import torch.nn.functional as F

class FactorVAELoss:
    def __init__(self, gamma):
        """
        Initialize the FactorVAE loss function.
        :param gamma: Weight for the Total Correlation (TC) penalty.
        """
        self.gamma = gamma

    def compute_loss(self, recon_x, x, mu, logvar, z, discriminator):
        """
        Compute the total loss for FactorVAE.
        :param recon_x: Reconstructed images.
        :param x: Original images.
        :param mu: Mean of the latent Gaussian.
        :param logvar: Log variance of the latent Gaussian.
        :param z: Latent variable samples.
        :param discriminator: Discriminator model for TC estimation.
        :return: Total loss, reconstruction loss, KL divergence, TC penalty.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total Correlation (TC) penalty
        tc_penalty = self.total_correlation(z, discriminator)

        # Total loss
        total_loss = recon_loss + kl_divergence + self.gamma * tc_penalty

        return total_loss, recon_loss, kl_divergence, tc_penalty

    def total_correlation(self, z, discriminator):
        """
        Estimate the Total Correlation (TC) using the discriminator.
        :param z: Latent variable samples.
        :param discriminator: Discriminator model.
        :return: Estimated TC penalty.
        """
        # Permute the latent variables to break correlations
        z_perm = z[torch.randperm(z.size(0))]

        # Discriminator predictions
        logits_real = discriminator(z)
        logits_perm = discriminator(z_perm)

        # TC estimation
        tc_estimate = (logits_real - logits_perm).mean()

        return tc_estimate
