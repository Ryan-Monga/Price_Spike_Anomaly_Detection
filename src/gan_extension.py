"""
GAN Extension Module (Experimental).
Implements a genuine 1D sequence Generative Adversarial Network (GAN)
for synthetic financial sequence generation and training data augmentation experiments.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import GANConfig, CORE_FEATURE_COLUMNS, MODELS_DIR

logger = logging.getLogger(__name__)

class SequenceGenerator(nn.Module):
    """
    Generator: maps latent noise z into synthetic multi-feature time series sequence.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, seq_len: int, num_features: int):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, num_features),
            nn.Tanh()  # Produces bounded standardized feature values
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        h0 = torch.relu(self.fc_in(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(h0)
        seq = self.fc_out(out)  # (batch, seq_len, num_features)
        return seq

class SequenceDiscriminator(nn.Module):
    """
    Discriminator: distinguishes real multi-feature time-series sequences from synthetic ones.
    """
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        out, hn = self.gru(x)
        # Use last hidden state
        prob = self.classifier(hn[-1])  # (batch, 1)
        return prob

class FinancialSequenceGAN:
    """
    Experimental GAN for synthesizing and augmenting financial sequence baselines.
    """
    def __init__(self, config: Optional[GANConfig] = None, device: Optional[str] = None):
        self.config = config or GANConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.generator = SequenceGenerator(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            seq_len=self.config.sequence_length,
            num_features=self.config.num_features
        ).to(self.device)

        self.discriminator = SequenceDiscriminator(
            seq_len=self.config.sequence_length,
            num_features=self.config.num_features,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        self.is_trained = False
        self.history: Dict[str, List[float]] = {"d_loss": [], "g_loss": []}
        self.metadata: Dict[str, Any] = {}

    def fit(self, real_sequences: np.ndarray) -> "FinancialSequenceGAN":
        """
        Trains GAN on real baseline sequences.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        dataset = TensorDataset(torch.tensor(real_sequences, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        criterion = nn.BCELoss()
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.999))

        logger.info(f"Training experimental Sequence GAN for {self.config.epochs} epochs...")
        self.history = {"d_loss": [], "g_loss": []}

        for epoch in range(1, self.config.epochs + 1):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            batches = 0

            for (real_batch,) in loader:
                batch_size = real_batch.size(0)
                real_batch = real_batch.to(self.device)

                # Real and fake labels
                real_labels = torch.ones((batch_size, 1), device=self.device) * 0.9  # label smoothing
                fake_labels = torch.zeros((batch_size, 1), device=self.device)

                # Train Discriminator
                opt_d.zero_grad()
                pred_real = self.discriminator(real_batch)
                loss_d_real = criterion(pred_real, real_labels)

                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                pred_fake = self.discriminator(fake_batch.detach())
                loss_d_fake = criterion(pred_fake, fake_labels)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                # Train Generator
                opt_g.zero_grad()
                pred_g_fake = self.discriminator(fake_batch)
                loss_g = criterion(pred_g_fake, real_labels)  # Trick D into thinking fake is real
                loss_g.backward()
                opt_g.step()

                d_loss_epoch += loss_d.item()
                g_loss_epoch += loss_g.item()
                batches += 1

            if batches > 0:
                self.history["d_loss"].append(d_loss_epoch / batches)
                self.history["g_loss"].append(g_loss_epoch / batches)

        self.is_trained = True
        self.metadata = {
            "latent_dim": self.config.latent_dim,
            "hidden_dim": self.config.hidden_dim,
            "sequence_length": self.config.sequence_length,
            "num_features": self.config.num_features,
            "epochs": self.config.epochs,
            "seed": self.config.seed,
            "limitations": (
                "EXPERIMENTAL NOTICE: Financial time series exhibit heavy tails, non-stationarity, "
                "and regime shifts. GAN-generated sequences provide synthetic baseline diversity but "
                "should NOT be treated as genuine historical price observations."
            )
        }
        return self

    def generate_synthetic_sequences(self, num_samples: int = 50) -> np.ndarray:
        """
        Generates synthetic multi-feature sequences.
        Checks for numerical validity and finite values.
        """
        if not self.is_trained:
            raise RuntimeError("GAN must be trained before generating synthetic sequences.")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()

        # Sanity check: Ensure finite and plausible values
        if np.any(np.isnan(synthetic)) or np.any(np.isinf(synthetic)):
            raise ValueError("GAN generated non-finite (NaN or Inf) values.")

        return synthetic
