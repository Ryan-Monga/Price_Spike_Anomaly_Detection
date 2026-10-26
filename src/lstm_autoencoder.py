"""
PyTorch LSTM Autoencoder Module for Temporal Anomaly Detection.
Encodes multi-feature rolling sequences into a latent representation and computes
reconstruction error as a temporal anomaly indicator.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import LSTMAutoencoderConfig, CORE_FEATURE_COLUMNS, MODELS_DIR
from src.preprocessing import FeatureScaler, create_causal_sequences

logger = logging.getLogger(__name__)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use final hidden state of the top layer
        last_hidden = hn[-1]  # shape: (batch, hidden_dim)
        latent = self.fc_latent(last_hidden)  # shape: (batch, latent_dim)
        return latent, last_hidden

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        # latent shape: (batch, latent_dim)
        init_repr = self.fc_hidden(latent)  # shape: (batch, hidden_dim)
        # Repeat representation across all time-steps as decoder input
        decoder_input = init_repr.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.lstm(decoder_input)
        reconstruction = self.fc_out(lstm_out)  # shape: (batch, seq_len, output_dim)
        return reconstruction

class LSTMAutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent, seq_len)
        return reconstructed

class LSTMAutoencoderDetector:
    """
    Manages end-to-end LSTM Autoencoder training, inference, and persistence.
    """
    def __init__(
        self,
        config: Optional[LSTMAutoencoderConfig] = None,
        feature_cols: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        self.config = config or LSTMAutoencoderConfig()
        self.feature_cols = feature_cols if feature_cols is not None else CORE_FEATURE_COLUMNS
        self.scaler = FeatureScaler(scaler_type="robust", feature_cols=self.feature_cols)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model: Optional[LSTMAutoencoderNet] = None
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.training_metadata: Dict[str, Any] = {}
        self.score_threshold: float = 0.5  # calibrated during validation
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> "LSTMAutoencoderDetector":
        """
        Trains the LSTM Autoencoder on training observations with validation tracking.
        """
        self._set_seed(self.config.seed)
        
        # Fit scaler on training set strictly
        train_scaled = self.scaler.fit_transform(train_df)
        
        seq_len = self.config.sequence_length
        if len(train_scaled) < seq_len + 5:
            raise ValueError(
                f"Training set size ({len(train_scaled)}) is too small for sequence length ({seq_len})."
            )

        X_train_seq, _ = create_causal_sequences(train_scaled, train_df.index, seq_len)
        
        train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )

        has_val = val_df is not None and len(val_df) >= seq_len
        val_loader = None
        val_dataset = None
        if has_val:
            val_scaled = self.scaler.transform(val_df)
            X_val_seq, _ = create_causal_sequences(val_scaled, val_df.index, seq_len)
            val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Initialize network
        num_features = len(self.feature_cols)
        self.model = LSTMAutoencoderNet(
            input_dim=num_features,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.training_history = {"train_loss": [], "val_loss": []}

        logger.info(f"Training LSTM Autoencoder on {self.device} for {self.config.epochs} epochs...")
        self.model.train()

        for epoch in range(1, self.config.epochs + 1):
            total_train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            epoch_train_loss = total_train_loss / len(train_dataset)
            self.training_history["train_loss"].append(epoch_train_loss)

            epoch_val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for (batch_val_x,) in val_loader:
                        batch_val_x = batch_val_x.to(self.device)
                        recon_val = self.model(batch_val_x)
                        v_loss = criterion(recon_val, batch_val_x)
                        val_loss_sum += v_loss.item() * len(batch_val_x)
                epoch_val_loss = val_loss_sum / len(val_dataset)
                self.training_history["val_loss"].append(epoch_val_loss)
                self.model.train()

            if progress_callback is not None:
                progress_callback(epoch, self.config.epochs, epoch_train_loss, epoch_val_loss)

        self.is_trained = True

        # Compute calibration range from training reconstruction errors
        train_recon_errors = self._compute_reconstruction_errors(X_train_seq)
        self._score_min = float(np.percentile(train_recon_errors, 2))
        self._score_max = float(np.percentile(train_recon_errors, 98))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1e-4

        # Threshold set to 95th percentile of normal baseline reconstruction errors
        self.score_threshold = float(np.percentile(train_recon_errors, 95))

        self.training_metadata = {
            "device": str(self.device),
            "sequence_length": self.config.sequence_length,
            "hidden_dim": self.config.hidden_dim,
            "latent_dim": self.config.latent_dim,
            "num_layers": self.config.num_layers,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "train_observations": len(train_df),
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "score_min": self._score_min,
            "score_max": self._score_max,
            "score_threshold": self.score_threshold
        }

        return self

    def _compute_reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Computes mean squared error per sequence."""
        self.model.eval()
        dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        errors = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)
                # MSE per sequence across all time steps and features
                diff = (recon - batch_x) ** 2
                seq_errors = diff.mean(dim=(1, 2)).cpu().numpy()
                errors.append(seq_errors)
                
        return np.concatenate(errors, axis=0)

    def predict_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Runs LSTM reconstruction inference on rolling sequences constructed from df.
        
        Returns:
            calibrated_scores: Array in [0, 1] aligned to target_timestamps.
            flags: Boolean array (True = anomaly).
            target_timestamps: DatetimeIndex matching the final observation of each sequence.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("LSTM Autoencoder is not trained.")

        scaled = self.scaler.transform(df)
        seq_len = self.config.sequence_length
        if len(scaled) < seq_len:
            raise ValueError(
                f"DataFrame has {len(scaled)} rows, fewer than required sequence length {seq_len}."
            )

        sequences, target_timestamps = create_causal_sequences(scaled, df.index, seq_len)
        raw_errors = self._compute_reconstruction_errors(sequences)

        # Calibrate error to [0, 1]
        calibrated = (raw_errors - self._score_min) / (self._score_max - self._score_min + 1e-8)
        calibrated_scores = np.clip(calibrated, 0.0, 1.0).astype(np.float64)

        # Flagged if raw reconstruction error exceeds training baseline threshold
        flags = raw_errors >= self.score_threshold

        return calibrated_scores, flags, target_timestamps

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments input DataFrame with lstm_score and lstm_anomaly.
        Observations prior to sequence_length warmup are filled with 0.0 / False.
        """
        df_out = df.copy()
        seq_len = self.config.sequence_length
        if len(df_out) < seq_len:
            df_out["lstm_score"] = 0.0
            df_out["lstm_anomaly"] = False
            return df_out

        scores, flags, timestamps = self.predict_sequences(df_out)

        # Align with original DataFrame index
        score_series = pd.Series(scores, index=timestamps)
        flag_series = pd.Series(flags, index=timestamps)

        # Backfill sequence scores to warmup prefix so early rows aren't artificially zero-diluted,
        # while keeping flags strictly False for un-warmed rows.
        mapped_scores = pd.Series(df_out.index.map(score_series), index=df_out.index)
        filled_scores = mapped_scores.bfill().fillna(0.0).astype(float)
        df_out["lstm_score"] = filled_scores
        df_out["lstm_anomaly"] = df_out.index.map(flag_series).fillna(False).astype(bool)

        return df_out

    def save(self, filepath: Path) -> Path:
        """Saves model weights, scaler, config, and metadata to disk."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save untrained LSTM model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "config": self.config,
            "feature_cols": self.feature_cols,
            "training_history": self.training_history,
            "training_metadata": self.training_metadata,
            "score_threshold": self.score_threshold,
            "_score_min": self._score_min,
            "_score_max": self._score_max
        }
        torch.save(payload, filepath)
        logger.info(f"Saved LSTM Autoencoder model to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "LSTMAutoencoderDetector":
        """Loads model weights, scaler, and metadata from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        try:
            payload = torch.load(filepath, map_location="cpu", weights_only=False)
            detector = cls(
                config=payload["config"],
                feature_cols=payload["feature_cols"],
                device=device
            )
            detector.scaler = payload["scaler"]
            detector.training_history = payload.get("training_history", {"train_loss": [], "val_loss": []})
            detector.training_metadata = payload.get("training_metadata", {})
            detector.score_threshold = payload.get("score_threshold", 0.5)
            detector._score_min = payload.get("_score_min", 0.0)
            detector._score_max = payload.get("_score_max", 1.0)

            # Rebuild architecture
            num_features = len(detector.feature_cols)
            detector.model = LSTMAutoencoderNet(
                input_dim=num_features,
                hidden_dim=detector.config.hidden_dim,
                latent_dim=detector.config.latent_dim,
                num_layers=detector.config.num_layers,
                dropout=detector.config.dropout
            ).to(detector.device)

            detector.model.load_state_dict(payload["model_state"])
            detector.model.eval()
            detector.is_trained = True
            logger.info(f"Successfully loaded LSTM Autoencoder from {filepath}")
            return detector
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM Autoencoder from {filepath}: {e}")
