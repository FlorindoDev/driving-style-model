import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torch.optim import Optimizer


# Device setup - utilizza GPU (CUDA/ROCm) se disponibile, altrimenti CPU
print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda.is_available (ROCm usa questa API):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class MaskedMSELoss(nn.MSELoss):
    """
    Custom MSE Loss che ignora i valori mascherati.
    Utile per gestire padding o valori mancanti nel dataset.
    """
    
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calcola MSE solo sui valori validi (mask == True).
        
        Args:
            input: Predicted values
            target: Ground truth values
            mask: Boolean mask (True = valid, False = ignored)
        """

        #includiamo nella loss  solo i parametri che ci interessano
        bool_mask = mask.bool()
        masked_input = input[bool_mask]
        masked_target = target[bool_mask]
        
        return super(MaskedMSELoss, self).forward(masked_input, masked_target)


class AutoEncoder(nn.Module):
    """
    Autoencoder per la compressione e ricostruzione di dati di guida.
    
    L'encoder comprime l'input in uno spazio latente ridotto,
    il decoder ricostruisce l'input originale dal codice latente.
    """
    
    def __init__(self, input_dim: int = 455, latent_dim: int = 32):
        """
        Args:
            input_dim: Dimensione dell'input
            latent_dim: Dimensione dello spazio latente compresso
        """
        super().__init__()
        self.losses: List[float] = []
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss_function = MaskedMSELoss().to(device)

        # Encoder: comprime x -> z (latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder: ricostruisce z -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x: torch.Tensor, encoder_only: bool = False) -> torch.Tensor:
        """
        Forward pass dell'autoencoder.
        
        Args:
            x: Input tensor
            encoder_only: Se True, ritorna solo il codice latente
        
        Returns:
            Latent code (se encoder_only=True) o ricostruzione completa
        """
        latent_code = self.encoder(x)
        
        if encoder_only:
            return latent_code
        
        reconstruction = self.decoder(latent_code)
        return reconstruction

    def train_model(
        self,
        optimizer: Optimizer,
        epochs: int,
        train_data: np.ndarray,
        mask: np.ndarray,
        batch_size: int = 32
    ) -> None:
        """
        Training loop dell'autoencoder.
        
        Args:
            optimizer: PyTorch optimizer
            epochs: Numero di epoche di training
            train_data: Dati di training (numpy array)
            mask: Mask per ignorare valori di padding
            batch_size: Dimensione dei batch
        """
        self.losses = []

        for epoch in range(epochs):
            print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)
            total_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                # Estrai batch corrente
                batch_input = train_data[i:i + batch_size]
                batch_mask = mask[i:i + batch_size]

                # Converti a tensori e sposta su device
                batch_input = torch.tensor(batch_input, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

                # Standard training step
                optimizer.zero_grad()
                reconstruction = self.forward(batch_input)
                loss = self.loss_function(reconstruction, batch_input, batch_mask)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Calcola e salva loss media dell'epoca
            avg_loss = total_loss / num_batches
            print(f"Training loss: {avg_loss:.6f}")
            self.losses.append(avg_loss)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut per ottenere solo il codice latente."""
        return self.forward(x, encoder_only=True)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica un codice latente."""
        return self.decoder(z)



   

