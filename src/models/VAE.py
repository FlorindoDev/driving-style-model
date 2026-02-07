import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Tuple
from torch.optim import Optimizer


# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class VAE(nn.Module):
    """
    Variational Autoencoder per la compressione di dati di guida.
    
    A differenza di un autoencoder standard, il VAE:
    - Codifica l'input in una distribuzione (μ, σ) invece che un vettore fisso
    - Campiona z dalla distribuzione usando il "reparameterization trick"
    - Aggiunge una KL divergence loss per regolarizzare lo spazio latente
    """
    
    def __init__(
        self, 
        input_dim: int = 356, 
        latent_dim: int = 24, 
        alpha_lrelu: float = 0.01,
        dropout_rate: float = 0.1,
        beta: float = 0.001  # Peso della KL divergence (β-VAE)
    ):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.to(device)
        
        # Encoder: estrae features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
        )
        
        # Layers per μ e log(σ²)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: ricostruisce l'input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            
            nn.Linear(256, input_dim),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica l'input in μ e log(σ²).
        
        Returns:
            mu: Media della distribuzione latente
            logvar: Log-varianza della distribuzione latente
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε
        
        Questo permette di backpropagare attraverso il campionamento stocastico.
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)     # ε ~ N(0, 1)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica un vettore latente."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo.
        
        Returns:
            x_hat: Ricostruzione dell'input
            mu: Media della distribuzione latente
            logvar: Log-varianza della distribuzione latente
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calcola la KL divergence tra q(z|x) e p(z) = N(0, 1).
        
        Formula: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def loss_function(
        self, 
        x_hat: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola la loss totale del VAE.
        
        Returns:
            total_loss: reconstruction_loss + β * kl_loss
            recon_loss: MSE della ricostruzione
            kl_loss: KL divergence
        """
        # Masked MSE reconstruction loss
        if mask is not None:
            bool_mask = mask.bool()
            recon_loss = nn.functional.mse_loss(x_hat[bool_mask], x[bool_mask])
        else:
            recon_loss = nn.functional.mse_loss(x_hat, x)
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def _run_epoch(
        self, 
        data: np.ndarray, 
        mask: np.ndarray, 
        batch_size: int,
        optimizer: Optimizer = None,
        is_training: bool = True
    ) -> Tuple[float, float, float]:
        """
        Esegue un'epoca di training o validazione.
        
        Returns:
            (total_loss, recon_loss, kl_loss) medie dell'epoca
        """
        if is_training:
            self.train()
        else:
            self.eval()
            
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0
        
        context = torch.no_grad() if not is_training else torch.enable_grad()
        
        with context:
            for i in range(0, len(data), batch_size):
                batch_input = data[i:i + batch_size]
                batch_mask = mask[i:i + batch_size]

                batch_input = torch.tensor(batch_input, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

                if is_training:
                    optimizer.zero_grad()
                    
                x_hat, mu, logvar = self.forward(batch_input)
                total_loss, recon_loss, kl_loss = self.loss_function(
                    x_hat, batch_input, mu, logvar, batch_mask
                )
                
                if is_training:
                    total_loss.backward()
                    optimizer.step()

                total_loss_sum += total_loss.item()
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                num_batches += 1

        return (
            total_loss_sum / num_batches,
            recon_loss_sum / num_batches,
            kl_loss_sum / num_batches
        )

    def train_model(
        self,
        optimizer: Optimizer,
        epochs: int,
        train_data: np.ndarray,
        mask: np.ndarray,
        val_data: np.ndarray = None,
        val_mask: np.ndarray = None,
        batch_size: int = 512,
        patience: int = 15,
        min_delta: float = 1e-5
    ) -> None:
        """
        Training loop del VAE con early stopping.
        """
        self.losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        best_weights = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)
            
            # Training
            train_total, train_recon, train_kl = self._run_epoch(
                train_data, mask, batch_size, optimizer, is_training=True
            )
            print(f"Train - Total: {train_total:.6f} | Recon: {train_recon:.6f} | KL: {train_kl:.6f}")
            self.losses.append(train_total)
            
            # Validation e early stopping
            if val_data is not None:
                val_total, val_recon, val_kl = self._run_epoch(
                    val_data, val_mask, batch_size, is_training=False
                )
                print(f"Val   - Total: {val_total:.6f} | Recon: {val_recon:.6f} | KL: {val_kl:.6f}")
                self.val_losses.append(val_total)
                
                # Check improvement
                if val_total < best_val_loss - min_delta:
                    print(f"  ✓ Miglioramento! (↓{best_val_loss - val_total:.6f})")
                    best_val_loss = val_total
                    best_weights = copy.deepcopy(self.state_dict())
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"  ✗ Patience: {epochs_without_improvement}/{patience}")
                    
                    if epochs_without_improvement >= patience:
                        print(f"\nEarly stopping! Best epoch: {best_epoch}")
                        break
        
        # Ripristina i pesi migliori
        if best_weights is not None:
            self.load_state_dict(best_weights)
            print(f"\nRipristinati pesi dell'epoca {best_epoch} (val_loss: {best_val_loss:.6f})")
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ottiene il vettore latente z per inference.
        Durante l'inference, usa solo μ (senza campionamento).
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu  # Per inference, usa la media come z