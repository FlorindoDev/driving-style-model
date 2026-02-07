import torch
import torch.nn as nn
import numpy as np
import copy
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
    
    def __init__(self, input_dim: int = 455, latent_dim: int = 32, alpha_lrelu: float = 0.01, dropout_rate: int = 0.1):
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
        self.to(device)

        # #Encoder: comprime x -> z (latent space)
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 192),
        #     nn.BatchNorm1d(192),
        #     nn.LeakyReLU(alpha_lrelu, inplace=True),
        #     nn.Linear(192, 96),
        #     nn.BatchNorm1d(96),
        #     nn.LeakyReLU(alpha_lrelu, inplace=True),
        #     nn.Linear(96, latent_dim),
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 96),
        #     nn.BatchNorm1d(96),
        #     nn.LeakyReLU(alpha_lrelu, inplace=True),

        #     nn.Linear(96, 192),
        #     nn.BatchNorm1d(192),
        #     nn.LeakyReLU(alpha_lrelu, inplace=True),

        #     nn.Linear(192, input_dim)
        # )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            nn.Dropout(dropout_rate),  # Regolarizzazione aggiuntiva
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha_lrelu, inplace=True),
            
            nn.Linear(64, latent_dim),  # Ultimo layer senza attivazione!
        )
    
        # Decoder simmetrico
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass dell'autoencoder.
        
        Args:
            x: Input tensor
            encoder_only: Se True, ritorna solo il codice latente
        
        Returns:
            Latent code (se encoder_only=True) o ricostruzione completa
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _run_epoch(
        self, 
        data: np.ndarray, 
        mask: np.ndarray, 
        batch_size: int,
        optimizer: Optimizer = None,
        is_training: bool = True
    ) -> float:
        """
        Esegue un'epoca di training o validazione.
        
        Args:
            data: Dati da processare
            mask: Mask per i dati
            batch_size: Dimensione dei batch
            optimizer: Optimizer (richiesto solo per training)
            is_training: Se True esegue training, altrimenti validazione
        
        Returns:
            Loss media dell'epoca
        """
        if is_training:
            self.train()
        else:
            self.eval()
            
        total_loss = 0.0
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
                    
                reconstruction = self.forward(batch_input)
                loss = self.loss_function(reconstruction, batch_input, batch_mask)
                
                if is_training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _check_early_stopping(
        self, 
        val_loss: float, 
        best_val_loss: float, 
        epochs_without_improvement: int,
        patience: int, 
        min_delta: float
    ) -> Tuple[float, int, bool]:
        """
        Controlla se attivare early stopping.
        
        Returns:
            (best_val_loss, epochs_without_improvement, should_stop)
        """
        if val_loss < best_val_loss - min_delta:
            improvement = best_val_loss - val_loss
            print(f"  ✓ Miglioramento! Loss: {val_loss:.6f} (↓{improvement:.6f}) - Patience resettata")
            return val_loss, 0, False
        
        epochs_without_improvement += 1
        print(f"  ✗ Nessun miglioramento - Patience: {epochs_without_improvement}/{patience} (best: {best_val_loss:.6f})")
        
        should_stop = epochs_without_improvement >= patience
        return best_val_loss, epochs_without_improvement, should_stop

    def train_model(
        self,
        optimizer: Optimizer,
        epochs: int,
        train_data: np.ndarray,
        mask: np.ndarray,
        val_data: np.ndarray = None,
        val_mask: np.ndarray = None,
        batch_size: int = 32,
        patience: int = 5,
        min_delta: float = 1e-4
    ) -> None:
        """
        Training loop dell'autoencoder con early stopping basato su validation loss.
        
        Args:
            optimizer: PyTorch optimizer
            epochs: Numero di epoche di training
            train_data: Dati di training (numpy array)
            mask: Mask per ignorare valori di padding nel training
            val_data: Dati di validazione per early stopping (numpy array)
            val_mask: Mask per i dati di validazione
            batch_size: Dimensione dei batch
            patience: Numero di epoche senza miglioramento prima di fermare il training
            min_delta: Miglioramento minimo della loss per considerarlo significativo
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
            train_loss = self._run_epoch(train_data, mask, batch_size, optimizer, is_training=True)
            print(f"Training loss: {train_loss:.6f}")
            self.losses.append(train_loss)
            
            # Validation e early stopping
            if val_data is not None:
                val_loss = self._run_epoch(val_data, val_mask, batch_size, is_training=False)
                print(f"Validation loss: {val_loss:.6f}")
                self.val_losses.append(val_loss)
                
                # Salva i pesi se è la miglior epoca
                if val_loss < best_val_loss - min_delta:
                    best_weights = copy.deepcopy(self.state_dict())
                    best_epoch = epoch + 1
                
                best_val_loss, epochs_without_improvement, should_stop = self._check_early_stopping(
                    val_loss, best_val_loss, epochs_without_improvement, patience, min_delta
                )
                
                if should_stop:
                    print(f"\nEarly stopping attivato! Training fermato all'epoca {epoch + 1}")
                    print(f"   Miglior validation loss: {best_val_loss:.6f}")
                    break
        
        # Ripristina i pesi della miglior epoca
        if best_weights is not None:
            self.load_state_dict(best_weights)
            print(f"\nRipristinati i pesi dell'epoca {best_epoch} (best val_loss: {best_val_loss:.6f})")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut per ottenere solo il codice latente."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica un codice latente."""
        return self.decoder(z)
