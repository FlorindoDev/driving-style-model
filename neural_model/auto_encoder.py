import time
import torch
import torch.nn as nn
import numpy as np

from torch import optim
import matplotlib.pyplot as plt
import csv


print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda.is_available (ROCm usa questa API):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



class AutoEncoder(nn.Module):
    def __init__(self, input_dim=455, latent_dim=32):
        super().__init__()
        self.losses = []
        self.input_dim= input_dim
        self.loss_function = MaskeredMSELoss().to(device)

        # Encoder: x -> z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder: z -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, encoder_only=False):
        z = self.encoder(x)
        if encoder_only:
            return z 
        else:
            x_hat = self.decoder(z)
            return x_hat


    def train(self, optimizer, epochs, input, mask, bach_size=32):
        outputs = []
        self.losses = []

        for epoch in range(epochs):
            print(f"\n\nEpoch {epoch + 1}/{epochs}\n" + "-" * 40)
            total_loss = 0
            num_batches = 0

            for i in range(0, len(input), bach_size):
                batch_input = np.atleast_1d(input[i:i+bach_size])
                batch_mask = np.atleast_1d(mask[i:i+bach_size])

                batch_input = torch.tensor(batch_input, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                outputs = self.forward(batch_input)
                loss = self.loss_function(outputs, batch_input, batch_mask)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Training loss: {avg_loss:.6f}")
            self.losses.append(avg_loss)



class MaskeredMSELoss(nn.MSELoss):
    def __init__(self):
        super(MaskeredMSELoss, self).__init__()

    def forward(self, input, target, mask):
        
        np_mask = mask.bool()
        masked_input = input[np_mask]
        masked_target = target[np_mask]

        return super(MaskeredMSELoss, self).forward(masked_input, masked_target)



   

