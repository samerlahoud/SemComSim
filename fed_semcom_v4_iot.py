"""
FedLoL IoT v10 - Complete streamlined implementation
10 clients, dynamic training, 64x64 images, visual verification
Round 01 │ MSE=0.0131 │ PSNR=18.82 dB │ Avg Loss=0.0486 │ Loss Std=0.0093
Round 09 │ MSE=0.0015 │ PSNR=28.14 dB │ Avg Loss=0.0073 │ Loss Std=0.0003
"""

import copy
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_CLIENTS      = 10           # Total clients
DYNAMIC_EPOCHS   = False        # Enable dynamic epochs
EPOCHS           = 3
DIRICHLET_ALPHA  = 1.0          # Non-IID level
ROUNDS           = 20           # Training rounds
BATCH_SIZE       = 16           # Batch size
LR               = 3e-4         # Learning rate
BOTTLENECK       = 256          # Semantic bottleneck
COMPRESSED       = 32           # Channel compression
SNR_DB           = 10           # Channel SNR
ALPHA_LOSS       = 0.8          # Loss weighting
PIXELS           = 64 * 64 * 3  # Image pixels

# Device (CPU/CUDA)
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True))   # 64->32
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True))  # 32->16
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True)) # 16->8
        self.fc = nn.Linear(128 * 8 * 8, bottleneck)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z = self.fc(f3.flatten(1))
        return z, (f1, f2)


class SemanticDecoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.fc = nn.Linear(bottleneck, 128 * 8 * 8)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))  # 8->16
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.ReLU(inplace=True))  # 16->32
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 16, 4, 2, 1), nn.ReLU(inplace=True))   # 32->64
        self.out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, z, skips):
        f1, f2 = skips
        x = self.fc(z).view(-1, 128, 8, 8)
        x = self.up1(x)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))


class ChannelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(BOTTLENECK, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, COMPRESSED)
        )
        
    def forward(self, f, snr_db=SNR_DB):
        return self.layers(f)


class ChannelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(COMPRESSED, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, BOTTLENECK)
        )
        
    def forward(self, x, snr_db=SNR_DB):
        return self.layers(x)


class IoTSemanticComm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_s = SemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = SemanticDecoder()

    def forward(self, img, snr_db=SNR_DB):
        # Semantic encoding
        z, skips = self.enc_s(img)
        
        # Channel encoding
        x = self.enc_c(z, snr_db)

        # Channel simulation
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        h = torch.randn_like(x)
        noise = sigma * torch.randn_like(x)
        y = h * x + noise
        x_hat = y / (h + 1e-6)

        # Channel decoding
        z_hat = self.dec_c(x_hat, snr_db)
        
        # Semantic reconstruction
        return self.dec_s(z_hat, skips)

    def get_model_size(self):
        param_size = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return param_size


# ------------------------------------------------------------------
# Loss Function
# ------------------------------------------------------------------
def hybrid_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    return alpha * mse_term + (1.0 - alpha) * l1_term


# ------------------------------------------------------------------
# Dynamic Training Strategy
# ------------------------------------------------------------------
def dynamic_epochs():
    """Each client randomly decides training epochs (1-5)"""
    return random.choice([0, 0, 0, 1, 2, 3, 4, 5])  # Weighted toward fewer epochs

def local_train(model, loader, epochs: int, client_id: int):
    """Local training on client device"""
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        for img, _ in loader:
            img = img.to(device)
            
            opt.zero_grad()
            recon = model(img)
            loss = hybrid_loss(recon, img)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"  Client {client_id+1}: {epochs} epochs, Loss = {avg_loss:.4f}")
    return avg_loss


# ------------------------------------------------------------------
# FedLoL Aggregation
# ------------------------------------------------------------------
def fedlol_aggregate(global_model, client_states, client_losses):
    """FedLoL aggregation based on loss performance"""
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    new_state = copy.deepcopy(global_model.state_dict())

    for k in new_state.keys():
        new_state[k] = sum(
            ((total_loss - client_losses[i]) / ((len(client_losses) - 1) * total_loss))
            * client_states[i][k]
            for i in range(len(client_losses))
        )
    global_model.load_state_dict(new_state)


# ------------------------------------------------------------------
# Data Loading and Partitioning
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def dirichlet_split(dataset, alpha: float, n_clients: int):
    """Split dataset using Dirichlet distribution for non-IID"""
    label_to_indices = {}
    for idx, (_, lbl) in enumerate(dataset):
        label_to_indices.setdefault(lbl, []).append(idx)

    clients = [[] for _ in range(n_clients)]
    for indices in label_to_indices.values():
        proportions = torch.distributions.Dirichlet(
            torch.full((n_clients,), alpha)
        ).sample()
        proportions = (proportions / proportions.sum()).tolist()
        split_points = [0] + list(
            torch.cumsum(torch.tensor(proportions) * len(indices), dim=0).long()
        )
        for cid in range(n_clients):
            clients[cid].extend(indices[split_points[cid]:split_points[cid + 1]])
    
    return [Subset(dataset, idxs) for idxs in clients]


# ------------------------------------------------------------------
# Visual Verification
# ------------------------------------------------------------------
def visual_check(model, val_loader, round_num, save_path="progress"):
    """Visual verification of reconstruction quality"""
    model.eval()
    
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch[:4].to(device)
        recon_batch = model(img_batch)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        for i in range(4):
            # Original images
            orig_img = img_batch[i].cpu().permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed images
            recon_img = recon_batch[i].cpu().permute(1, 2, 0).clamp(0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Recon {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Round {round_num:02d} - Reconstruction Quality')
        plt.tight_layout()
        plt.savefig(f'{save_path}_round_{round_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.show()


# ------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------
def main():
    print("FedLoL IoT v10 - Starting Training")
    print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {BATCH_SIZE} batch size")
    
    # Load dataset
    try:
        train_full = datasets.ImageFolder("./tiny-imagenet-20/train", TRANSFORM)
        val_full = datasets.ImageFolder("./tiny-imagenet-20/val", TRANSFORM)
        print("Using Tiny ImageNet dataset")
    except:
        train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
        val_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM)
        print("Using CIFAR-10 dataset")

    # Create non-IID client splits
    client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
    val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    global_model = IoTSemanticComm().to(device)
    print(f"Model size: {global_model.get_model_size():.2f} MB")
    print(f"Total parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # Create client data loaders
    client_loaders = [
        DataLoader(client_sets[cid], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        for cid in range(NUM_CLIENTS)
    ]
    
    # Print client data distribution
    print("\nClient Data Distribution:")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid+1}: {len(client_sets[cid])} samples")

    print(f"\nStarting federated training for {ROUNDS} rounds...")
    print("=" * 70)

    # Training loop
    for rnd in range(1, ROUNDS + 1):
        print(f"\nRound {rnd:02d}")
        
        client_states, client_losses = [], []

        # Each client trains with dynamic epochs
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            if DYNAMIC_EPOCHS:
                epochs = dynamic_epochs()  # Each client decides training amount
            else:
                epochs = EPOCHS    
            loss_val = local_train(local_model, client_loaders[cid], epochs, cid)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)

        # FedLoL aggregation
        fedlol_aggregate(global_model, client_states, client_losses)

        # Validation
        global_model.eval()
        with torch.no_grad():
            mse_sum, n_img = 0.0, 0
            for img, _ in val_loader:
                img = img.to(device)
                recon = global_model(img)
                mse_sum += nn.functional.mse_loss(recon, img, reduction='sum').item()
                n_img += img.size(0)
                
                if n_img >= 1000:  # Process subset for efficiency
                    break

        # Calculate metrics
        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-10))
        avg_client_loss = sum(client_losses) / len(client_losses)
        loss_std = np.std(client_losses)

        print(f"Round {rnd:02d} │ MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
              f"Avg Loss={avg_client_loss:.4f} │ Loss Std={loss_std:.4f}")

        # Visual check every 3 rounds
        if rnd % 3 == 0:
            visual_check(global_model, val_loader, rnd)

        print("-" * 70)

    print("\nTraining completed!")
    
    # Final visual check
    print("Generating final reconstruction comparison...")
    visual_check(global_model, val_loader, ROUNDS, "final")
    
    return global_model


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = main()