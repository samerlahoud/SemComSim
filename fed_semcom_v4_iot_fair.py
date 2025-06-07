"""
FedLoL IoT v11 - Utilitarian vs Proportional Fairness Comparison
Two client selection strategies: Utilitarian (performance-first) vs Proportional Fairness (effort-balanced)
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
ROUNDS           = 50           # Training rounds
BATCH_SIZE       = 16           # Batch size
LR               = 3e-4         # Learning rate
BOTTLENECK       = 256          # Semantic bottleneck
COMPRESSED       = 32           # Channel compression
SNR_DB           = 10           # Channel SNR
ALPHA_LOSS       = 0.8          # Loss weighting
PIXELS           = 64 * 64 * 3  # Image pixels

# NEW: Client Selection Strategy Configuration
CLIENT_SELECTION = "proportional"  # "utilitarian", "proportional", or "baseline"
CLIENTS_PER_ROUND = 6             # Number of clients selected each round
TARGET_EFFORT = 1000              # Target training steps per client (for proportional fairness)
MAX_EPOCHS_CAP = 15               # Maximum epochs to prevent overfitting

# Device (CPU/CUDA)
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Model Architecture (UNCHANGED)
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
# Loss Function (UNCHANGED)
# ------------------------------------------------------------------
def hybrid_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    return alpha * mse_term + (1.0 - alpha) * l1_term


# ------------------------------------------------------------------
# NEW: Client Selection Strategies
# ------------------------------------------------------------------
class ClientManager:
    def __init__(self, num_clients, client_dataset_sizes):
        self.num_clients = num_clients
        self.dataset_sizes = client_dataset_sizes
        self.participation_history = [0] * num_clients  # Track participation count
        self.total_effort_contributed = [0] * num_clients  # Track total effort
        
    def utilitarian_selection(self, clients_per_round):
        """
        Utilitarian: Select clients with highest potential contribution
        Favor data-rich clients who are willing to train more
        """
        client_scores = []
        for cid in range(self.num_clients):
            # Base willingness (varies by client type)
            if self.dataset_sizes[cid] > 800:  # Large dataset clients
                willing_epochs = random.choice([2, 3, 4])
            elif self.dataset_sizes[cid] > 300:  # Medium dataset clients  
                willing_epochs = random.choice([3, 4, 5])
            else:  # Small dataset clients
                willing_epochs = random.choice([4, 5, 6])
            
            # Utilitarian score: maximize total training steps
            potential_contribution = self.dataset_sizes[cid] * willing_epochs
            client_scores.append((cid, potential_contribution, willing_epochs))
        
        # Select top clients by potential contribution
        client_scores.sort(key=lambda x: x[1], reverse=True)
        selected = client_scores[:clients_per_round]
        
        return [(cid, epochs) for cid, _, epochs in selected]
    
    def proportional_fairness_selection(self, clients_per_round, target_effort):
        """
        Proportional Fairness: Ensure balanced effort distribution
        Adjust epochs to normalize computational effort across clients
        """
        # Calculate participation weights (favor underparticipated clients)
        total_participation = sum(self.participation_history) + 1e-6
        participation_weights = [1.0 / (1.0 + p / total_participation) 
                               for p in self.participation_history]
        
        # Calculate fair epochs for each client based on dataset size
        client_configs = []
        for cid in range(self.num_clients):
            # Proportional fairness: equal effort = equal influence
            fair_epochs = max(1, min(target_effort // self.dataset_sizes[cid], MAX_EPOCHS_CAP))
            
            # Selection probability based on participation fairness
            selection_weight = participation_weights[cid]
            
            client_configs.append((cid, fair_epochs, selection_weight))
        
        # Select clients probabilistically based on fairness weights
        weights = [config[2] for config in client_configs]
        selected_indices = np.random.choice(
            self.num_clients, 
            size=min(clients_per_round, self.num_clients),
            replace=False, 
            p=np.array(weights) / sum(weights)
        )
        
        selected = [(client_configs[i][0], client_configs[i][1]) for i in selected_indices]
        return selected
    
    def update_participation(self, selected_clients_epochs):
        """Update participation tracking"""
        for cid, epochs in selected_clients_epochs:
            self.participation_history[cid] += 1
            self.total_effort_contributed[cid] += self.dataset_sizes[cid] * epochs
    
    def get_fairness_metrics(self):
        """Calculate fairness metrics for analysis"""
        # Participation fairness (Gini coefficient)
        participation_gini = self._gini_coefficient(self.participation_history)
        
        # Effort fairness  
        effort_gini = self._gini_coefficient(self.total_effort_contributed)
        
        return {
            'participation_gini': participation_gini,
            'effort_gini': effort_gini,
            'participation_dist': self.participation_history.copy(),
            'effort_dist': self.total_effort_contributed.copy()
        }
    
    def _gini_coefficient(self, values):
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
        if sum(values) == 0:
            return 0
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * sum(values)) - (n + 1) / n


# ------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------
def local_train(model, loader, epochs: int, client_id: int):
    """Local training on client device (UNCHANGED logic)"""
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
    print(f"  Client {client_id+1}: {epochs} epochs, {len(loader.dataset)} samples, "
          f"Effort={len(loader.dataset) * epochs}, Loss={avg_loss:.4f}")
    return avg_loss


# ------------------------------------------------------------------
# FedLoL Aggregation (UNCHANGED)
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
# Data Loading and Partitioning (UNCHANGED)
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
# Visual Verification (UNCHANGED)
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
        #plt.show()


# ------------------------------------------------------------------
# Enhanced Main Training Loop
# ------------------------------------------------------------------
def main():
    print(f"FedLoL IoT v11 - Client Selection Strategy: {CLIENT_SELECTION.upper()}")
    print(f"Configuration: {NUM_CLIENTS} clients, {CLIENTS_PER_ROUND} selected per round, {ROUNDS} rounds")
    
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
    
    # Initialize client manager
    dataset_sizes = [len(client_sets[cid]) for cid in range(NUM_CLIENTS)]
    client_manager = ClientManager(NUM_CLIENTS, dataset_sizes)
    
    # Print client data distribution
    print("\nClient Data Distribution:")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid+1}: {dataset_sizes[cid]} samples")

    print(f"\nStarting federated training for {ROUNDS} rounds...")
    print("=" * 80)

    # Training loop
    for rnd in range(1, ROUNDS + 1):
        print(f"\nRound {rnd:02d}")
        
        # NEW: Client selection based on strategy
        if CLIENT_SELECTION == "utilitarian":
            selected_clients = client_manager.utilitarian_selection(CLIENTS_PER_ROUND)
        elif CLIENT_SELECTION == "proportional":
            selected_clients = client_manager.proportional_fairness_selection(CLIENTS_PER_ROUND, TARGET_EFFORT)
        else:  # baseline: select all clients with fixed epochs
            selected_clients = [(cid, EPOCHS) for cid in range(NUM_CLIENTS)]
        
        print(f"Selected clients: {[(cid+1, epochs) for cid, epochs in selected_clients]}")
        
        client_states, client_losses = [], []

        # Train selected clients
        for cid, epochs in selected_clients:
            local_model = copy.deepcopy(global_model)
            loss_val = local_train(local_model, client_loaders[cid], epochs, cid)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)

        # Update participation tracking
        client_manager.update_participation(selected_clients)

        # FedLoL aggregation (UNCHANGED)
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

        # NEW: Fairness metrics
        fairness_metrics = client_manager.get_fairness_metrics()
        
        print(f"Round {rnd:02d} │ MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
              f"Avg Loss={avg_client_loss:.4f} │ Loss Std={loss_std:.4f}")
        print(f"         │ Participation Gini={fairness_metrics['participation_gini']:.3f} │ "
              f"Effort Gini={fairness_metrics['effort_gini']:.3f}")

        # Visual check every 5 rounds
        if rnd % 5 == 0:
            visual_check(global_model, val_loader, rnd)

        print("-" * 80)

    # Final analysis
    print("\nTraining completed!")
    print("\nFinal Fairness Analysis:")
    final_metrics = client_manager.get_fairness_metrics()
    print(f"Participation distribution: {final_metrics['participation_dist']}")
    print(f"Total effort distribution: {final_metrics['effort_dist']}")
    print(f"Participation Gini coefficient: {final_metrics['participation_gini']:.3f}")
    print(f"Effort Gini coefficient: {final_metrics['effort_gini']:.3f}")
    
    # Final visual check
    print("Generating final reconstruction comparison...")
    visual_check(global_model, val_loader, ROUNDS, "final")
    
    return global_model


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = main()