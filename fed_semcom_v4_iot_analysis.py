"""
FedLoL Client Contribution Analysis
Tracks relationship between data size, training epochs, and influence on results
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
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_CLIENTS      = 10
DYNAMIC_EPOCHS   = True         # Enable to see epoch variation impact
EPOCHS           = 3
DIRICHLET_ALPHA  = 1.0          # Lower = more non-IID
ROUNDS           = 10           # Reduced for analysis
BATCH_SIZE       = 16
LR               = 3e-4
BOTTLENECK       = 256
COMPRESSED       = 32
SNR_DB           = 10
ALPHA_LOSS       = 0.8
PIXELS           = 64 * 64 * 3

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Model Architecture (Same as original)
# ------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    def __init__(self, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True))
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
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 16, 4, 2, 1), nn.ReLU(inplace=True))
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
            nn.Linear(BOTTLENECK, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, COMPRESSED)
        )
    def forward(self, f, snr_db=SNR_DB):
        return self.layers(f)

class ChannelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(COMPRESSED, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
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
        z, skips = self.enc_s(img)
        x = self.enc_c(z, snr_db)
        
        # Channel simulation
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        h = torch.randn_like(x)
        noise = sigma * torch.randn_like(x)
        y = h * x + noise
        x_hat = y / (h + 1e-6)
        
        z_hat = self.dec_c(x_hat, snr_db)
        return self.dec_s(z_hat, skips)

# ------------------------------------------------------------------
# Loss and Training
# ------------------------------------------------------------------
def hybrid_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    return alpha * mse_term + (1.0 - alpha) * l1_term

def dynamic_epochs():
    return random.choice([0, 0, 0, 1, 2, 3, 4, 5])

def local_train(model, loader, epochs: int, client_id: int):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

# ------------------------------------------------------------------
# FedLoL Aggregation with Contribution Tracking
# ------------------------------------------------------------------
def fedlol_aggregate_with_analysis(global_model, client_states, client_losses):
    """FedLoL aggregation with contribution weight analysis"""
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    new_state = copy.deepcopy(global_model.state_dict())
    
    # Calculate contribution weights for analysis
    weights = []
    for i in range(len(client_losses)):
        weight = (total_loss - client_losses[i]) / ((len(client_losses) - 1) * total_loss)
        weights.append(weight)
    
    # Aggregate
    for k in new_state.keys():
        new_state[k] = sum(weights[i] * client_states[i][k] for i in range(len(client_losses)))
    
    global_model.load_state_dict(new_state)
    return weights

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def dirichlet_split(dataset, alpha: float, n_clients: int):
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
# Analysis Functions
# ------------------------------------------------------------------
def create_contribution_table(round_data):
    """Create comprehensive analysis table"""
    df = pd.DataFrame(round_data)
    
    # Add derived metrics
    df['Loss_Rank'] = df.groupby('Round')['Local_Loss'].rank(ascending=True)
    df['Weight_Rank'] = df.groupby('Round')['Contribution_Weight'].rank(ascending=False)
    df['Data_Efficiency'] = df['Contribution_Weight'] / (df['Data_Size'] + 1e-6)
    df['Training_Efficiency'] = df['Contribution_Weight'] / (df['Epochs'] + 1e-6)
    
    return df

def analyze_correlations(df):
    """Analyze correlations between factors"""
    correlations = {}
    
    # Overall correlations
    correlations['Data_Size_vs_Weight'] = df['Data_Size'].corr(df['Contribution_Weight'])
    correlations['Epochs_vs_Weight'] = df['Epochs'].corr(df['Contribution_Weight'])
    correlations['Loss_vs_Weight'] = df['Local_Loss'].corr(df['Contribution_Weight'])
    correlations['Data_Size_vs_Loss'] = df['Data_Size'].corr(df['Local_Loss'])
    correlations['Epochs_vs_Loss'] = df['Epochs'].corr(df['Local_Loss'])
    
    return correlations

def print_round_analysis(round_num, client_data, global_mse, global_psnr):
    """Print analysis for a specific round"""
    df_round = pd.DataFrame(client_data)
    df_round = df_round.sort_values('Contribution_Weight', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"ROUND {round_num} ANALYSIS")
    print(f"Global MSE: {global_mse:.6f} | Global PSNR: {global_psnr:.2f} dB")
    print(f"{'='*80}")
    
    print(f"{'Client':<8} {'Data':<6} {'Epochs':<7} {'Loss':<8} {'Weight':<8} {'Efficiency':<10}")
    print(f"{'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")
    
    for _, row in df_round.iterrows():
        efficiency = row['Contribution_Weight'] / (row['Data_Size'] + 1e-6) * 1000
        print(f"{row['Client_ID']:<8} {row['Data_Size']:<6} {row['Epochs']:<7} "
              f"{row['Local_Loss']:<8.4f} {row['Contribution_Weight']:<8.4f} {efficiency:<10.2f}")

# ------------------------------------------------------------------
# Main Training with Analysis
# ------------------------------------------------------------------
def main_with_analysis():
    print("FedLoL Client Contribution Analysis")
    print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds")
    
    # Load dataset
    try:
        train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
        val_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM)
        print("Using CIFAR-10 dataset")
    except:
        print("Could not load dataset")
        return
    
    # Create client splits
    client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
    val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    global_model = IoTSemanticComm().to(device)
    
    # Create client loaders
    client_loaders = [
        DataLoader(client_sets[cid], batch_size=BATCH_SIZE, shuffle=True)
        for cid in range(NUM_CLIENTS)
    ]
    
    # Storage for analysis
    all_round_data = []
    
    print(f"\nClient Data Distribution:")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid+1}: {len(client_sets[cid])} samples")
    
    # Training loop with analysis
    for rnd in range(1, ROUNDS + 1):
        client_states, client_losses = [], []
        client_epochs = []
        
        # Client training
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            epochs = dynamic_epochs() if DYNAMIC_EPOCHS else EPOCHS
            client_epochs.append(epochs)
            
            loss_val = local_train(local_model, client_loaders[cid], epochs, cid)
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)
        
        # FedLoL aggregation with analysis
        contribution_weights = fedlol_aggregate_with_analysis(global_model, client_states, client_losses)
        
        # Global validation
        global_model.eval()
        with torch.no_grad():
            mse_sum, n_img = 0.0, 0
            for img, _ in val_loader:
                img = img.to(device)
                recon = global_model(img)
                mse_sum += nn.functional.mse_loss(recon, img, reduction='sum').item()
                n_img += img.size(0)
                if n_img >= 1000:
                    break
        
        global_mse = mse_sum / (n_img * PIXELS)
        global_psnr = 10.0 * math.log10(1.0 / max(global_mse, 1e-10))
        
        # Store round data
        round_client_data = []
        for cid in range(NUM_CLIENTS):
            round_client_data.append({
                'Round': rnd,
                'Client_ID': cid + 1,
                'Data_Size': len(client_sets[cid]),
                'Epochs': client_epochs[cid],
                'Local_Loss': client_losses[cid],
                'Contribution_Weight': contribution_weights[cid],
                'Global_MSE': global_mse,
                'Global_PSNR': global_psnr
            })
        
        all_round_data.extend(round_client_data)
        
        # Print round analysis
        print_round_analysis(rnd, round_client_data, global_mse, global_psnr)
    
    # Final comprehensive analysis
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    df = create_contribution_table(all_round_data)
    correlations = analyze_correlations(df)
    
    print("\nCorrelation Analysis:")
    print(f"Data Size vs Contribution Weight: {correlations['Data_Size_vs_Weight']:.3f}")
    print(f"Training Epochs vs Contribution Weight: {correlations['Epochs_vs_Weight']:.3f}")
    print(f"Local Loss vs Contribution Weight: {correlations['Loss_vs_Weight']:.3f}")
    print(f"Data Size vs Local Loss: {correlations['Data_Size_vs_Loss']:.3f}")
    print(f"Training Epochs vs Local Loss: {correlations['Epochs_vs_Loss']:.3f}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    summary_stats = df.groupby('Client_ID').agg({
        'Data_Size': 'first',
        'Epochs': 'mean',
        'Local_Loss': 'mean',
        'Contribution_Weight': 'mean',
        'Data_Efficiency': 'mean',
        'Training_Efficiency': 'mean'
    }).round(4)
    
    print(summary_stats.to_string())
    
    # Top contributors analysis
    print(f"\nTop Contributors (by average contribution weight):")
    top_contributors = summary_stats.sort_values('Contribution_Weight', ascending=False).head(5)
    print(top_contributors[['Data_Size', 'Epochs', 'Contribution_Weight']].to_string())
    
    return df, global_model

if __name__ == "__main__":
    analysis_df, trained_model = main_with_analysis()