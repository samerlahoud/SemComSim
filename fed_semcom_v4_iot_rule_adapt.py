"""
FedLoL IoT v11 - Channel-Aware & Device-Adaptive Semantic Compression
Adaptive bottleneck based on SNR and device capabilities
Preserves original semantic communication chain with adaptive enhancements
This is rule based with static thresholds for bottleneck sizes
Interesting but still not fully adaptive
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
import json
import csv
import os
from datetime import datetime
import pickle

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# Configuration (Enhanced)
# ------------------------------------------------------------------
NUM_CLIENTS      = 5           # Total clients
DYNAMIC_EPOCHS   = False        # Enable dynamic epochs
EPOCHS           = 3
DIRICHLET_ALPHA  = 1.0          # Non-IID level
ROUNDS           = 10           # Training rounds
BATCH_SIZE       = 16           # Batch size
LR               = 3e-4         # Learning rate

# ADAPTIVE COMPRESSION PARAMETERS (NEW)
MIN_BOTTLENECK   = 64           # Minimum bottleneck size
MAX_BOTTLENECK   = 512          # Maximum bottleneck size
DEFAULT_BOTTLENECK = 256        # Default/fallback bottleneck

COMPRESSED       = 32           # Channel compression (kept same)
SNR_DB           = 10           # Default channel SNR
ALPHA_LOSS       = 0.8          # Loss weighting
PIXELS           = 64 * 64 * 3  # Image pixels

# Device (CPU/CUDA)
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Smart Logging System for Research Analysis
# ------------------------------------------------------------------
class AdaptiveFedLoLLogger:
    """Comprehensive logging system for adaptive federated semantic communication"""
    
    def __init__(self, experiment_name="adaptive_fedlol", base_dir="./experiments"):
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{experiment_name}_{timestamp}"
        self.exp_dir = os.path.join(base_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize data structures
        self.experiment_config = {}
        self.round_metrics = []
        self.client_details = []
        self.convergence_data = []
        self.device_profiles = []
        
        print(f"📊 Logging experiment to: {self.exp_dir}")
    
    def log_config(self, config_dict):
        """Log experiment configuration"""
        self.experiment_config = config_dict
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def log_device_profiles(self, profiles):
        """Log client device profiles"""
        self.device_profiles = profiles
        profiles_path = os.path.join(self.exp_dir, "device_profiles.json")
        
        # Convert to serializable format
        serializable_profiles = []
        for i, profile in enumerate(profiles):
            serializable_profiles.append({
                'client_id': i + 1,
                'device_type': profile['type'],
                'description': profile['profile']['description'],
                'max_bottleneck': profile['profile']['max_bottleneck'],
                'min_bottleneck': profile['profile']['min_bottleneck'],
                'power_budget': profile['profile']['power_budget'],
                'initial_energy': profile['energy_level'],
                'base_snr': profile['base_snr'],
                'channel_variance': profile['channel_variance']
            })
        
        with open(profiles_path, 'w') as f:
            json.dump(serializable_profiles, f, indent=2)
    
    def log_round(self, round_num, global_metrics, client_data, channel_conditions):
        """Log comprehensive round data"""
        
        # Global round metrics
        round_entry = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            **global_metrics,
            'channel_stats': channel_conditions
        }
        self.round_metrics.append(round_entry)
        
        # Individual client data for this round
        for client_data_entry in client_data:
            client_entry = {
                'round': round_num,
                **client_data_entry
            }
            self.client_details.append(client_entry)
        
        # Update convergence tracking
        if len(self.round_metrics) >= 2:
            prev_psnr = self.round_metrics[-2]['psnr']
            curr_psnr = global_metrics['psnr']
            convergence_entry = {
                'round': round_num,
                'psnr_improvement': curr_psnr - prev_psnr,
                'loss_std': global_metrics['loss_std'],
                'bottleneck_variance': np.var([cd['bottleneck'] for cd in client_data]),
                'snr_diversity': channel_conditions['snr_std']
            }
            self.convergence_data.append(convergence_entry)
    
    def save_data(self):
        """Save all logged data to files for analysis"""
        
        # Round-by-round metrics (CSV for easy plotting)
        rounds_path = os.path.join(self.exp_dir, "round_metrics.csv")
        if self.round_metrics:
            fieldnames = self.round_metrics[0].keys()
            with open(rounds_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.round_metrics)
        
        # Client details (CSV for per-client analysis)
        clients_path = os.path.join(self.exp_dir, "client_details.csv")
        if self.client_details:
            fieldnames = self.client_details[0].keys()
            with open(clients_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.client_details)
        
        # Convergence analysis
        convergence_path = os.path.join(self.exp_dir, "convergence_data.csv")
        if self.convergence_data:
            fieldnames = self.convergence_data[0].keys()
            with open(convergence_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.convergence_data)
        
        # Summary statistics
        summary = self.generate_summary()
        summary_path = os.path.join(self.exp_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Data saved to {self.exp_dir}")
        print(f"📈 Ready for analysis with {len(self.round_metrics)} rounds, {len(self.client_details)} client entries")
    
    def generate_summary(self):
        """Generate experiment summary statistics"""
        if not self.round_metrics:
            return {}
        
        psnr_values = [r['psnr'] for r in self.round_metrics]
        loss_values = [r['avg_loss'] for r in self.round_metrics]
        bottleneck_values = [r['avg_bottleneck'] for r in self.round_metrics]
        
        # Client type performance analysis
        client_type_stats = {}
        for device_type in ['high_end', 'mid_range', 'low_power', 'ultra_low']:
            type_clients = [cd for cd in self.client_details if cd['device_type'] == device_type]
            if type_clients:
                type_losses = [cd['loss'] for cd in type_clients]
                type_bottlenecks = [cd['bottleneck'] for cd in type_clients]
                client_type_stats[device_type] = {
                    'count': len(type_clients) // len(self.round_metrics),  # Clients of this type
                    'avg_loss': np.mean(type_losses),
                    'avg_bottleneck': np.mean(type_bottlenecks),
                    'bottleneck_range': [min(type_bottlenecks), max(type_bottlenecks)]
                }
        
        return {
            'experiment_name': self.exp_name,
            'total_rounds': len(self.round_metrics),
            'total_clients': len(self.device_profiles),
            'final_performance': {
                'psnr': psnr_values[-1],
                'avg_loss': loss_values[-1],
                'avg_bottleneck': bottleneck_values[-1],
                'compression_ratio': (256 - bottleneck_values[-1]) / 256
            },
            'convergence_analysis': {
                'psnr_improvement_total': psnr_values[-1] - psnr_values[0],
                'best_psnr': max(psnr_values),
                'convergence_round': self.estimate_convergence_round(),
                'final_stability': np.std(psnr_values[-3:]) if len(psnr_values) >= 3 else None
            },
            'efficiency_metrics': {
                'avg_compression_ratio': np.mean([(256 - b) / 256 for b in bottleneck_values]),
                'max_compression_achieved': max([(256 - b) / 256 for b in bottleneck_values]),
                'resource_savings_percent': ((256 - np.mean(bottleneck_values)) / 256) * 100
            },
            'client_type_performance': client_type_stats,
            'channel_diversity': {
                'max_snr_spread': max([r['channel_stats']['snr_range'] for r in self.round_metrics]),
                'avg_snr_diversity': np.mean([r['channel_stats']['snr_std'] for r in self.round_metrics])
            }
        }
    
    def estimate_convergence_round(self):
        """Estimate when convergence occurred based on PSNR improvement rate"""
        if len(self.convergence_data) < 3:
            return None
        
        improvements = [cd['psnr_improvement'] for cd in self.convergence_data]
        for i in range(len(improvements) - 2):
            # Check if last 3 improvements are all < 0.5 dB
            if all(imp < 0.5 for imp in improvements[i:i+3]):
                return self.convergence_data[i]['round']
        return None


# ------------------------------------------------------------------
# Device Profiling System (unchanged)
# ------------------------------------------------------------------
DEVICE_PROFILES = {
    'high_end': {
        'max_bottleneck': 512, 
        'min_bottleneck': 256, 
        'power_budget': 1.0,
        'description': 'RPi 4, Jetson Nano'
    },
    'mid_range': {
        'max_bottleneck': 256, 
        'min_bottleneck': 128, 
        'power_budget': 0.7,
        'description': 'ESP32-S3, powerful MCU'
    },
    'low_power': {
        'max_bottleneck': 128, 
        'min_bottleneck': 64, 
        'power_budget': 0.4,
        'description': 'ESP32, Arduino'
    },
    'ultra_low': {
        'max_bottleneck': 64, 
        'min_bottleneck': 32, 
        'power_budget': 0.2,
        'description': 'Ultra-low power sensors'
    }
}

def get_adaptive_bottleneck(snr_db, device_profile, energy_level=1.0):
    """
    Adaptive bottleneck selection based on channel and device constraints
    
    Args:
        snr_db: Current channel SNR in dB
        device_profile: Device capability profile
        energy_level: Battery level (0.0-1.0)
    
    Returns:
        Optimal bottleneck size
    """
    # Base bottleneck from device capability
    max_bottleneck = device_profile['max_bottleneck']
    min_bottleneck = device_profile['min_bottleneck']
    
    # Channel-based scaling factor
    if snr_db > 15:
        channel_factor = 1.0      # High SNR: Use full capability
    elif snr_db > 5:
        channel_factor = 0.75     # Medium SNR: Reduce for reliability
    else:
        channel_factor = 0.5      # Low SNR: Prioritize robustness
    
    # Energy-based scaling
    if energy_level < 0.2:        # Critical battery
        energy_factor = 0.5
    elif energy_level < 0.5:      # Low battery
        energy_factor = 0.75
    else:
        energy_factor = 1.0       # Full power
    
    # Compute adaptive bottleneck
    adaptive_size = int(max_bottleneck * channel_factor * energy_factor)
    
    # Clamp to device constraints and valid range
    adaptive_size = max(min_bottleneck, min(adaptive_size, max_bottleneck))
    adaptive_size = max(MIN_BOTTLENECK, min(adaptive_size, MAX_BOTTLENECK))
    
    return adaptive_size

def assign_device_profiles(num_clients):
    """Assign device profiles to clients (simulating heterogeneous IoT network)"""
    profiles = []
    device_types = list(DEVICE_PROFILES.keys())
    
    # Distribute device types (weighted toward mid/low power for realistic IoT)
    weights = [0.1, 0.3, 0.4, 0.2]  # high_end, mid_range, low_power, ultra_low
    
    for cid in range(num_clients):
        device_type = np.random.choice(device_types, p=weights)
        # Add some energy variation
        energy_level = np.random.uniform(0.3, 1.0)
        
        # Individual channel characteristics (NEW)
        base_snr = np.random.uniform(2, 18)  # Each device has different base SNR (2-18 dB)
        channel_variance = np.random.uniform(2, 6)  # Different temporal variation (2-6 dB)
        
        profiles.append({
            'type': device_type,
            'profile': DEVICE_PROFILES[device_type],
            'energy_level': energy_level,
            'base_snr': base_snr,              # Individual base channel quality
            'channel_variance': channel_variance # Individual channel variation
        })
    
    return profiles

# ------------------------------------------------------------------
# Enhanced Model Architecture (Adaptive Bottleneck)
# ------------------------------------------------------------------
class AdaptiveSemanticEncoder(nn.Module):
    """Enhanced semantic encoder with adaptive bottleneck support"""
    def __init__(self, max_bottleneck: int = MAX_BOTTLENECK):
        super().__init__()
        self.max_bottleneck = max_bottleneck
        
        # Keep original convolutional layers (unchanged)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True))   # 64->32
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True))  # 32->16
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True)) # 16->8
        
        # Adaptive fully connected layer (supports multiple bottleneck sizes)
        self.fc = nn.Linear(128 * 8 * 8, max_bottleneck)

    def forward(self, x, bottleneck_size=None):
        if bottleneck_size is None:
            bottleneck_size = self.max_bottleneck
            
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        # Full feature extraction
        z_full = self.fc(f3.flatten(1))
        
        # Adaptive truncation/selection for smaller bottlenecks
        if bottleneck_size < self.max_bottleneck:
            z = z_full[:, :bottleneck_size]
        else:
            z = z_full
            
        return z, (f1, f2)


class AdaptiveSemanticDecoder(nn.Module):
    """Enhanced semantic decoder with adaptive bottleneck support"""
    def __init__(self, max_bottleneck: int = MAX_BOTTLENECK):
        super().__init__()
        self.max_bottleneck = max_bottleneck
        
        # Adaptive input layer
        self.fc = nn.Linear(max_bottleneck, 128 * 8 * 8)
        
        # Keep original decoder layers (unchanged)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))  # 8->16
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.ReLU(inplace=True))  # 16->32
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 16, 4, 2, 1), nn.ReLU(inplace=True))   # 32->64
        self.out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, z, skips, bottleneck_size=None):
        if bottleneck_size is None:
            bottleneck_size = self.max_bottleneck
            
        f1, f2 = skips
        
        # Pad with zeros if bottleneck is smaller than max
        if bottleneck_size < self.max_bottleneck:
            padding = torch.zeros(z.shape[0], self.max_bottleneck - bottleneck_size, 
                                device=z.device, dtype=z.dtype)
            z_padded = torch.cat([z, padding], dim=1)
        else:
            z_padded = z
            
        x = self.fc(z_padded).view(-1, 128, 8, 8)
        x = self.up1(x)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up3(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))


class ChannelEncoder(nn.Module):
    """Channel encoder (unchanged from original)"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(MAX_BOTTLENECK, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, COMPRESSED)
        )
        
    def forward(self, f, snr_db=SNR_DB):
        return self.layers(f)


class ChannelDecoder(nn.Module):
    """Channel decoder (unchanged from original)"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(COMPRESSED, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, MAX_BOTTLENECK)
        )
        
    def forward(self, x, snr_db=SNR_DB):
        return self.layers(x)


class AdaptiveIoTSemanticComm(nn.Module):
    """Enhanced IoT Semantic Communication with adaptive compression"""
    def __init__(self):
        super().__init__()
        self.enc_s = AdaptiveSemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = AdaptiveSemanticDecoder()

    def forward(self, img, snr_db=SNR_DB, bottleneck_size=DEFAULT_BOTTLENECK):
        # Semantic encoding with adaptive bottleneck
        z, skips = self.enc_s(img, bottleneck_size)
        
        # Pad semantic features for channel processing if needed
        if bottleneck_size < MAX_BOTTLENECK:
            padding = torch.zeros(z.shape[0], MAX_BOTTLENECK - bottleneck_size, 
                                device=z.device, dtype=z.dtype)
            z_padded = torch.cat([z, padding], dim=1)
        else:
            z_padded = z
        
        # Channel encoding
        x = self.enc_c(z_padded, snr_db)

        # Channel simulation (unchanged from original)
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        h = torch.randn_like(x)
        noise = sigma * torch.randn_like(x)
        y = h * x + noise
        x_hat = y / (h + 1e-6)

        # Channel decoding
        z_hat_padded = self.dec_c(x_hat, snr_db)
        
        # Extract relevant semantic features
        z_hat = z_hat_padded[:, :bottleneck_size]
        
        # Semantic reconstruction
        return self.dec_s(z_hat, skips, bottleneck_size)

    def get_model_size(self):
        param_size = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return param_size


# ------------------------------------------------------------------
# Loss Function (unchanged)
# ------------------------------------------------------------------
def hybrid_loss(pred, target, alpha: float = ALPHA_LOSS):
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    return alpha * mse_term + (1.0 - alpha) * l1_term


# ------------------------------------------------------------------
# Enhanced Training Functions
# ------------------------------------------------------------------
def dynamic_epochs():
    """Each client randomly decides training epochs (1-5)"""
    return random.choice([0, 0, 0, 1, 2, 3, 4, 5])  # Weighted toward fewer epochs

def local_train_adaptive(model, loader, epochs: int, client_id: int, device_profile, snr_db=SNR_DB):
    """Local training with adaptive compression"""
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    total_loss = 0
    num_batches = 0
    
    # Simulate energy depletion during training
    energy_level = device_profile['energy_level']
    
    for epoch in range(epochs):
        for batch_idx, (img, _) in enumerate(loader):
            img = img.to(device)
            
            # Get adaptive bottleneck for current conditions
            current_energy = energy_level * (1.0 - 0.1 * epoch / max(epochs, 1))  # Energy decreases
            bottleneck_size = get_adaptive_bottleneck(snr_db, device_profile['profile'], current_energy)
            
            opt.zero_grad()
            recon = model(img, snr_db, bottleneck_size)
            loss = hybrid_loss(recon, img)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    final_bottleneck = get_adaptive_bottleneck(snr_db, device_profile['profile'], current_energy)
    
    print(f"  Client {client_id+1} ({device_profile['type']}): {epochs} epochs, "
          f"SNR={snr_db:.1f}dB, Bottleneck={final_bottleneck}, Loss={avg_loss:.4f}")
    
    return avg_loss, final_bottleneck


# ------------------------------------------------------------------
# Enhanced FedLoL Aggregation (NEW)
# ------------------------------------------------------------------
def adaptive_fedlol_aggregate(global_model, client_states, client_losses, client_bottlenecks):
    """Enhanced FedLoL aggregation considering device capabilities"""
    eps = 1e-8
    total_loss = sum(client_losses) + eps
    new_state = copy.deepcopy(global_model.state_dict())

    # Weight clients by both loss performance AND capability
    client_weights = []
    for i, (loss, bottleneck) in enumerate(zip(client_losses, client_bottlenecks)):
        # Loss-based weight (original FedLoL)
        loss_weight = (total_loss - loss) / ((len(client_losses) - 1) * total_loss)
        
        # Capability-based weight (NEW: higher capacity devices get slightly more weight)
        capability_weight = bottleneck / MAX_BOTTLENECK
        
        # Combined weight (balanced between performance and capability)
        combined_weight = 0.8 * loss_weight + 0.2 * capability_weight
        client_weights.append(combined_weight)
    
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    # Aggregate with enhanced weights
    for k in new_state.keys():
        new_state[k] = sum(
            client_weights[i] * client_states[i][k]
            for i in range(len(client_losses))
        )
    
    global_model.load_state_dict(new_state)
    
    print(f"  Aggregation: Avg bottleneck={np.mean(client_bottlenecks):.1f}, "
          f"Range=[{min(client_bottlenecks)}-{max(client_bottlenecks)}]")


# ------------------------------------------------------------------
# Data Loading (unchanged)
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
# Enhanced Visual Verification
# ------------------------------------------------------------------
def visual_check_adaptive(model, val_loader, round_num, device_profiles, save_path="adaptive_progress"):
    """Visual verification with different device capabilities"""
    model.eval()
    
    with torch.no_grad():
        img_batch, _ = next(iter(val_loader))
        img_batch = img_batch[:4].to(device)
        
        # Test different bottleneck sizes
        bottleneck_sizes = [64, 128, 256, 512]
        
        fig, axes = plt.subplots(len(bottleneck_sizes) + 1, 4, figsize=(12, 15))
        
        for i in range(4):
            # Original images
            orig_img = img_batch[i].cpu().permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructions with different bottlenecks
            for j, bottleneck in enumerate(bottleneck_sizes):
                recon = model(img_batch[i:i+1], SNR_DB, bottleneck)
                recon_img = recon[0].cpu().permute(1, 2, 0).clamp(0, 1)
                axes[j+1, i].imshow(recon_img)
                axes[j+1, i].set_title(f'B={bottleneck}')
                axes[j+1, i].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Original', rotation=90, size='large')
        for j, bottleneck in enumerate(bottleneck_sizes):
            axes[j+1, 0].set_ylabel(f'Bottleneck\n{bottleneck}', rotation=90, size='large')
        
        plt.suptitle(f'Round {round_num:02d} - Adaptive Compression Comparison')
        plt.tight_layout()
        plt.savefig(f'{save_path}_round_{round_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.show()


# ------------------------------------------------------------------
# Main Training Loop (Enhanced)
# ------------------------------------------------------------------
def main():
    print("FedLoL IoT v11 - Channel-Aware & Device-Adaptive Semantic Compression")
    print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {BATCH_SIZE} batch size")
    print(f"Adaptive bottleneck range: {MIN_BOTTLENECK}-{MAX_BOTTLENECK} dimensions")
    
    # Initialize comprehensive logging
    logger = AdaptiveFedLoLLogger(
        experiment_name=f"adaptive_fedlol_{NUM_CLIENTS}clients_{ROUNDS}rounds",
        base_dir="./experiments"
    )
    
    # Log experiment configuration
    config = {
        'num_clients': NUM_CLIENTS,
        'rounds': ROUNDS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'min_bottleneck': MIN_BOTTLENECK,
        'max_bottleneck': MAX_BOTTLENECK,
        'default_bottleneck': DEFAULT_BOTTLENECK,
        'compressed_dims': COMPRESSED,
        'default_snr_db': SNR_DB,
        'alpha_loss': ALPHA_LOSS,
        'dynamic_epochs': DYNAMIC_EPOCHS,
        'epochs': EPOCHS,
        'dirichlet_alpha': DIRICHLET_ALPHA,
        'seed': SEED,
        'device': str(device)
    }
    logger.log_config(config)
    
    # Load dataset (unchanged)
    try:
        train_full = datasets.ImageFolder("./tiny-imagenet-20/train", TRANSFORM)
        val_full = datasets.ImageFolder("./tiny-imagenet-20/val", TRANSFORM)
        dataset_name = "Tiny ImageNet"
        print("Using Tiny ImageNet dataset")
    except:
        train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
        val_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM)
        dataset_name = "CIFAR-10"
        print("Using CIFAR-10 dataset")

    # Create non-IID client splits
    client_sets = dirichlet_split(train_full, DIRICHLET_ALPHA, NUM_CLIENTS)
    val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize adaptive model
    global_model = AdaptiveIoTSemanticComm().to(device)
    print(f"Model size: {global_model.get_model_size():.2f} MB")
    print(f"Total parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # Assign device profiles to clients and log them
    device_profiles = assign_device_profiles(NUM_CLIENTS)
    logger.log_device_profiles(device_profiles)
    
    print("\nClient Device Profiles:")
    for cid, profile in enumerate(device_profiles):
        print(f"  Client {cid+1}: {profile['type']} - {profile['profile']['description']} "
              f"(Max: {profile['profile']['max_bottleneck']}, Energy: {profile['energy_level']:.2f}, "
              f"Base SNR: {profile['base_snr']:.1f}±{profile['channel_variance']:.1f} dB)")

    # Create client data loaders
    client_loaders = [
        DataLoader(client_sets[cid], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        for cid in range(NUM_CLIENTS)
    ]
    
    # Log data distribution
    print("\nClient Data Distribution:")
    for cid in range(NUM_CLIENTS):
        print(f"  Client {cid+1}: {len(client_sets[cid])} samples")
    
    print(f"\nStarting adaptive federated training for {ROUNDS} rounds...")
    print("=" * 80)

    # Training loop with comprehensive logging
    for rnd in range(1, ROUNDS + 1):
        print(f"\nRound {rnd:02d}")
        
        client_states, client_losses, client_bottlenecks, client_snrs = [], [], [], []
        client_round_data = []

        # Each client trains with individual channel conditions and adaptive compression
        for cid in range(NUM_CLIENTS):
            # Generate individual channel SNR for this client
            base_snr = device_profiles[cid]['base_snr']
            variance = device_profiles[cid]['channel_variance']
            current_snr = base_snr + random.uniform(-variance, variance)
            current_snr = max(0, current_snr)  # SNR can't be negative
            
            local_model = copy.deepcopy(global_model)
            if DYNAMIC_EPOCHS:
                epochs = dynamic_epochs()
            else:
                epochs = EPOCHS    
                
            loss_val, bottleneck_used = local_train_adaptive(
                local_model, client_loaders[cid], epochs, cid, device_profiles[cid], current_snr
            )
            
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)
            client_bottlenecks.append(bottleneck_used)
            client_snrs.append(current_snr)
            
            # Detailed client data for logging
            client_round_data.append({
                'client_id': cid + 1,
                'device_type': device_profiles[cid]['type'],
                'base_snr': device_profiles[cid]['base_snr'],
                'current_snr': current_snr,
                'channel_variance': device_profiles[cid]['channel_variance'],
                'max_bottleneck_capability': device_profiles[cid]['profile']['max_bottleneck'],
                'bottleneck': bottleneck_used,
                'epochs': epochs,
                'loss': loss_val,
                'energy_level': device_profiles[cid]['energy_level'],
                'compression_ratio': (256 - bottleneck_used) / 256,
                'adaptation_factor': bottleneck_used / device_profiles[cid]['profile']['max_bottleneck']
            })

        # Enhanced FedLoL aggregation
        adaptive_fedlol_aggregate(global_model, client_states, client_losses, client_bottlenecks)
        
        # Channel condition analysis
        avg_snr = np.mean(client_snrs)
        min_snr, max_snr = min(client_snrs), max(client_snrs)
        snr_std = np.std(client_snrs)
        snr_range = max_snr - min_snr
        
        channel_conditions = {
            'avg_snr': avg_snr,
            'snr_std': snr_std,
            'min_snr': min_snr,
            'max_snr': max_snr,
            'snr_range': snr_range
        }
        
        print(f"  Channel conditions: SNR={avg_snr:.1f}±{snr_std:.1f} dB, Range=[{min_snr:.1f}-{max_snr:.1f}]")

        # Validation with adaptive model
        global_model.eval()
        with torch.no_grad():
            mse_sum, n_img = 0.0, 0
            
            # Test with average conditions across clients
            test_snr = np.mean(client_snrs)
            test_bottleneck = int(np.mean(client_bottlenecks))
            
            for img, _ in val_loader:
                img = img.to(device)
                recon = global_model(img, test_snr, test_bottleneck)
                mse_sum += nn.functional.mse_loss(recon, img, reduction='sum').item()
                n_img += img.size(0)
                
                if n_img >= 1000:  # Process subset for efficiency
                    break

        # Calculate comprehensive metrics
        mse_mean = mse_sum / (n_img * PIXELS)
        psnr_mean = 10.0 * math.log10(1.0 / max(mse_mean, 1e-10))
        avg_client_loss = sum(client_losses) / len(client_losses)
        loss_std = np.std(client_losses)
        avg_bottleneck = np.mean(client_bottlenecks)
        
        # Global metrics for logging
        global_metrics = {
            'mse': mse_mean,
            'psnr': psnr_mean,
            'avg_loss': avg_client_loss,
            'loss_std': loss_std,
            'avg_bottleneck': avg_bottleneck,
            'min_bottleneck': min(client_bottlenecks),
            'max_bottleneck': max(client_bottlenecks),
            'bottleneck_std': np.std(client_bottlenecks),
            'compression_ratio': (256 - avg_bottleneck) / 256,
            'test_snr': test_snr,
            'test_bottleneck': test_bottleneck,
            'dataset': dataset_name
        }
        
        # Log comprehensive round data
        logger.log_round(rnd, global_metrics, client_round_data, channel_conditions)

        print(f"Round {rnd:02d} │ MSE={mse_mean:.4f} │ PSNR={psnr_mean:.2f} dB │ "
              f"Avg Loss={avg_client_loss:.4f} │ Loss Std={loss_std:.4f} │ "
              f"Avg Bottleneck={avg_bottleneck:.1f}")

        # Visual check every 4 rounds
        if rnd % 4 == 0:
            visual_check_adaptive(global_model, val_loader, rnd, device_profiles)

        print("-" * 80)

    print("\nAdaptive training completed!")
    
    # Save all logged data
    logger.save_data()
    
    # Final comprehensive visual check
    print("Generating final adaptive compression comparison...")
    visual_check_adaptive(global_model, val_loader, ROUNDS, device_profiles, "final_adaptive")
    
    print(f"\n🎯 Experiment completed! Data saved to: {logger.exp_dir}")
    print("📊 Ready for advanced plotting and analysis!")
    
    return global_model, logger


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    trained_model, experiment_logger = main()
    print(f"\n🎯 Run complete! Check {experiment_logger.exp_dir} for detailed data files.")
    print("📈 Ready for advanced plotting and analysis!")