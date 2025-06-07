"""
FedLoL IoT v12 - LEARNED Channel-Aware & Device-Adaptive Semantic Compression
End-to-end learned adaptation replacing rule-based bottleneck selection
Preserves original semantic communication chain with learned enhancement
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

# ADAPTIVE COMPRESSION PARAMETERS
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
    
    def __init__(self, experiment_name="learned_adaptive_fedlol", base_dir="./experiments"):
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
            'adaptation_type': 'learned',
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
# Device Profiling System
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
        
        # Individual channel characteristics
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
# Learned Adaptation Component (NEW - replaces rule-based selection)
# ------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """Encodes context information for learned adaptation"""
    def __init__(self, context_dim=32):
        super().__init__()
        self.context_dim = context_dim
        
    def encode_context(self, snr_db, device_profile, energy_level=1.0):
        """Convert context to tensor for neural network"""
        # Normalize inputs to [0, 1] range for stable training
        snr_normalized = max(0.0, min(snr_db / 20.0, 1.0))  # Assume max 20 dB
        max_bottleneck = device_profile['max_bottleneck']
        min_bottleneck = device_profile['min_bottleneck'] 
        power_budget = device_profile['power_budget']
        
        # Create context vector
        context = torch.tensor([
            snr_normalized,
            energy_level,
            max_bottleneck / MAX_BOTTLENECK,  # Normalized device capability
            min_bottleneck / MAX_BOTTLENECK,  # Normalized minimum
            power_budget,                     # Power budget
            # Add device type one-hot encoding
            1.0 if device_profile == DEVICE_PROFILES['high_end'] else 0.0,
            1.0 if device_profile == DEVICE_PROFILES['mid_range'] else 0.0,
            1.0 if device_profile == DEVICE_PROFILES['low_power'] else 0.0,
            1.0 if device_profile == DEVICE_PROFILES['ultra_low'] else 0.0,
        ], dtype=torch.float32)
        
        return context


class LearnedBottleneckPredictor(nn.Module):
    """Small network that learns optimal bottleneck selection"""
    def __init__(self, context_dim=9, hidden_dim=32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, context):
        """Predict optimal bottleneck ratio for given context"""
        ratio = self.predictor(context)
        return ratio


def get_learned_bottleneck(snr_db, device_profile, energy_level, bottleneck_predictor, context_encoder, round_num=1):
    """
    Learned replacement for rule-based bottleneck selection with exploration
    
    Args:
        snr_db: Current channel SNR in dB
        device_profile: Device capability profile
        energy_level: Battery level (0.0-1.0)
        bottleneck_predictor: Learned predictor network
        context_encoder: Context encoding helper
        round_num: Current training round (for exploration)
    
    Returns:
        Optimal bottleneck size
    """
    # Encode context
    context = context_encoder.encode_context(snr_db, device_profile, energy_level)
    
    # Move context to same device as model
    context = context.to(next(bottleneck_predictor.parameters()).device)
    
    # Predict optimal bottleneck ratio
    with torch.no_grad():  # No gradients during inference
        bottleneck_ratio = bottleneck_predictor(context.unsqueeze(0)).item()
    
    # EXPLORATION MECHANISM (NEW): Add controlled noise in early rounds
    if round_num <= 5:
        # Add exploration noise to prevent getting stuck
        exploration_noise = random.uniform(-0.2, 0.2) * (6 - round_num) / 5.0
        bottleneck_ratio = max(0.0, min(1.0, bottleneck_ratio + exploration_noise))
    
    # Convert ratio to actual bottleneck size within device constraints
    max_bottleneck = device_profile['max_bottleneck']
    min_bottleneck = device_profile['min_bottleneck']
    
    # Scale ratio to device range
    bottleneck_size = min_bottleneck + bottleneck_ratio * (max_bottleneck - min_bottleneck)
    bottleneck_size = int(bottleneck_size)
    
    # Clamp to valid range
    bottleneck_size = max(MIN_BOTTLENECK, min(bottleneck_size, MAX_BOTTLENECK))
    
    return bottleneck_size

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


class LearnedAdaptiveIoTSemanticComm(nn.Module):
    """Enhanced IoT Semantic Communication with LEARNED adaptive compression"""
    def __init__(self):
        super().__init__()
        self.enc_s = AdaptiveSemanticEncoder()
        self.enc_c = ChannelEncoder()
        self.dec_c = ChannelDecoder()
        self.dec_s = AdaptiveSemanticDecoder()
        
        # NEW: Add learned adaptation components
        self.context_encoder = ContextEncoder()
        self.bottleneck_predictor = LearnedBottleneckPredictor()

    def forward(self, img, snr_db=SNR_DB, bottleneck_size=DEFAULT_BOTTLENECK):
        # Semantic encoding with adaptive bottleneck (unchanged)
        z, skips = self.enc_s(img, bottleneck_size)
        
        # Pad semantic features for channel processing if needed (unchanged)
        if bottleneck_size < MAX_BOTTLENECK:
            padding = torch.zeros(z.shape[0], MAX_BOTTLENECK - bottleneck_size, 
                                device=z.device, dtype=z.dtype)
            z_padded = torch.cat([z, padding], dim=1)
        else:
            z_padded = z
        
        # Channel encoding (unchanged)
        x = self.enc_c(z_padded, snr_db)

        # Channel simulation (unchanged from original)
        sigma = math.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        h = torch.randn_like(x)
        noise = sigma * torch.randn_like(x)
        y = h * x + noise
        x_hat = y / (h + 1e-6)

        # Channel decoding (unchanged)
        z_hat_padded = self.dec_c(x_hat, snr_db)
        
        # Extract relevant semantic features (unchanged)
        z_hat = z_hat_padded[:, :bottleneck_size]
        
        # Semantic reconstruction (unchanged)
        return self.dec_s(z_hat, skips, bottleneck_size)
    
    def predict_bottleneck(self, snr_db, device_profile, energy_level=1.0, round_num=1):
        """Use learned network to predict optimal bottleneck size"""
        return get_learned_bottleneck(
            snr_db, device_profile, energy_level, 
            self.bottleneck_predictor, self.context_encoder, round_num
        )

    def get_model_size(self):
        param_size = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return param_size


# ------------------------------------------------------------------
# Explicit Bottleneck Predictor Training (CRITICAL FIX)
# ------------------------------------------------------------------
def train_bottleneck_predictor(model, context_info, bottleneck_used, performance_metrics, predictor_optimizer):
    """
    Explicitly train the bottleneck predictor based on adaptation effectiveness
    
    Args:
        model: The semantic communication model
        context_info: Context that led to the bottleneck decision
        bottleneck_used: The bottleneck size that was actually used
        performance_metrics: Dict with PSNR, loss, etc. from using this bottleneck
        predictor_optimizer: Optimizer for the bottleneck predictor
    """
    # Encode the context that led to this decision
    context = model.context_encoder.encode_context(
        context_info['snr'], 
        context_info['device_profile'], 
        context_info['energy']
    ).to(next(model.bottleneck_predictor.parameters()).device)
    
    # Get what the predictor would have chosen
    predicted_ratio = model.bottleneck_predictor(context.unsqueeze(0))
    
    # Convert to actual bottleneck size for comparison
    device_max = context_info['device_profile']['max_bottleneck']
    device_min = context_info['device_profile']['min_bottleneck']
    predicted_bottleneck = device_min + predicted_ratio.item() * (device_max - device_min)
    predicted_bottleneck = int(predicted_bottleneck)
    
    # Calculate adaptation quality score
    # Good adaptation = high PSNR + appropriate resource usage
    psnr = performance_metrics.get('psnr', 0)
    efficiency = (device_max - bottleneck_used) / device_max  # Higher = more efficient
    
    # Context appropriateness score
    snr_db = context_info['snr']
    energy_level = context_info['energy']
    
    # Ideal bottleneck ratio based on context
    snr_factor = max(0, min(1, snr_db / 20.0))
    energy_factor = energy_level
    ideal_ratio = (snr_factor + energy_factor) / 2.0
    ideal_bottleneck = device_min + ideal_ratio * (device_max - device_min)
    
    # Performance-based reward (higher PSNR is better)
    performance_reward = psnr / 35.0  # Normalize assuming ~35 dB max PSNR
    
    # Efficiency reward (using fewer resources when possible)
    efficiency_reward = efficiency * 0.3
    
    # Context matching reward (did we adapt appropriately?)
    context_match = 1.0 - abs(bottleneck_used - ideal_bottleneck) / (device_max - device_min)
    context_reward = context_match * 0.2
    
    # Total adaptation quality score
    adaptation_quality = performance_reward + efficiency_reward + context_reward
    
    # Training target: We want predictor to output the ratio that leads to this quality
    target_ratio = (bottleneck_used - device_min) / (device_max - device_min)
    target_ratio = torch.tensor(target_ratio, dtype=torch.float32, device=predicted_ratio.device)
    
    # Weight the training by adaptation quality (learn more from good decisions)
    quality_weight = max(0.1, adaptation_quality)  # Don't completely ignore bad decisions
    
    # Loss: Weighted MSE between predicted and target ratio
    predictor_loss = quality_weight * nn.functional.mse_loss(predicted_ratio.squeeze(), target_ratio)
    
    # Update predictor weights
    predictor_optimizer.zero_grad()
    predictor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.bottleneck_predictor.parameters(), max_norm=1.0)
    predictor_optimizer.step()
    
    return {
        'predictor_loss': predictor_loss.item(),
        'adaptation_quality': adaptation_quality,
        'predicted_bottleneck': predicted_bottleneck,
        'target_bottleneck': bottleneck_used,
        'quality_weight': quality_weight
    }


def evaluate_bottleneck_performance(model, img_batch, snr_db, bottleneck_size):
    """
    Evaluate how well a specific bottleneck size performs
    
    Returns:
        Dict with performance metrics (PSNR, loss, etc.)
    """
    model.eval()
    with torch.no_grad():
        recon = model(img_batch, snr_db, bottleneck_size)
        
        # Calculate performance metrics
        mse = nn.functional.mse_loss(recon, img_batch)
        psnr = 10.0 * math.log10(1.0 / max(mse.item(), 1e-10))
        loss = hybrid_loss(recon, img_batch)
        
        return {
            'psnr': psnr,
            'mse': mse.item(),
            'loss': loss.item()
        }
def adaptive_semantic_loss(pred, target, bottleneck_size, context_info, device_profile, round_num, alpha: float = ALPHA_LOSS):
    """
    Multi-objective loss that encourages both quality and intelligent adaptation
    
    Args:
        pred: Reconstructed image
        target: Original image  
        bottleneck_size: Used bottleneck size
        context_info: Dict with SNR, energy, etc.
        device_profile: Device capability info
        round_num: Current training round
        alpha: Loss weighting factor
    """
    # Base reconstruction loss (unchanged)
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    recon_loss = alpha * mse_term + (1.0 - alpha) * l1_term
    
    # Extract context
    snr_db = context_info['snr']
    energy_level = context_info['energy']
    device_max = device_profile['max_bottleneck']
    device_min = device_profile['min_bottleneck']
    
    # 1. EFFICIENCY REWARD: Encourage using smaller bottlenecks when possible
    efficiency_reward = (device_max - bottleneck_size) / device_max * 0.05
    
    # 2. CONTEXT APPROPRIATENESS PENALTIES: Penalize poor adaptation decisions
    context_penalty = 0.0
    
    # Poor SNR penalty: Using large bottleneck with bad channel
    if snr_db < 5 and bottleneck_size > device_max * 0.7:
        context_penalty += 0.1 * (bottleneck_size / device_max)
    
    # Low energy penalty: Using large bottleneck with low battery
    if energy_level < 0.3 and bottleneck_size > device_max * 0.6:
        context_penalty += 0.08 * (bottleneck_size / device_max)
    
    # Under-utilization penalty: Using tiny bottleneck with great conditions
    if snr_db > 15 and energy_level > 0.8 and bottleneck_size < device_max * 0.4:
        context_penalty += 0.05
    
    # 3. ADAPTATION INCENTIVE: Encourage different decisions for different contexts
    adaptation_incentive = 0.0
    
    # Reward for using device range appropriately (not always same size)
    bottleneck_ratio = (bottleneck_size - device_min) / (device_max - device_min)
    
    # Ideal adaptation: poor conditions → small ratio, good conditions → large ratio
    snr_factor = max(0, min(1, snr_db / 20.0))  # Normalize to 0-1
    energy_factor = energy_level
    ideal_ratio = (snr_factor + energy_factor) / 2.0
    
    # Reward for matching ideal adaptation pattern
    adaptation_match = 1.0 - abs(bottleneck_ratio - ideal_ratio)
    adaptation_incentive = adaptation_match * 0.03
    
    # 4. EXPLORATION BONUS: Encourage exploration in early rounds
    exploration_bonus = 0.0
    if round_num <= 5:
        # Bonus for using diverse bottleneck sizes during learning phase
        exploration_bonus = 0.02
    
    # Combine all components
    total_loss = (recon_loss 
                  - efficiency_reward 
                  + context_penalty 
                  - adaptation_incentive 
                  - exploration_bonus)
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'efficiency_reward': efficiency_reward,
        'context_penalty': context_penalty,
        'adaptation_incentive': adaptation_incentive,
        'exploration_bonus': exploration_bonus
    }


def hybrid_loss(pred, target, alpha: float = ALPHA_LOSS):
    """Original loss function (kept for compatibility)"""
    mse_term = nn.functional.mse_loss(pred, target)
    l1_term = nn.functional.l1_loss(pred, target)
    return alpha * mse_term + (1.0 - alpha) * l1_term


# ------------------------------------------------------------------
# Enhanced Training Functions
# ------------------------------------------------------------------
def dynamic_epochs():
    """Each client randomly decides training epochs (1-5)"""
    return random.choice([0, 0, 0, 1, 2, 3, 4, 5])  # Weighted toward fewer epochs

def local_train_adaptive(model, loader, epochs: int, client_id: int, device_profile, snr_db=SNR_DB, round_num=1):
    """Local training with EXPLICIT PREDICTOR TRAINING"""
    model.train()
    
    # SEPARATE OPTIMIZERS (CRITICAL FIX)
    # Main model optimizer (semantic + channel components)
    main_optimizer = optim.Adam([
        {'params': model.enc_s.parameters()},
        {'params': model.enc_c.parameters()},
        {'params': model.dec_c.parameters()},
        {'params': model.dec_s.parameters()}
    ], lr=LR, weight_decay=1e-4)
    
    # Bottleneck predictor optimizer (EXPLICIT TRAINING)
    predictor_optimizer = optim.Adam([
        {'params': model.context_encoder.parameters()},
        {'params': model.bottleneck_predictor.parameters()}
    ], lr=LR * 2.0, weight_decay=1e-4)  # Slightly higher LR for adaptation learning
    
    total_loss = 0
    total_predictor_metrics = {
        'predictor_loss': 0, 'adaptation_quality': 0, 'quality_weight': 0
    }
    total_loss_components = {
        'recon_loss': 0, 'efficiency_reward': 0, 'context_penalty': 0,
        'adaptation_incentive': 0, 'exploration_bonus': 0
    }
    num_batches = 0
    bottleneck_decisions = []
    performance_history = []
    
    # Simulate energy depletion during training
    energy_level = device_profile['energy_level']
    
    for epoch in range(epochs):
        for batch_idx, (img, _) in enumerate(loader):
            img = img.to(device)
            
            # Get current context
            current_energy = energy_level * (1.0 - 0.1 * epoch / max(epochs, 1))
            context_info = {
                'snr': snr_db,
                'energy': current_energy,
                'device_profile': device_profile['profile'],
                'round': round_num
            }
            
            # STEP 1: Get bottleneck prediction
            bottleneck_size = model.predict_bottleneck(snr_db, device_profile['profile'], current_energy, round_num)
            bottleneck_decisions.append(bottleneck_size)
            
            # STEP 2: Train main model with this bottleneck
            main_optimizer.zero_grad()
            recon = model(img, snr_db, bottleneck_size)
            
            # Enhanced adaptive loss for main model
            loss, loss_components = adaptive_semantic_loss(
                recon, img, bottleneck_size, context_info, device_profile['profile'], round_num
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            main_optimizer.step()
            
            # STEP 3: EXPLICITLY TRAIN BOTTLENECK PREDICTOR (CRITICAL NEW STEP)
            # Evaluate performance of the bottleneck decision we just made
            performance_metrics = evaluate_bottleneck_performance(model, img, snr_db, bottleneck_size)
            performance_history.append(performance_metrics)
            
            # Train predictor based on how well this decision worked
            predictor_metrics = train_bottleneck_predictor(
                model, context_info, bottleneck_size, performance_metrics, predictor_optimizer
            )
            
            # Track metrics
            total_loss += loss.item()
            for key, value in loss_components.items():
                total_loss_components[key] += value
            for key, value in predictor_metrics.items():
                if key in total_predictor_metrics:
                    total_predictor_metrics[key] += value
            
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for key in total_loss_components:
        total_loss_components[key] /= num_batches if num_batches > 0 else 1
    for key in total_predictor_metrics:
        total_predictor_metrics[key] /= num_batches if num_batches > 0 else 1
    
    # Calculate adaptation metrics
    bottleneck_diversity = len(set(bottleneck_decisions)) if bottleneck_decisions else 1
    bottleneck_std = np.std(bottleneck_decisions) if bottleneck_decisions else 0
    avg_psnr = np.mean([p['psnr'] for p in performance_history]) if performance_history else 0
    
    # Get final bottleneck for reporting
    final_bottleneck = model.predict_bottleneck(snr_db, device_profile['profile'], current_energy, round_num)
    
    print(f"  Client {client_id+1} ({device_profile['type']}): {epochs} epochs, "
          f"SNR={snr_db:.1f}dB, Bottleneck={final_bottleneck}, Loss={avg_loss:.4f}")
    print(f"    📊 Adaptation: {bottleneck_diversity} unique sizes, Std={bottleneck_std:.1f}, Avg PSNR={avg_psnr:.1f}")
    print(f"    🧠 Predictor: Loss={total_predictor_metrics['predictor_loss']:.4f}, "
          f"Quality={total_predictor_metrics['adaptation_quality']:.3f}")
    print(f"    🎯 Main Loss: Recon={total_loss_components['recon_loss']:.4f}, "
          f"Efficiency={total_loss_components['efficiency_reward']:.3f}")
    
    return avg_loss, final_bottleneck, total_loss_components, total_predictor_metrics


# ------------------------------------------------------------------
# Enhanced FedLoL Aggregation
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
def visual_check_adaptive(model, val_loader, round_num, device_profiles, save_path="learned_adaptive_progress"):
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
        
        plt.suptitle(f'Round {round_num:02d} - Learned Adaptive Compression')
        plt.tight_layout()
        plt.savefig(f'{save_path}_round_{round_num:02d}.png', dpi=150, bbox_inches='tight')
        plt.show()


# ------------------------------------------------------------------
# Main Training Loop (Enhanced with Learned Adaptation)
# ------------------------------------------------------------------
def main():
    print("🧠 FedLoL IoT v12 - EXPLICIT PREDICTOR TRAINING Adaptive Semantic Compression")
    print("🚀 CRITICAL FIX: Direct bottleneck predictor training implemented!")
    print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {BATCH_SIZE} batch size")
    print(f"Adaptive bottleneck range: {MIN_BOTTLENECK}-{MAX_BOTTLENECK} dimensions")
    print("")
    print("🎯 EXPLICIT PREDICTOR TRAINING Features:")
    print("   • Separate optimizer for bottleneck predictor")
    print("   • Performance-based predictor training")
    print("   • Adaptation quality scoring")
    print("   • Context-effectiveness learning")
    print("   • Real-time predictor loss tracking")
    print("")
    print("🧠 Training Architecture:")
    print("   • Main Model: Trains semantic + channel components")
    print("   • Predictor: Trains adaptation decisions based on performance")
    print("   • Dual optimization: Both networks learn simultaneously")
    print("")
    print("📈 Expected Behavior:")
    print("   • Rounds 1-3: Predictor learns from performance feedback")
    print("   • Rounds 4-6: Strong SNR↔Bottleneck correlations emerge")
    print("   • Rounds 7+: Intelligent context-aware adaptation")
    print("")
    
    # Initialize comprehensive logging
    logger = AdaptiveFedLoLLogger(
        experiment_name=f"explicit_predictor_training_fedlol_{NUM_CLIENTS}clients_{ROUNDS}rounds",
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
        'device': str(device),
        'adaptation_type': 'learned'
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

    # Initialize LEARNED adaptive model
    global_model = LearnedAdaptiveIoTSemanticComm().to(device)
    print(f"Model size: {global_model.get_model_size():.2f} MB")
    print(f"Total parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print("📚 Using LEARNED bottleneck adaptation (will improve with training!)")
    print("🔄 Early rounds: network learns optimal adaptation policies")
    print("🎯 Later rounds: should outperform rule-based approach")

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
    
    print(f"\nStarting LEARNED adaptive federated training for {ROUNDS} rounds...")
    print("🧠 Network will learn optimal bottleneck policies during training!")
    print("=" * 80)

    # Training loop with comprehensive logging
    for rnd in range(1, ROUNDS + 1):
        print(f"\nRound {rnd:02d}")
        
        client_states, client_losses, client_bottlenecks, client_snrs = [], [], [], []
        client_round_data = []

        # Each client trains with individual channel conditions and ENHANCED LEARNED adaptive compression
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
                
            loss_val, bottleneck_used, loss_components, predictor_metrics = local_train_adaptive(
                local_model, client_loaders[cid], epochs, cid, device_profiles[cid], current_snr, rnd
            )
            
            client_states.append(local_model.state_dict())
            client_losses.append(loss_val)
            client_bottlenecks.append(bottleneck_used)
            client_snrs.append(current_snr)
            
            # Detailed client data for logging (enhanced with loss components)
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
                'adaptation_factor': bottleneck_used / device_profiles[cid]['profile']['max_bottleneck'],
                # Loss component tracking
                'recon_loss': loss_components['recon_loss'],
                'efficiency_reward': loss_components['efficiency_reward'],
                'context_penalty': loss_components['context_penalty'],
                'adaptation_incentive': loss_components['adaptation_incentive'],
                # NEW: Predictor training tracking
                'predictor_loss': predictor_metrics['predictor_loss'],
                'adaptation_quality': predictor_metrics['adaptation_quality'],
                'quality_weight': predictor_metrics['quality_weight']
            })

        # Enhanced FedLoL aggregation
        adaptive_fedlol_aggregate(global_model, client_states, client_losses, client_bottlenecks)
        
        # ADAPTATION ANALYSIS (ENHANCED): Monitor learning progress
        bottleneck_diversity = len(set(client_bottlenecks))
        bottleneck_range = max(client_bottlenecks) - min(client_bottlenecks)
        bottleneck_std = np.std(client_bottlenecks)
        
        # Context vs Bottleneck correlation analysis
        snr_bottleneck_corr = np.corrcoef(client_snrs, client_bottlenecks)[0, 1] if len(client_snrs) > 1 else 0
        
        # Predictor training analysis
        avg_predictor_loss = np.mean([cd['predictor_loss'] for cd in client_round_data])
        avg_adaptation_quality = np.mean([cd['adaptation_quality'] for cd in client_round_data])
        avg_quality_weight = np.mean([cd['quality_weight'] for cd in client_round_data])
        
        print(f"  🧠 Adaptation Learning:")
        print(f"    Bottleneck diversity: {bottleneck_diversity} unique sizes")
        print(f"    Bottleneck range: {min(client_bottlenecks)}-{max(client_bottlenecks)} (spread: {bottleneck_range})")
        print(f"    SNR-Bottleneck correlation: {snr_bottleneck_corr:.3f}")
        
        print(f"  🎯 Predictor Training:")
        print(f"    Avg predictor loss: {avg_predictor_loss:.4f}")
        print(f"    Avg adaptation quality: {avg_adaptation_quality:.3f}")
        print(f"    Avg quality weight: {avg_quality_weight:.3f}")
        
        # Learning progress assessment
        if snr_bottleneck_corr > 0.3:
            print(f"    ✅ Excellent! Strong SNR→Bottleneck learning")
        elif snr_bottleneck_corr > 0.1:
            print(f"    🔄 Good progress! SNR adaptation developing")
        elif abs(snr_bottleneck_corr) < 0.1:
            print(f"    ⚠️  Warning: No clear SNR adaptation yet")
        else:
            print(f"    🔄 Learning in progress...")
            
        if avg_adaptation_quality > 0.7:
            print(f"    ✅ High adaptation quality achieved!")
        elif avg_adaptation_quality > 0.5:
            print(f"    🔄 Moderate adaptation quality")
        else:
            print(f"    ⚠️  Low adaptation quality - predictor still learning")
        
        # Channel condition summary
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

        # Validation with LEARNED adaptive model
        global_model.eval()
        with torch.no_grad():
            mse_sum, n_img = 0.0, 0
            
            # Use learned prediction for validation bottleneck
            # Average context across clients for validation
            avg_energy = np.mean([dp['energy_level'] for dp in device_profiles])
            avg_device_profile = device_profiles[len(device_profiles)//2]['profile']  # Use median device
            test_snr = np.mean(client_snrs)
            test_bottleneck = global_model.predict_bottleneck(test_snr, avg_device_profile, avg_energy, rnd)
            
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

    print("\nEXPLICIT PREDICTOR TRAINING completed!")
    
    # Save all logged data
    logger.save_data()
    
    # Final comprehensive visual check
    print("Generating final explicit predictor training compression comparison...")
    visual_check_adaptive(global_model, val_loader, ROUNDS, device_profiles, "final_explicit_predictor_training")
    
    print(f"\n🎯 EXPLICIT PREDICTOR TRAINING experiment completed! Data saved to: {logger.exp_dir}")
    print("📊 Ready for advanced plotting and analysis!")
    print("🧠 Predictor should now show REAL learning with:")
    print("   ✅ Strong SNR-bottleneck correlation (>0.3)")
    print("   ✅ High adaptation quality scores (>0.7)")
    print("   ✅ Decreasing predictor loss over rounds")
    print("   ✅ Context-appropriate decision patterns")
    
    return global_model, logger


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    trained_model, experiment_logger = main()
    print(f"\n🎯 EXPLICIT PREDICTOR TRAINING complete! Check {experiment_logger.exp_dir} for detailed data files.")
    print("📈 Ready for advanced plotting and analysis!")
    print("🧠 Predictor network should now demonstrate REAL learning behavior!")
    print("")
    print("📊 Key success indicators:")
    print("   • SNR-Bottleneck correlation > 0.3 by Round 5")
    print("   • Bottleneck diversity > 3 unique sizes")
    print("   • Adaptation quality > 0.7 by Round 7")
    print("   • Decreasing predictor loss over rounds")
    print("   • Context-appropriate decision patterns")