"""
Adaptive Voiced/Unvoiced Weighting Module for Speech Enhancement
Implements a learnable importance network that adapts frequency-band weights
based on voiced/unvoiced characteristics of the input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveVUVWeightingModule(nn.Module):
    """
    Adaptive VUV weighting module that learns frequency-band importance weights
    
    This module extracts VUV features from reference signals and predicts
    adaptive weights for each frequency band, enabling end-to-end optimization
    of the loss weighting strategy.
    
    Args:
        n_fft: FFT size for STFT
        n_freq_bands: Number of frequency bands to divide the spectrum
    """
    def __init__(self, n_fft=512, n_freq_bands=8):
        super().__init__()
        self.n_freq_bands = n_freq_bands
        self.freq_bins = n_fft // 2 + 1  # 257 for n_fft=512
        self.bins_per_band = self.freq_bins // n_freq_bands
        
        # Lightweight MLP to predict frequency-band importance weights
        # Input: n_freq_bands * 2 (voiced and unvoiced energy per band)
        # Output: n_freq_bands * 2 (importance weight for voiced and unvoiced per band)
        self.importance_net = nn.Sequential(
            nn.Linear(n_freq_bands * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_freq_bands * 2),
            nn.Sigmoid()  # Output weights in [0, 1]
        )
        
        # Initialize weights to produce near-uniform importance initially
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights to produce near-uniform importance
        This ensures stable training in early iterations
        """
        for layer in self.importance_net:
            if isinstance(layer, nn.Linear):
                # Xavier initialization with small gain
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    # Bias initialized to 0.5 so sigmoid outputs ~0.62
                    nn.init.constant_(layer.bias, 0.5)
    
    def extract_vuv_features(self, voiced_mag, unvoiced_mag):
        """
        Extract frequency-band energy features from VUV spectrograms
        
        Args:
            voiced_mag: (B, F, T) Voiced reference STFT magnitude
            unvoiced_mag: (B, F, T) Unvoiced reference STFT magnitude
        
        Returns:
            features: (B, n_freq_bands * 2) Band energy features
        """
        B, F, T = voiced_mag.shape
        
        # Compute average energy per frequency band
        voiced_band_energy = []
        unvoiced_band_energy = []
        
        for i in range(self.n_freq_bands):
            start = i * self.bins_per_band
            end = min((i + 1) * self.bins_per_band, F)
            
            # Average energy in this band (across frequency and time)
            voiced_band = voiced_mag[:, start:end, :].pow(2).mean(dim=[1, 2])  # (B,)
            unvoiced_band = unvoiced_mag[:, start:end, :].pow(2).mean(dim=[1, 2])  # (B,)
            
            voiced_band_energy.append(voiced_band)
            unvoiced_band_energy.append(unvoiced_band)
        
        # Stack into feature tensor (B, n_freq_bands)
        voiced_features = torch.stack(voiced_band_energy, dim=1)
        unvoiced_features = torch.stack(unvoiced_band_energy, dim=1)
        
        # Log compression for numerical stability
        voiced_features = torch.log1p(voiced_features)
        unvoiced_features = torch.log1p(unvoiced_features)
        
        # Concatenate: [voiced_band_0..7, unvoiced_band_0..7]
        features = torch.cat([voiced_features, unvoiced_features], dim=1)  # (B, n_freq_bands * 2)
        
        return features
    
    def forward(self, pred_mag, clean_mag, voiced_mag, unvoiced_mag, vuv_threshold=2.0):
        """
        Forward pass: compute adaptive weighted frequency-domain loss
        
        Args:
            pred_mag: (B, F, T) Predicted audio STFT magnitude
            clean_mag: (B, F, T) Clean audio STFT magnitude
            voiced_mag: (B, F, T) Voiced reference STFT magnitude
            unvoiced_mag: (B, F, T) Unvoiced reference STFT magnitude
            vuv_threshold: Energy ratio threshold for VUV detection
        
        Returns:
            loss: Scalar weighted loss
            attention_weights: (B, n_freq_bands, 2) Learned weights for visualization
        """
        B, F, T = pred_mag.shape
        
        # Step 1: Extract VUV features
        vuv_features = self.extract_vuv_features(voiced_mag, unvoiced_mag)  # (B, n_freq_bands*2)
        
        # Step 2: Predict importance weights via MLP
        importance_weights = self.importance_net(vuv_features)  # (B, n_freq_bands * 2)
        
        # Scale weights to [0.3, 2.0] to avoid zeros and extreme values
        importance_weights = 0.3 + importance_weights * 1.7
        
        # Step 3: Split into voiced and unvoiced weights
        voiced_weights = importance_weights[:, :self.n_freq_bands]      # (B, n_freq_bands)
        unvoiced_weights = importance_weights[:, self.n_freq_bands:]    # (B, n_freq_bands)
        
        # Step 4: Compute VUV mask based on energy ratio
        # mask = 1 for voiced, 0 for unvoiced
        unvoiced_energy = unvoiced_mag.pow(2)
        voiced_energy = voiced_mag.pow(2)
        vuv_mask = (unvoiced_energy <= vuv_threshold * voiced_energy).float()
        
        # Step 5: Build frequency-band weight map
        weight_map = torch.zeros(B, F, T, device=pred_mag.device)
        
        for i in range(self.n_freq_bands):
            start = i * self.bins_per_band
            end = min((i + 1) * self.bins_per_band, F)
            
            band_mask = vuv_mask[:, start:end, :]
            
            # Apply voiced weight where mask=1, unvoiced weight where mask=0
            band_weight = (band_mask * voiced_weights[:, i:i+1].unsqueeze(-1) +
                          (1 - band_mask) * unvoiced_weights[:, i:i+1].unsqueeze(-1))
            
            weight_map[:, start:end, :] = band_weight
        
        # Step 6: Compute weighted loss
        mag_diff = torch.abs(pred_mag - clean_mag)
        weighted_loss = (mag_diff * weight_map).mean()
        
        # Step 7: Return loss and attention weights (for visualization)
        attention_weights = torch.stack([voiced_weights, unvoiced_weights], dim=2)  # (B, n_freq_bands, 2)
        
        return weighted_loss, attention_weights


def adaptive_vuv_loss(pred, clean, unvoiced, voiced, 
                      importance_module, 
                      n_fft=512, hop_size=160, win_length=400,
                      vuv_threshold=2.0):
    """
    Compute adaptive VUV-weighted loss using the importance module
    
    Args:
        pred: (B, 1, L) or (B, L) Predicted audio
        clean: (B, 1, L) or (B, L) Clean audio
        unvoiced: (B, 1, L) or (B, L) Unvoiced reference
        voiced: (B, 1, L) or (B, L) Voiced reference
        importance_module: AdaptiveVUVWeightingModule instance
        n_fft, hop_size, win_length: STFT parameters
        vuv_threshold: VUV detection threshold
    
    Returns:
        loss: Scalar loss value
        attention_weights: (B, n_freq_bands, 2) Learned weights
    """
    # Ensure 2D: (B, L)
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if clean.dim() == 3:
        clean = clean.squeeze(1)
    if unvoiced.dim() == 3:
        unvoiced = unvoiced.squeeze(1)
    if voiced.dim() == 3:
        voiced = voiced.squeeze(1)
    
    # Compute STFT for all signals
    window = torch.hann_window(win_length, device=pred.device)
    
    pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop_size, 
                           win_length=win_length, window=window, 
                           return_complex=True)
    clean_stft = torch.stft(clean, n_fft=n_fft, hop_length=hop_size,
                            win_length=win_length, window=window,
                            return_complex=True)
    voiced_stft = torch.stft(voiced, n_fft=n_fft, hop_length=hop_size,
                             win_length=win_length, window=window,
                             return_complex=True)
    unvoiced_stft = torch.stft(unvoiced, n_fft=n_fft, hop_length=hop_size,
                               win_length=win_length, window=window,
                               return_complex=True)
    
    # Compute magnitude spectrograms
    pred_mag = torch.abs(pred_stft)
    clean_mag = torch.abs(clean_stft)
    voiced_mag = torch.abs(voiced_stft)
    unvoiced_mag = torch.abs(unvoiced_stft)
    
    # Compute adaptive weighted loss via importance module
    loss, attention_weights = importance_module(
        pred_mag, clean_mag, voiced_mag, unvoiced_mag,
        vuv_threshold=vuv_threshold
    )
    
    return loss, attention_weights


def visualize_attention_weights(attention_weights, iteration, save_dir='./vis'):
    """
    Visualize learned importance weights for analysis
    
    Args:
        attention_weights: (B, n_freq_bands, 2) Tensor
            [:, :, 0] = voiced weights
            [:, :, 1] = unvoiced weights
        iteration: Current training iteration
        save_dir: Directory to save visualization
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Average across batch
    weights = attention_weights.mean(dim=0).cpu().numpy()  # (n_freq_bands, 2)
    n_bands = weights.shape[0]
    
    # Frequency band labels
    freq_bands = [f"{i}k-{i+1}k" for i in range(n_bands)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_bands)
    width = 0.35
    
    ax.bar(x - width/2, weights[:, 0], width, label='Voiced Weight', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, weights[:, 1], width, label='Unvoiced Weight', alpha=0.8, color='coral')
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Learned Weight')
    ax.set_title(f'Adaptive VUV Weights at Iteration {iteration}')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_bands, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2.5])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_weights_iter{iteration}.png', dpi=150)
    plt.close()
    
    print(f"Attention weights visualization saved to {save_dir}/attention_weights_iter{iteration}.png")
