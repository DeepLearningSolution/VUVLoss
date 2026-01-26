# VUV (Voiced/Unvoiced) Utilities for Loss Computation
# Copyright (c) 2026

import torch
import torch.nn.functional as F
import numpy as np


def compute_time_domain_vuv_mask(unvoiced_audio, voiced_audio, frame_size=80, threshold=2.0):
    """
    Compute time-domain voiced/unvoiced mask.
    
    Args:
        unvoiced_audio: Reference unvoiced audio (B, L) or (B, 1, L)
        voiced_audio: Reference voiced audio (B, L) or (B, 1, L)
        frame_size: Frame length for energy computation
        threshold: Energy ratio threshold for VUV detection
    
    Returns:
        voiced_mask: (B, L), 1=voiced, 0=unvoiced
    """
    if len(unvoiced_audio.shape) == 3:
        unvoiced_audio = unvoiced_audio.squeeze(1)
    if len(voiced_audio.shape) == 3:
        voiced_audio = voiced_audio.squeeze(1)
    
    B, L = unvoiced_audio.shape
    device = unvoiced_audio.device
    
    voiced_mask = torch.ones(B, L, device=device)
    
    n_frames = L // frame_size
    
    for b in range(B):
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            
            unvoiced_energy = torch.sum(unvoiced_audio[b, start:end] ** 2)
            voiced_energy = torch.sum(voiced_audio[b, start:end] ** 2)
            
            if unvoiced_energy > threshold * voiced_energy:
                voiced_mask[b, start:end] = 0  # Unvoiced
        
        if n_frames * frame_size < L:
            start = n_frames * frame_size
            unvoiced_energy = torch.sum(unvoiced_audio[b, start:] ** 2)
            voiced_energy = torch.sum(voiced_audio[b, start:] ** 2)
            if unvoiced_energy > threshold * voiced_energy:
                voiced_mask[b, start:] = 0  # Unvoiced
    
    return voiced_mask


def compute_freq_domain_vuv_mask(unvoiced_audio, voiced_audio, 
                                  fft_size, hop_size, win_length, 
                                  threshold=2.0):
    """
    Compute frequency-domain voiced/unvoiced mask.
    
    Args:
        unvoiced_audio: Reference unvoiced audio (B, L) or (B, 1, L)
        voiced_audio: Reference voiced audio (B, L) or (B, 1, L)
        fft_size: FFT size
        hop_size: Hop size
        win_length: Window length
        threshold: Energy ratio threshold for VUV detection
    
    Returns:
        voiced_mask: (B, freq_bins, frames), 1=voiced, 0=unvoiced
    """
    if len(unvoiced_audio.shape) == 3:
        unvoiced_audio = unvoiced_audio.squeeze(1)
    if len(voiced_audio.shape) == 3:
        voiced_audio = voiced_audio.squeeze(1)
    
    device = unvoiced_audio.device
    
    window = torch.hann_window(win_length, device=device)
    
    unvoiced_stft = torch.stft(
        unvoiced_audio, 
        n_fft=fft_size, 
        hop_length=hop_size, 
        win_length=win_length,
        window=window,
        return_complex=True
    )
    voiced_stft = torch.stft(
        voiced_audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    
    unvoiced_mag = torch.abs(unvoiced_stft)  # (B, freq_bins, frames)
    voiced_mag = torch.abs(voiced_stft)      # (B, freq_bins, frames)
    
    unvoiced_energy = unvoiced_mag ** 2
    voiced_energy = voiced_mag ** 2
    
    # voiced_mask: 1=voiced, 0=unvoiced
    voiced_mask = (unvoiced_energy <= threshold * voiced_energy).float()
    
    return voiced_mask


def compute_weighted_time_loss(loss_per_sample, voiced_mask, voiced_weight=2.0, unvoiced_weight=0.5):
    """
    Apply VUV-based weighting to time-domain loss.
    
    Args:
        loss_per_sample: (B, C, L), per-sample loss values
        voiced_mask: (B, L), 1=voiced, 0=unvoiced
        voiced_weight: Weight for voiced frames
        unvoiced_weight: Weight for unvoiced frames
    
    Returns:
        Weighted average loss (scalar)
    """
    if voiced_mask.dim() == 2:
        voiced_mask = voiced_mask.unsqueeze(1)  # (B, 1, L)
    
    # Create weight map: voiced_weight for voiced, unvoiced_weight for unvoiced
    weight_map = voiced_mask * voiced_weight + (1 - voiced_mask) * unvoiced_weight
    
    weighted_loss = loss_per_sample * weight_map
    return weighted_loss.mean()


def compute_weighted_freq_loss(loss_per_bin, voiced_mask, 
                                 voiced_weight_low=1.5, unvoiced_weight_low=0.3,
                                 voiced_weight_high=0.5, unvoiced_weight_high=2.0,
                                 freq_split_ratio=0.5):
    """
    Apply VUV-based weighting to frequency-domain loss with frequency-dependent weights.
    
    Args:
        loss_per_bin: (B, frames, freq_bins), per-bin loss values (T x F format from stft())
        voiced_mask: (B, frames, freq_bins), 1=voiced, 0=unvoiced (T x F format)
        voiced_weight_low: Weight for voiced frames in low frequency
        unvoiced_weight_low: Weight for unvoiced frames in low frequency
        voiced_weight_high: Weight for voiced frames in high frequency
        unvoiced_weight_high: Weight for unvoiced frames in high frequency
        freq_split_ratio: Ratio to split low/high frequency (0.5 = half-half)
    
    Returns:
        Weighted average loss (scalar)
    """
    B, T, F = loss_per_bin.shape  # Updated: now expects (B, T, F) format
    freq_split = int(F * freq_split_ratio)
    
    # Low frequency part (frequency is last dimension)
    loss_low = loss_per_bin[:, :, :freq_split]  # (B, T, F_low)
    mask_low = voiced_mask[:, :, :freq_split]
    weight_map_low = mask_low * voiced_weight_low + (1 - mask_low) * unvoiced_weight_low
    weighted_loss_low = (loss_low * weight_map_low).mean()
    
    # High frequency part
    loss_high = loss_per_bin[:, :, freq_split:]  # (B, T, F_high)
    mask_high = voiced_mask[:, :, freq_split:]
    weight_map_high = mask_high * voiced_weight_high + (1 - mask_high) * unvoiced_weight_high
    weighted_loss_high = (loss_high * weight_map_high).mean()
    
    return (weighted_loss_low + weighted_loss_high) / 2


if __name__ == "__main__":

    print("Testing VUV utilities...")
    
    B, L = 2, 16000  
    unvoiced = torch.randn(B, L) * 0.1 
    voiced = torch.randn(B, L) * 1.0  
    
    time_mask = compute_time_domain_vuv_mask(unvoiced, voiced, frame_size=80, threshold=2.0)
    print(f"Time domain mask shape: {time_mask.shape}")
    print(f"Voiced ratio: {time_mask.mean().item():.2%}")
    

    freq_mask = compute_freq_domain_vuv_mask(
        unvoiced, voiced,
        fft_size=512, hop_size=160, win_length=400,
        threshold=2.0
    )
    print(f"Freq domain mask shape: {freq_mask.shape}")
    print(f"Unvoiced ratio: {freq_mask.mean().item():.2%}")
    
    print("VUV utilities test passed!")
