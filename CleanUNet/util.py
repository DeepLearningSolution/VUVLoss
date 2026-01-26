import os
import time 
import functools
import numpy as np
from math import cos, pi, floor, sin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from stft_loss import MultiResolutionSTFTLoss


def flatten(v):
    return [x for y in v for x in y]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find latest checkpoint
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            number = f[:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")
        
        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")
        
        print(" ")


####################### lr scheduler: Linear Warmup then Cosine Decay #############################

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch

# Original Copyright 2019 Kim Seonghyeon
#  MIT License (https://opensource.org/licenses/MIT)


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cosine(start, end, proportion):
    cos_val = cos(pi * proportion) + 1
    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, cur_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = cur_iter

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class LinearWarmupCosineDecay:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        iteration=0,
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cosine'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cosine': anneal_cosine}

        cur_iter_phase1 = iteration
        cur_iter_phase2 = max(0, iteration - phase1)
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, cur_iter_phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, cur_iter_phase2, phase_map[phase[1]]),
        ]

        if iteration < phase1:
            self.phase = 0
        else:
            self.phase = 1

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr


####################### model util #############################

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


@torch.no_grad()
def sampling(net, noisy_audio):
    """
    Perform denoising (forward) step
    """

    return net(noisy_audio)


def loss_fn(net, X, ell_p, ell_p_lambda, stft_lambda, mrstftloss, 
             use_vuv_loss=False,
             time_vuv_loss=False,         # NEW: Enable VUV weighting in time domain only
             freq_vuv_loss=False,         # NEW: Enable VUV weighting in frequency domain only
             vuv_frame_size=80, vuv_time_thd=2.0, vuv_freq_thd=2.0,
             time_voiced_weight=2.0, time_unvoiced_weight=0.5,
             freq_low_voiced_weight=1.5, freq_low_unvoiced_weight=0.3,
             freq_high_voiced_weight=0.5, freq_high_unvoiced_weight=2.0,
             use_adaptive_vuv=False,      # NEW: Enable adaptive VUV
             adaptive_vuv_lambda=2.0,     # NEW: Adaptive VUV weight
             importance_module=None,      # NEW: Importance module instance
             adaptive_vuv_config=None,    # NEW: Adaptive VUV config dict
             **kwargs):
    """
    Loss function in CleanUNet with optional VUV-weighted loss
    
    Supports both fixed VUV weighting and adaptive VUV weighting strategies

    Parameters:
    net: network
    X: training data tuple (clean_audio, noisy_audio) or (clean_audio, noisy_audio, unvoiced_audio, voiced_audio)
    ell_p: ell_p norm (1 or 2) of the AE loss
    ell_p_lambda: factor of the AE loss
    stft_lambda: factor of the STFT loss
    mrstftloss: multi-resolution STFT loss function
    use_vuv_loss: whether to use fixed VUV weighting
    use_adaptive_vuv: whether to use adaptive VUV weighting (NEW)
    adaptive_vuv_lambda: weight for adaptive VUV loss (NEW)
    importance_module: adaptive VUV weighting module (NEW)
    adaptive_vuv_config: config dict for adaptive VUV (NEW)
    vuv_frame_size: frame size for time-domain VUV detection
    vuv_time_thd: threshold for time-domain VUV detection
    vuv_freq_thd: threshold for freq-domain VUV detection
    time_voiced_weight: weight for voiced frames in time-domain loss
    time_unvoiced_weight: weight for unvoiced frames in time-domain loss
    freq_low_voiced_weight: weight for voiced frames in low-freq loss
    freq_low_unvoiced_weight: weight for unvoiced frames in low-freq loss
    freq_high_voiced_weight: weight for voiced frames in high-freq loss
    freq_high_unvoiced_weight: weight for unvoiced frames in high-freq loss

    Returns:
    loss: value of objective function
    output_dic: values of each component of loss
    """

    if len(X) == 2:
        clean_audio, noisy_audio = X
        unvoiced_audio, voiced_audio = None, None
    elif len(X) == 4:
        clean_audio, noisy_audio, unvoiced_audio, voiced_audio = X
    else:
        raise ValueError(f"X should be tuple of length 2 or 4, got {len(X)}")
    
    B, C, L = clean_audio.shape
    output_dic = {}
    loss = 0.0
    
    voiced_mask = None
    # Apply time-domain VUV weighting only if time_vuv_loss is explicitly enabled
    if (time_vuv_loss or (use_vuv_loss and not freq_vuv_loss)) and \
       unvoiced_audio is not None and voiced_audio is not None:
        from vuv_utils import compute_time_domain_vuv_mask, compute_weighted_time_loss
        voiced_mask = compute_time_domain_vuv_mask(
            unvoiced_audio, voiced_audio,
            frame_size=vuv_frame_size,
            threshold=vuv_time_thd
        )
    
    denoised_audio = net(noisy_audio)

    if ell_p == 2:
        ae_loss_per_sample = (denoised_audio - clean_audio) ** 2
    elif ell_p == 1:
        ae_loss_per_sample = torch.abs(denoised_audio - clean_audio)
    else:
        raise NotImplementedError
    
    if voiced_mask is not None:
        ae_loss = compute_weighted_time_loss(
            ae_loss_per_sample, voiced_mask,
            voiced_weight=time_voiced_weight,
            unvoiced_weight=time_unvoiced_weight
        )
    else:
        ae_loss = ae_loss_per_sample.mean()
    
    loss += ae_loss * ell_p_lambda
    output_dic["reconstruct"] = ae_loss.data * ell_p_lambda

    # ============ 2. Frequency-domain loss ============
    
    # Option A: Adaptive VUV loss (NEW)
    if use_adaptive_vuv and importance_module is not None and \
       unvoiced_audio is not None and voiced_audio is not None:
        from adaptive_vuv_loss import adaptive_vuv_loss
        
        # Get config parameters
        if adaptive_vuv_config is None:
            adaptive_vuv_config = {}
        
        adaptive_loss, attention_weights = adaptive_vuv_loss(
            denoised_audio, clean_audio,
            unvoiced_audio, voiced_audio,
            importance_module=importance_module,
            n_fft=adaptive_vuv_config.get('n_fft', 512),
            hop_size=adaptive_vuv_config.get('hop_size', 160),
            win_length=adaptive_vuv_config.get('win_length', 400),
            vuv_threshold=adaptive_vuv_config.get('vuv_threshold', vuv_freq_thd)
        )
        
        loss += adaptive_loss * adaptive_vuv_lambda
        output_dic["adaptive_vuv"] = adaptive_loss.data * adaptive_vuv_lambda
        output_dic["attention_weights"] = attention_weights  # For logging/visualization
    
    # Option B: Traditional multi-resolution STFT loss
    if stft_lambda > 0:
        # Apply frequency-domain VUV weighting only if freq_vuv_loss is explicitly enabled
        if (freq_vuv_loss or (use_vuv_loss and not time_vuv_loss)) and \
           not use_adaptive_vuv and unvoiced_audio is not None and voiced_audio is not None:
            # Fixed VUV-weighted STFT loss
            sc_loss, mag_loss = mrstftloss(
                denoised_audio.squeeze(1), clean_audio.squeeze(1),
                unvoiced_audio=unvoiced_audio,
                voiced_audio=voiced_audio,
                vuv_threshold=vuv_freq_thd,
                freq_low_voiced_weight=freq_low_voiced_weight,
                freq_low_unvoiced_weight=freq_low_unvoiced_weight,
                freq_high_voiced_weight=freq_high_voiced_weight,
                freq_high_unvoiced_weight=freq_high_unvoiced_weight
            )
        else:
            # Standard STFT loss
            sc_loss, mag_loss = mrstftloss(denoised_audio.squeeze(1), clean_audio.squeeze(1))
        
        loss += (sc_loss + mag_loss) * stft_lambda
        output_dic["stft_sc"] = sc_loss.data * stft_lambda
        output_dic["stft_mag"] = mag_loss.data * stft_lambda

    return loss, output_dic

