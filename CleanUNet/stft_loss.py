# Adapted from https://github.com/kan-bayashi/ParallelWaveGAN

# Original Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        
        Returns:
            Tensor: Spectral convergence loss value.
            
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        
        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", 
        band="full"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band 

        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2  # only select high frequency bands
            sc_loss  = self.spectral_convergence_loss(x_mag[:,freq_mask_ind:,:], y_mag[:,freq_mask_ind:,:])
            mag_loss = self.log_stft_magnitude_loss(x_mag[:,freq_mask_ind:,:], y_mag[:,freq_mask_ind:,:])
        elif self.band == "full":
            sc_loss  = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag) 
        else: 
            raise NotImplementedError

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240],
        window="hann_window", sc_lambda=0.1, mag_lambda=0.1, band="full"
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            *_lambda (float): a balancing factor across different losses.
            band (str): high-band or full-band loss

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, band)]

    def forward(self, x, y, unvoiced_audio=None, voiced_audio=None, vuv_threshold=2.0,
                freq_low_voiced_weight=1.5, freq_low_unvoiced_weight=0.3,
                freq_high_voiced_weight=0.5, freq_high_unvoiced_weight=2.0):
        """Calculate forward propagation with VUV-weighted loss.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).
            unvoiced_audio (Tensor, optional): Unvoiced reference signal for VUV masking.
            voiced_audio (Tensor, optional): Voiced reference signal for VUV masking.
            vuv_threshold (float): Threshold for VUV detection.
            freq_low_voiced_weight (float): Weight for voiced frames in low frequency.
            freq_low_unvoiced_weight (float): Weight for unvoiced frames in low frequency.
            freq_high_voiced_weight (float): Weight for voiced frames in high frequency.
            freq_high_unvoiced_weight (float): Weight for unvoiced frames in high frequency.

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
            if unvoiced_audio is not None:
                unvoiced_audio = unvoiced_audio.view(-1, unvoiced_audio.size(2))
            if voiced_audio is not None:
                voiced_audio = voiced_audio.view(-1, voiced_audio.size(2))
        
        sc_loss = 0.0
        mag_loss = 0.0
        
        for f in self.stft_losses:
            if unvoiced_audio is not None and voiced_audio is not None:
                from vuv_utils import compute_freq_domain_vuv_mask, compute_weighted_freq_loss
                
                voiced_mask = compute_freq_domain_vuv_mask(
                    unvoiced_audio, voiced_audio,
                    fft_size=f.fft_size,
                    hop_size=f.shift_size,
                    win_length=f.win_length,
                    threshold=vuv_threshold
                )
                
                x_mag = stft(x, f.fft_size, f.shift_size, f.win_length, f.window)
                y_mag = stft(y, f.fft_size, f.shift_size, f.win_length, f.window)
                
                if f.band == "high":
                    freq_mask_ind = x_mag.shape[1] // 2
                    x_mag = x_mag[:, freq_mask_ind:, :]
                    y_mag = y_mag[:, freq_mask_ind:, :]
                    voiced_mask = voiced_mask[:, freq_mask_ind:, :]
                
                sc_loss_per_bin = torch.abs(x_mag - y_mag) / (y_mag + 1e-8)
                mag_loss_per_bin = torch.abs(torch.log(y_mag + 1e-8) - torch.log(x_mag + 1e-8))
                
                sc_l = compute_weighted_freq_loss(
                    sc_loss_per_bin, voiced_mask,
                    voiced_weight_low=freq_low_voiced_weight,
                    unvoiced_weight_low=freq_low_unvoiced_weight,
                    voiced_weight_high=freq_high_voiced_weight,
                    unvoiced_weight_high=freq_high_unvoiced_weight,
                    freq_split_ratio=0.5
                )
                
                mag_l = compute_weighted_freq_loss(
                    mag_loss_per_bin, voiced_mask,
                    voiced_weight_low=freq_low_voiced_weight,
                    unvoiced_weight_low=freq_low_unvoiced_weight,
                    voiced_weight_high=freq_high_voiced_weight,
                    unvoiced_weight_high=freq_high_unvoiced_weight,
                    freq_split_ratio=0.5
                )
            else:
                sc_l, mag_l = f(x, y)
            
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss *= self.sc_lambda
        sc_loss /= len(self.stft_losses)
        mag_loss *= self.mag_lambda
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
