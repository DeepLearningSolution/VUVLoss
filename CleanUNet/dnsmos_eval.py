"""
DNS-MOS Evaluation Script for Speech Enhancement
Computes SIG, BAK, and OVRL scores using Microsoft's DNSMOS model
"""

import os
import argparse
import glob
import numpy as np
import soundfile as sf
from tqdm import tqdm
import requests
import onnxruntime as ort


class DNSMOSEvaluator:
    """
    DNS-MOS P.835 Evaluator
    
    Computes three scores:
    - SIG: Signal quality (speech distortion)
    - BAK: Background quality (noise level)
    - OVRL: Overall quality
    """
    
    def __init__(self, primary_model_path=None, use_gpu=False):
        """
        Initialize DNS-MOS evaluator
        
        Args:
            primary_model_path: Path to ONNX model file
            use_gpu: Whether to use GPU for inference
        """
        self.sr = 16000  # DNS-MOS requires 16kHz
        self.INPUT_LENGTH = 9.01  # Input audio length in seconds
        
        # Use local DNSMOS models if not specified
        if primary_model_path is None:
            primary_model_path = 'dnsmos_models/sig_bak_ovr.onnx'
            
        if not os.path.exists(primary_model_path):
            print(f"Error: DNSMOS model not found at {primary_model_path}")
            print("Please ensure DNSMOS is downloaded to the DNSMOS directory")
            raise FileNotFoundError(f"Model not found: {primary_model_path}")
        
        # Initialize ONNX runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.onnx_sess = ort.InferenceSession(primary_model_path, 
                                               sess_options=sess_options,
                                               providers=providers)
    
    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """
        Polynomial fitting for MOS scores
        
        This calibration is from the official DNSMOS implementation
        """
        if is_personalized_MOS:
            # Personalized MOS (P.808)
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            # General MOS (P.835)
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        
        return sig_poly, bak_poly, ovr_poly
    
    def predict(self, audio, sr=16000):
        """
        Predict DNS-MOS scores for audio
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate
            
        Returns:
            dict: {'SIG': float, 'BAK': float, 'OVRL': float}
        """
        import librosa
        
        # Resample if needed
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=-1)
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Pad audio if too short
        len_samples = int(self.INPUT_LENGTH * self.sr)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        # Process audio in segments
        num_hops = int(np.floor(len(audio) / self.sr) - self.INPUT_LENGTH) + 1
        hop_len_samples = self.sr
        
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        
        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            
            # Prepare input features (raw waveform)
            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            
            # ONNX inference
            oi = {'input_1': input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            
            # Apply polynomial calibration (P.835)
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS=False
            )
            
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
        
        # Average scores across segments
        avg_sig = np.mean(predicted_mos_sig_seg) if predicted_mos_sig_seg else np.nan
        avg_bak = np.mean(predicted_mos_bak_seg) if predicted_mos_bak_seg else np.nan
        avg_ovr = np.mean(predicted_mos_ovr_seg) if predicted_mos_ovr_seg else np.nan
        
        # Clip to valid range [1, 5]
        avg_sig = np.clip(avg_sig, 1.0, 5.0)
        avg_bak = np.clip(avg_bak, 1.0, 5.0)
        avg_ovr = np.clip(avg_ovr, 1.0, 5.0)
        
        return {
            'SIG': float(avg_sig),
            'BAK': float(avg_bak),
            'OVRL': float(avg_ovr)
        }


def evaluate_directory(audio_dir, output_file=None, use_gpu=False):
    """
    Evaluate all audio files in a directory
    
    Args:
        audio_dir: Directory containing audio files
        output_file: Path to save results
        use_gpu: Whether to use GPU
        
    Returns:
        dict: Results for all files
    """
    # Initialize evaluator
    try:
        evaluator = DNSMOSEvaluator(use_gpu=use_gpu)
    except Exception as e:
        print(f"Failed to initialize DNSMOS evaluator: {e}")
        print("\nPlease install required packages:")
        print("  pip install librosa requests onnxruntime")
        return None
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}")
        return None
    
    print(f"Found {len(audio_files)} audio files")
    
    # Evaluate each file
    results = {}
    sig_scores = []
    bak_scores = []
    ovr_scores = []
    
    for audio_file in tqdm(audio_files, desc="Evaluating"):
        try:
            # Load audio
            audio, sr = sf.read(audio_file)
            
            # Predict scores
            scores = evaluator.predict(audio, sr)
            
            # Store results
            filename = os.path.basename(audio_file)
            results[filename] = scores
            
            sig_scores.append(scores['SIG'])
            bak_scores.append(scores['BAK'])
            ovr_scores.append(scores['OVRL'])
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Compute statistics
    results['mean'] = {
        'SIG': float(np.mean(sig_scores)),
        'BAK': float(np.mean(bak_scores)),
        'OVRL': float(np.mean(ovr_scores))
    }
    
    results['std'] = {
        'SIG': float(np.std(sig_scores)),
        'BAK': float(np.std(bak_scores)),
        'OVRL': float(np.std(ovr_scores))
    }
    
    # Print results
    print("\n" + "="*60)
    print("DNS-MOS Evaluation Results")
    print("="*60)
    print(f"SIG:  {results['mean']['SIG']:.3f} ± {results['std']['SIG']:.3f}")
    print(f"BAK:  {results['mean']['BAK']:.3f} ± {results['std']['BAK']:.3f}")
    print(f"OVRL: {results['mean']['OVRL']:.3f} ± {results['std']['OVRL']:.3f}")
    print("="*60)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            f.write("DNS-MOS Evaluation Results\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"SIG:  {results['mean']['SIG']:.3f} ± {results['std']['SIG']:.3f}\n")
            f.write(f"BAK:  {results['mean']['BAK']:.3f} ± {results['std']['BAK']:.3f}\n")
            f.write(f"OVRL: {results['mean']['OVRL']:.3f} ± {results['std']['OVRL']:.3f}\n\n")
            
            f.write("Per-file results:\n")
            f.write("-"*60 + "\n")
            
            for filename in sorted(results.keys()):
                if filename not in ['mean', 'std']:
                    scores = results[filename]
                    f.write(f"{filename}: SIG={scores['SIG']:.3f}, "
                           f"BAK={scores['BAK']:.3f}, OVRL={scores['OVRL']:.3f}\n")
        
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio quality using DNS-MOS")
    parser.add_argument('audio_dir', type=str, help="Directory containing audio files to evaluate")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file to save results")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed")
        print("Install with: pip install librosa")
        exit(1)
    
    # Evaluate
    evaluate_directory(args.audio_dir, args.output, args.gpu)
