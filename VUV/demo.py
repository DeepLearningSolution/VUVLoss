#!/usr/bin/env python3
"""
demo.py - WORLD voice analysis and synthesis demo program
"""

import numpy as np
import scipy.io.wavfile as wavfile
import world_separate
import os

def demo1_process_file():
    """Demo 1: Direct audio file processing"""
    print("=== Demo 1: Process audio file ===")
    
    # Get current file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup input/output file paths
    input_file = os.path.join(current_dir, "input.wav")
    output_file_ch0 = os.path.join(current_dir, "output_demo1_ch0.wav")
    output_file_ch1 = os.path.join(current_dir, "output_demo1_ch1.wav")
    output_file_ch2 = os.path.join(current_dir, "output_demo1_ch2.wav")
    
    # Display file path info
    print(f"Current script directory: {current_dir}")
    print(f"Input file path: {input_file}")
    print(f"Output file paths:")
    print(f"  - Channel 0: {output_file_ch0}")
    print(f"  - Channel 1: {output_file_ch1}")
    print(f"  - Channel 2: {output_file_ch2}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist, creating test file...")
        
        # Create test audio file
        fs = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(fs * duration))
        
        # Generate test signal (speech-like signal)
        f0 = 100  # Fundamental frequency 100Hz
        signal = (0.5 * np.sin(2 * np.pi * f0 * t) +        # Fundamental
                 0.3 * np.sin(2 * np.pi * f0 * 2 * t) +     # 2nd harmonic
                 0.2 * np.sin(2 * np.pi * f0 * 3 * t) +     # 3rd harmonic
                 0.1 * np.sin(2 * np.pi * f0 * 4 * t))      # 4th harmonic
        
        # Add envelope and noise
        envelope = 0.8 * (1 + 0.3 * np.sin(2 * np.pi * 1.5 * t))
        noise = 0.02 * np.random.randn(len(signal))
        signal = signal * envelope + noise
        
        # Save test file
        signal_int = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(input_file, fs, signal_int)
        print(f"✓ Test file created: {input_file}")
    
    try:
        # Read audio file
        print(f"Reading audio file: {input_file}")
        fs, audio_data = wavfile.read(input_file)
        
        # Convert to float64 format [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float64) / 2147483648.0
        elif audio_data.dtype == np.float32:
            audio_float = audio_data.astype(np.float64)
        else:
            audio_float = audio_data.astype(np.float64)
        
        # If stereo, take first channel
        if len(audio_float.shape) > 1:
            audio_float = audio_float[:, 0]
            print("✓ Stereo detected, using first channel")
        
        print(f"✓ Audio file info:")
        print(f"  - Sample rate: {fs} Hz")
        print(f"  - Duration: {len(audio_float)/fs:.2f} seconds")
        print(f"  - Sample count: {len(audio_float)}")
        print(f"  - Data type: {audio_data.dtype}")
        
        # Perform WORLD analysis and synthesis
        print("Performing WORLD analysis and synthesis...")
        original_length = len(audio_float)
        result_ch0, result_ch1, result_ch2 = world_separate.analyze_synthesize(
            audio_data=audio_float,
            fs=fs,
            frame_period=5.0  
        )
        
        result_ch0 = result_ch0[:original_length]
        result_ch1 = result_ch1[:original_length]
        result_ch2 = result_ch2[:original_length]
        
        print(f"✓ WORLD processing complete")
        print(f"  - Channel 0 output samples: {len(result_ch0)}")
        print(f"  - Channel 1 output samples: {len(result_ch1)}")
        print(f"  - Channel 2 output samples: {len(result_ch2)}")
        print(f"  - Channel 0 output duration: {len(result_ch0)/fs:.2f} seconds")
        print(f"  - Channel 1 output duration: {len(result_ch1)/fs:.2f} seconds")
        print(f"  - Channel 2 output duration: {len(result_ch2)/fs:.2f} seconds")
        
        # Convert back to integer format and save three channels
        result_ch0_int = np.clip(result_ch0 * 32767, -32768, 32767).astype(np.int16)
        result_ch1_int = np.clip(result_ch1 * 32767, -32768, 32767).astype(np.int16)
        result_ch2_int = np.clip(result_ch2 * 32767, -32768, 32767).astype(np.int16)
        
        wavfile.write(output_file_ch0, fs, result_ch0_int)
        wavfile.write(output_file_ch1, fs, result_ch1_int)
        wavfile.write(output_file_ch2, fs, result_ch2_int)
        
        print(f"✓ Processing complete, output files:")
        print(f"  - Channel 0: {output_file_ch0}")
        print(f"  - Channel 1: {output_file_ch1}")
        print(f"  - Channel 2: {output_file_ch2}")
        
        # Calculate statistics for each channel
        original_rms = np.sqrt(np.mean(audio_float**2))
        result_ch0_rms = np.sqrt(np.mean(result_ch0**2))
        result_ch1_rms = np.sqrt(np.mean(result_ch1**2))
        result_ch2_rms = np.sqrt(np.mean(result_ch2**2))
        
        print(f"✓ Audio statistics:")
        print(f"  - Original RMS: {original_rms:.6f}")
        print(f"  - Channel 0 RMS: {result_ch0_rms:.6f} (Ratio: {result_ch0_rms/original_rms:.3f})")
        print(f"  - Channel 1 RMS: {result_ch1_rms:.6f} (Ratio: {result_ch1_rms/original_rms:.3f})")
        print(f"  - Channel 2 RMS: {result_ch2_rms:.6f} (Ratio: {result_ch2_rms/original_rms:.3f})")
        
        # Display channel information
        print(f"✓ Channel descriptions:")
        print(f"  - Channel 0 (mode=0): Original WORLD synthesis")
        print(f"  - Channel 1 (mode=1): Alternative synthesis mode 1")
        print(f"  - Channel 2 (mode=2): Alternative synthesis mode 2")
        
        return True
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return False

def main():
    """Main function"""
    print("WORLD Voice Analysis and Synthesis Demo Program")
    print("=" * 50)
    
    # Run demo
    demo1_process_file()
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    
    # Get current file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List generated files
    output_files = [
        "input.wav",
        "output_demo1_ch0.wav",
        "output_demo1_ch1.wav", 
        "output_demo1_ch2.wav"
    ]
    
    print("Generated files:")
    for file in output_files:
        full_path = os.path.join(current_dir, file)
        if os.path.exists(full_path):
            print(f"  ✓ {full_path}")

if __name__ == "__main__":
    main()