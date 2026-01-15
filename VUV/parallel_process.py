#!/usr/bin/env python3
"""
Parallel WORLD processing script
Process all wav files in input directory and output voiced/unvoiced channels
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import world_separate
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

def process_single_file(args: Tuple[str, str, int, float]) -> Tuple[str, bool, str]:
    """
    Process a single audio file
    
    Args:
        args: (input_file, output_dir, fs, frame_period)
    
    Returns:
        (input_file, success, message)
    """
    input_file, output_dir, fs, frame_period = args
    
    try:
        # Read audio file
        fs_input, audio_data = wavfile.read(input_file)
        
        # Convert to float64 format [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float64) / 2147483648.0
        elif audio_data.dtype == np.float32:
            audio_float = audio_data.astype(np.float64)
        else:
            audio_float = audio_data.astype(np.float64)
        
        # Handle stereo
        if len(audio_float.shape) > 1:
            audio_float = audio_float[:, 0]
        
        original_length = len(audio_float)
        
        # WORLD analysis and synthesis
        result_ch0, result_ch1, result_ch2 = world_separate.analyze_synthesize(
            audio_data=audio_float,
            fs=fs if fs is not None else fs_input,
            frame_period=frame_period
        )
        
        # Align to original length
        result_ch0 = result_ch0[:original_length]
        result_ch1 = result_ch1[:original_length]
        
        # Prepare output filenames
        input_path = Path(input_file)
        base_name = input_path.stem
        
        # ch0 -> unvoiced (in unvoiced/ subdirectory), ch1 -> voiced (in voiced/ subdirectory)
        unvoiced_dir = os.path.join(output_dir, "unvoiced")
        voiced_dir = os.path.join(output_dir, "voiced")
        
        output_file_ch0 = os.path.join(unvoiced_dir, f"{base_name}.wav")
        output_file_ch1 = os.path.join(voiced_dir, f"{base_name}.wav")
        
        # Convert to int16 and save
        target_fs = fs if fs is not None else fs_input
        
        result_ch0_int = np.clip(result_ch0 * 32767, -32768, 32767).astype(np.int16)
        result_ch1_int = np.clip(result_ch1 * 32767, -32768, 32767).astype(np.int16)
        
        wavfile.write(output_file_ch0, target_fs, result_ch0_int)
        wavfile.write(output_file_ch1, target_fs, result_ch1_int)
        
        return (input_file, True, f"OK -> {base_name}.wav")
        
    except Exception as e:
        return (input_file, False, f"Error: {str(e)}")


def batch_process(input_dir: str,
                  output_dir: str,
                  fs: int = None,
                  frame_period: float = 5.0,
                  num_workers: int = None,
                  file_pattern: str = "*.wav") -> None:
    """
    Batch process all wav files in parallel
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        fs: Target sample rate (None = use original)
        frame_period: Frame period in milliseconds
        num_workers: Number of parallel workers (None = auto detect)
        file_pattern: File matching pattern
    """
    # Create output directories (main + subdirectories)
    os.makedirs(output_dir, exist_ok=True)
    unvoiced_dir = os.path.join(output_dir, "unvoiced")
    voiced_dir = os.path.join(output_dir, "voiced")
    os.makedirs(unvoiced_dir, exist_ok=True)
    os.makedirs(voiced_dir, exist_ok=True)
    
    # Get all input files and sort by filename
    input_files = sorted(list(Path(input_dir).glob(file_pattern)))
    
    if not input_files:
        print(f"Error: No files matching {file_pattern} found in {input_dir}")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print("=" * 70)
    print("WORLD Parallel Processing - Voiced/Unvoiced Separation")
    print("=" * 70)
    print(f"Input directory:   {input_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"  - Unvoiced dir:  {unvoiced_dir}")
    print(f"  - Voiced dir:    {voiced_dir}")
    print(f"File count:        {len(input_files)}")
    print(f"Parallel workers:  {num_workers}")
    print(f"Sample rate:       {fs if fs else 'original'} Hz")
    print(f"Frame period:      {frame_period} ms")
    print("=" * 70)
    print(f"Output structure:")
    print(f"  {output_dir}/")
    print(f"  ├── unvoiced/")
    print(f"  │   └── *.wav (channel 0 - unvoiced)")
    print(f"  └── voiced/")
    print(f"      └── *.wav (channel 1 - voiced)")
    print("=" * 70)
    
    # Prepare task arguments
    tasks = [(str(f), output_dir, fs, frame_period) for f in input_files]
    
    # Start parallel processing
    start_time = time.time()
    
    print("\nProcessing...")
    with Pool(processes=num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_single_file, tasks), 1):
            input_file, success, message = result
            status = "[OK]" if success else "[FAIL]"
            filename = Path(input_file).name
            #print(f"  [{i:4d}/{len(input_files):4d}] {status} {filename}")
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Statistics
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = len(results) - success_count
    
    print("=" * 70)
    print("Processing completed!")
    print(f"Total time:       {elapsed_time:.2f} seconds")
    print(f"Success:          {success_count} files")
    print(f"Failed:           {fail_count} files")
    print(f"Average speed:    {elapsed_time/len(input_files):.2f} sec/file")
    print(f"Throughput:       {len(input_files)/elapsed_time:.2f} files/sec")
    print("=" * 70)


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description='Parallel WORLD processing for voiced/unvoiced separation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_dir output_dir
  %(prog)s input_dir output_dir --workers 32 --fs 16000
  %(prog)s input_dir output_dir --frame-period 2.5 --workers 60
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing wav files')
    parser.add_argument('output_dir', help='Output directory for processed files')
    parser.add_argument('--fs', type=int, default=None,
                       help='Target sample rate in Hz (default: use original)')
    parser.add_argument('--frame-period', type=float, default=5.0,
                       help='Frame period in milliseconds (default: 5.0)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--pattern', default='*.wav',
                       help='File pattern to match (default: *.wav)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Run batch processing
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        frame_period=args.frame_period,
        num_workers=args.workers,
        file_pattern=args.pattern
    )


if __name__ == "__main__":
    main()
