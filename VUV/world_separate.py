"""
world_separate.py - Python wrapper for WORLD voice analysis/synthesis
"""

import ctypes
import numpy as np
import os
from ctypes import POINTER, c_double, c_int, c_void_p, CDLL
from typing import Union, Tuple
import warnings
from scipy import signal

class WorldSeparateError(Exception):
    """WORLD Separate module exception"""
    pass

def apply_highpass_filter(audio_data: np.ndarray, 
                          fs: int, 
                          cutoff: float = 50.0, 
                          order: int = 5) -> np.ndarray:
    """
    Apply high-pass filter to remove frequencies below cutoff
    
    Args:
        audio_data: Audio data, numpy array, float64 type
        fs: Sample rate (Hz)
        cutoff: Cutoff frequency (Hz), default 50Hz
        order: Filter order, default 5
    
    Returns:
        Filtered audio data with same length as input
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Nyquist frequency
    nyquist = 0.5 * fs
    
    # Normalized cutoff frequency
    normal_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if normal_cutoff >= 1.0:
        warnings.warn(f"Cutoff frequency {cutoff}Hz is too high for sampling rate {fs}Hz, skipping filter")
        return audio_data
    
    if normal_cutoff <= 0:
        warnings.warn(f"Invalid cutoff frequency {cutoff}Hz, skipping filter")
        return audio_data
    
    try:
        # Design Butterworth high-pass filter
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        
        # Apply filter using filtfilt to avoid phase distortion and maintain alignment
        # filtfilt applies filter forward and backward, so output length = input length
        filtered_data = signal.filtfilt(b, a, audio_data)
        
        return filtered_data.astype(np.float64)
        
    except Exception as e:
        warnings.warn(f"High-pass filter failed: {e}, returning original data")
        return audio_data

class WorldSeparate:
    """WORLD voice analysis and synthesis module"""
    
    def __init__(self, lib_path: str = None):
        """
        Initialize WORLD analysis and synthesis module
        
        Args:
            lib_path: Dynamic library path, if None then auto-search
        """
        self._lib = None
        # 获取当前文件所在目录并组合路径
        if lib_path is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_lib_path = os.path.join(current_dir, lib_path)
            self._load_library(full_lib_path)
        else:
            self._load_library(None)
        self._setup_functions()
    
    def _load_library(self, lib_path: str = None):
        """Load dynamic library"""
        if lib_path is None:
            # Get current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Auto-search library files
            search_paths = [
                os.path.join(current_dir, 'World/build/lib/libworld_separate.so'),
                os.path.join(current_dir, 'World/build/lib/libworld_separate.dylib'),
                os.path.join(current_dir, 'World/build/lib/release/libworld_separate.so'),
                os.path.join(current_dir, 'World/build/lib/release/libworld_separate.dylib')
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
            
            if lib_path is None:
                raise WorldSeparateError(f"Cannot find libworld_separate dynamic library. Searched paths:\n" + 
                                       "\n".join(f"  - {p}" for p in search_paths))
        
        try:
            self._lib = CDLL(lib_path)
        except OSError as e:
            raise WorldSeparateError(f"Cannot load dynamic library {lib_path}: {e}")
    
    def _setup_functions(self):
        """Setup function signatures"""
        # SeparateAndSynthesize_C function
        self._lib.SeparateAndSynthesize_C.argtypes = [
            POINTER(c_double),  # x
            c_int,              # x_length
            c_int,              # fs
            c_double,           # frame_period
            POINTER(POINTER(c_double)),  # y0_out
            POINTER(c_int),             # y0_length_out
            POINTER(POINTER(c_double)),  # y1_out
            POINTER(c_int),             # y1_length_out
            POINTER(POINTER(c_double)),  # y2_out
            POINTER(c_int)              # y2_length_out
        ]
        self._lib.SeparateAndSynthesize_C.restype = None
        
        # free_audio_data function
        self._lib.free_audio_data.argtypes = [POINTER(c_double)]
        self._lib.free_audio_data.restype = None

    def analyze_synthesize(self, 
                          audio_data: np.ndarray, 
                          fs: int = 48000, 
                          frame_period: float = 5.0,
                          apply_highpass: bool = False,
                          highpass_cutoff: float = 50.0) -> tuple:
        """
        Perform voice analysis and synthesis using WORLD
        
        Args:
            audio_data: Audio data, numpy array, float64 type, range [-1, 1]
            fs: Sample rate (Hz)
            frame_period: Frame period (ms)
            apply_highpass: Whether to apply high-pass filter (default: True)
            highpass_cutoff: High-pass filter cutoff frequency in Hz (default: 50.0)
        
        Returns:
            tuple: (y0, y1, y2) Reconstructed audio data for three channels
                   All channels have same length as input after filtering
    
        Raises:
            WorldSeparateError: Raised when processing fails
        """
        # Input validation
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        if audio_data.ndim != 1:
            if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                audio_data = audio_data.flatten()
            else:
                raise WorldSeparateError("Audio data must be a 1D array")
        
        # Convert to float64
        audio_data = audio_data.astype(np.float64)
        
        # Check range
        if np.abs(audio_data).max() > 1.0:
            warnings.warn("Audio data exceeds [-1, 1] range, will normalize")
            audio_data = audio_data / np.abs(audio_data).max()
        
        x_length = len(audio_data)
        if x_length == 0:
            raise WorldSeparateError("Audio data is empty")
        
        # Create C array
        x_array = (c_double * x_length)(*audio_data)
        
        # 输出参数
        y0_out = POINTER(c_double)()
        y0_length_out = c_int()
        y1_out = POINTER(c_double)()
        y1_length_out = c_int()
        y2_out = POINTER(c_double)()
        y2_length_out = c_int()
        
        try:
            # Call C function
            self._lib.SeparateAndSynthesize_C(
                x_array,
                x_length,
                fs,
                frame_period,
                ctypes.byref(y0_out),
                ctypes.byref(y0_length_out),
                ctypes.byref(y1_out),
                ctypes.byref(y1_length_out),
                ctypes.byref(y2_out),
                ctypes.byref(y2_length_out)
            )
            
            # Convert to numpy arrays
            y0_length = y0_length_out.value
            y1_length = y1_length_out.value
            y2_length = y2_length_out.value
            
            if y0_length <= 0 or y1_length <= 0 or y2_length <= 0:
                raise WorldSeparateError("Synthesis failed, output length is 0")
            
            y0_array = np.ctypeslib.as_array(y0_out, shape=(y0_length,))
            y1_array = np.ctypeslib.as_array(y1_out, shape=(y1_length,))
            y2_array = np.ctypeslib.as_array(y2_out, shape=(y2_length,))
            
            result0 = y0_array.copy()
            result1 = y1_array.copy()
            result2 = y2_array.copy()
            
            # Free C-allocated memory
            self._lib.free_audio_data(y0_out)
            self._lib.free_audio_data(y1_out)
            self._lib.free_audio_data(y2_out)
            
            # Apply high-pass filter to remove low-frequency components (< cutoff Hz)
            if apply_highpass and highpass_cutoff > 0:
                original_length = len(result0)
                result0 = apply_highpass_filter(result0, fs, highpass_cutoff)
                result1 = apply_highpass_filter(result1, fs, highpass_cutoff)
                
                # Ensure length alignment (filtfilt should preserve length, but double-check)
                result0 = result0[:original_length]
                result1 = result1[:original_length]
            
            return result0, result1, result2
            
        except Exception as e:
            # Ensure memory is freed
            if y0_out:
                try:
                    self._lib.free_audio_data(y0_out)
                except:
                    pass
            if y1_out:
                try:
                    self._lib.free_audio_data(y1_out)
                except:
                    pass
            if y2_out:
                try:
                    self._lib.free_audio_data(y2_out)
                except:
                    pass
            raise WorldSeparateError(f"WORLD analysis and synthesis failed: {e}")
    
    def process_file(self, 
                     input_path: str, 
                     output_path: str, 
                     fs: int = None, 
                     frame_period: float = 5.0,
                     apply_highpass: bool = False,
                     highpass_cutoff: float = 50.0) -> bool:
        """
        Process audio file
        
        Args:
            input_path: Input file path
            output_path: Output file path
            fs: Sample rate, if None then get from input file
            frame_period: Frame period (ms)
            apply_highpass: Whether to apply high-pass filter (default: True)
            highpass_cutoff: High-pass filter cutoff frequency in Hz (default: 50.0)
        
        Returns:
            Returns True on success, False otherwise
        """
        try:
            import scipy.io.wavfile as wavfile
        except ImportError:
            raise WorldSeparateError("scipy is required: pip install scipy")
        
        try:
            # Read input file
            fs_input, audio_int = wavfile.read(input_path)
            if fs is None:
                fs = fs_input
            
            # Convert to float64 format
            if audio_int.dtype == np.int16:
                audio_float = audio_int.astype(np.float64) / 32768.0
            elif audio_int.dtype == np.int32:
                audio_float = audio_int.astype(np.float64) / 2147483648.0
            elif audio_int.dtype == np.float32:
                audio_float = audio_int.astype(np.float64)
            else:
                audio_float = audio_int.astype(np.float64)
            
            # If stereo, take first channel
            if len(audio_float.shape) > 1:
                audio_float = audio_float[:, 0]
            
            # Process audio
            result0, result1, result2 = self.analyze_synthesize(audio_float, fs, frame_period, apply_highpass, highpass_cutoff)
            
            # By default, save the first channel (result0)
            # Convert back to integer format and save
            result_int = np.clip(result0 * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(output_path, fs, result_int)
            
            return True
            
        except Exception as e:
            print(f"File processing failed: {e}")
            return False

# Global instance
_world_separate = None

def get_world_separate(lib_path: str = None) -> WorldSeparate:
    """Get WorldSeparate instance (singleton pattern)"""
    global _world_separate
    if _world_separate is None:
        _world_separate = WorldSeparate(lib_path)
    return _world_separate

def analyze_synthesize(audio_data: np.ndarray, 
                      fs: int = 48000, 
                      frame_period: float = 5.0,
                      apply_highpass: bool = False,
                      highpass_cutoff: float = 50.0) -> tuple:
    """
    Convenience function: Perform voice analysis and synthesis using WORLD
    
    Args:
        audio_data: Audio data, numpy array
        fs: Sample rate
        frame_period: Frame period
        apply_highpass: Whether to apply high-pass filter (default: True)
        highpass_cutoff: High-pass filter cutoff frequency in Hz (default: 50.0)
    
    Returns:
        tuple: (y0, y1, y2) Reconstructed audio data for three channels
    """
    return get_world_separate().analyze_synthesize(audio_data, fs, frame_period, apply_highpass, highpass_cutoff)

def process_file(input_path: str, 
                output_path: str, 
                fs: int = None, 
                frame_period: float = 5.0,
                apply_highpass: bool = False,
                highpass_cutoff: float = 50.0) -> bool:
    """
    Convenience function: Process audio file
    
    Args:
        input_path: Input file path
        output_path: Output file path
        fs: Sample rate
        frame_period: Frame period
        apply_highpass: Whether to apply high-pass filter (default: True)
        highpass_cutoff: High-pass filter cutoff frequency in Hz (default: 50.0)
    
    Returns:
        Returns True on success
    """
    return get_world_separate().process_file(input_path, output_path, fs, frame_period, apply_highpass, highpass_cutoff)

# Version information
__version__ = "1.0.0"
__author__ = "xinxzhen"
__email__ = "xinxzhen@cisco.com"