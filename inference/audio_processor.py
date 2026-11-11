"""
Audio Processing Pipeline for AnimalCLAP Wildlife Acoustic Intelligence.

Implements strict validation, resampling to 48,000 Hz, stereo-to-mono downmixing,
silence detection, and sliding window chunking for long recordings.
"""

import io
import os
import math
from typing import Tuple, List, Dict, Any, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import resample


TARGET_SAMPLE_RATE = 48000
CLIP_DURATION_SEC = 10.0
CLIP_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_DURATION_SEC)  # 480,000 samples
SILENCE_THRESHOLD_RMS = 1e-4  # ~ -80 dBFS


class AudioValidationError(ValueError):
    """Raised when an audio recording fails format, corruption, or content checks."""
    pass


class AudioProcessor:
    """Preprocesses wildlife audio recordings for AnimalCLAP inference."""

    @staticmethod
    def inspect_audio(file_source: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
        """
        Inspect an audio file without loading entire data into memory if possible.
        Returns metadata: sample_rate, duration_seconds, channels, format, is_silent, rms.
        """
        try:
            if isinstance(file_source, bytes):
                file_source = io.BytesIO(file_source)

            # Using soundfile for robust header parsing
            info = sf.info(file_source)
            duration = float(info.duration)
            sr = int(info.samplerate)
            channels = int(info.channels)
            fmt = str(info.format)
            subtype = str(info.subtype)
            
            return {
                "valid": True,
                "duration_seconds": duration,
                "sample_rate": sr,
                "channels": channels,
                "format": fmt,
                "subtype": subtype,
                "frames": info.frames
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to parse audio header: {str(e)}"
            }

    @classmethod
    def load_and_preprocess(
        cls,
        file_source: Union[str, bytes, io.BytesIO],
        chunk_long_audio: bool = True,
        chunk_step_sec: float = 5.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load audio, validate, downmix to mono, resample to 48kHz, and chunk if necessary.
        
        Returns:
            chunks_tensor: torch.Tensor of shape [N, 480000] (float32)
            metadata: Dict with original & processed properties
        """
        if isinstance(file_source, bytes):
            file_source = io.BytesIO(file_source)

        try:
            # Read via soundfile
            data, orig_sr = sf.read(file_source, dtype="float32")
        except Exception as sf_err:
            try:
                # Fallback to torchaudio if soundfile fails
                if isinstance(file_source, io.BytesIO):
                    file_source.seek(0)
                wf, orig_sr = torchaudio.load(file_source)
                data = wf.numpy().T
            except Exception as ta_err:
                raise AudioValidationError(
                    f"Corrupted or unsupported audio file. SoundFile error: {sf_err}; Torchaudio error: {ta_err}"
                )

        if data is None or len(data) == 0:
            raise AudioValidationError("Audio file is completely empty (zero samples).")

        # Downmix multi-channel to mono
        if data.ndim == 2:
            mono_data = data.mean(axis=1)
        elif data.ndim == 1:
            mono_data = data
        else:
            raise AudioValidationError(f"Unexpected audio dimensions: {data.ndim}")

        # Basic statistics
        raw_samples = len(mono_data)
        duration_sec = raw_samples / float(orig_sr)
        rms = float(np.sqrt(np.mean(mono_data**2)))
        peak = float(np.max(np.abs(mono_data)))
        is_silent = rms < SILENCE_THRESHOLD_RMS

        # Convert to torch tensor
        wf = torch.from_numpy(mono_data).to(torch.float32)

        # Resample to 48,000 Hz if necessary
        if orig_sr != TARGET_SAMPLE_RATE:
            try:
                wf = resample(wf, orig_sr, TARGET_SAMPLE_RATE)
            except Exception as resample_err:
                raise AudioValidationError(
                    f"Resampling from {orig_sr} Hz to {TARGET_SAMPLE_RATE} Hz failed: {resample_err}"
                )

        total_samples_48k = wf.numel()

        # Build chunks of 480,000 samples (10.0 seconds at 48kHz)
        chunks = []
        chunk_metadata = []

        if total_samples_48k <= CLIP_NUM_SAMPLES or not chunk_long_audio:
            # Single clip with padding if needed
            if total_samples_48k < CLIP_NUM_SAMPLES:
                padded = torch.nn.functional.pad(wf, (0, CLIP_NUM_SAMPLES - total_samples_48k))
            else:
                padded = wf[:CLIP_NUM_SAMPLES]
            chunks.append(padded)
            chunk_metadata.append({"start_sec": 0.0, "end_sec": min(duration_sec, CLIP_DURATION_SEC)})
        else:
            # Long audio: sliding window with overlap
            step_samples = int(chunk_step_sec * TARGET_SAMPLE_RATE)
            start = 0
            while start < total_samples_48k:
                end = start + CLIP_NUM_SAMPLES
                chunk = wf[start:end]
                if chunk.numel() < CLIP_NUM_SAMPLES:
                    chunk = torch.nn.functional.pad(chunk, (0, CLIP_NUM_SAMPLES - chunk.numel()))
                chunks.append(chunk)
                start_sec = round(start / TARGET_SAMPLE_RATE, 2)
                end_sec = round(min(total_samples_48k, end) / TARGET_SAMPLE_RATE, 2)
                chunk_metadata.append({"start_sec": start_sec, "end_sec": end_sec})
                start += step_samples

        chunks_tensor = torch.stack(chunks, dim=0)

        metadata = {
            "original_sample_rate": orig_sr,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "original_duration_seconds": round(duration_sec, 2),
            "original_channels": data.shape[1] if data.ndim == 2 else 1,
            "rms_energy": rms,
            "peak_amplitude": peak,
            "is_silent": is_silent,
            "num_chunks": len(chunks),
            "chunk_windows": chunk_metadata
        }

        return chunks_tensor, metadata

