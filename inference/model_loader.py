"""
Official AnimalCLAP Model Loader with CUDA RTX 4050 Acceleration and Graceful CPU Fallback.
"""

import os
import sys
import time
import logging
from typing import Optional, Tuple, Dict, Any

# Ensure offline huggingface loading once cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from transformers import ClapProcessor, ClapModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ProjectionMLP(nn.Module):
    """Projection head used in the official AnimalCLAP ICASSP 2026 architecture."""
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _extract_tensor(x):
    """Extract torch.Tensor from transformers 5.x ModelOutput or tuple if needed."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
        return x.last_hidden_state
    if hasattr(x, "audio_features") and x.audio_features is not None:
        return x.audio_features
    if hasattr(x, "text_features") and x.text_features is not None:
        return x.text_features
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


class HFCLAPContrastive(nn.Module):
    """
    Exact AnimalCLAP model definition from official dahlian00/AnimalCLAP repository.
    Combines LAION CLAP (HTS-AT audio encoder + RoBERTa text encoder) with
    taxonomy-aware contrastive projection MLP heads.
    """
    def __init__(
        self,
        model_id: str = "laion/clap-htsat-unfused",
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 512
    ):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        self.backbone = ClapModel.from_pretrained(model_id, use_safetensors=True, local_files_only=True)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

        feat_dim = getattr(getattr(self.backbone, "config", object()), "projection_dim", 512)
        self.audio_head = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.text_head = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)

        self.processor.feature_extractor.do_resample = False
        self.processor.feature_extractor.return_attention_mask = False

    def _dev(self):
        return next(self.parameters()).device

    def encode_audio(self, audio: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        """Encodes audio waveform tensor of shape [batch, samples] at 48kHz."""
        audio_list = [a.cpu().numpy() for a in audio]
        try:
            inputs = self.processor(
                audio=audio_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
        except (TypeError, ValueError):
            inputs = self.processor(
                audios=audio_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        res = self.backbone.get_audio_features(**inputs)
        return _extract_tensor(res)

    def encode_text(self, texts: list) -> torch.Tensor:
        """Encodes candidate biological text descriptions."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        res = self.backbone.get_text_features(**inputs)
        return _extract_tensor(res)


class AnimalCLAPManager:
    """Manages model lifecycle, hardware selection, and GPU memory monitoring."""

    _instance: Optional["AnimalCLAPManager"] = None
    _model: Optional[HFCLAPContrastive] = None
    _device: Optional[torch.device] = None
    _load_info: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "AnimalCLAPManager":
        if cls._instance is None:
            cls._instance = AnimalCLAPManager()
        return cls._instance

    @staticmethod
    def get_gpu_status() -> Dict[str, Any]:
        """Inspects current GPU environment and memory statistics."""
        cuda_avail = torch.cuda.is_available()
        status = {
            "cuda_available": cuda_avail,
            "device_count": torch.cuda.device_count() if cuda_avail else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU (No GPU active)",
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0,
            "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if cuda_avail else 0.0,
            "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2) if cuda_avail else 0.0,
            "max_vram_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 2) if cuda_avail else 0.0,
        }
        return status

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        force_device: Optional[str] = None
    ) -> Tuple[HFCLAPContrastive, torch.device, Dict[str, Any]]:
        """
        Loads the official AnimalCLAP model into memory.
        Uses CUDA if available, with automatic CPU fallback.
        """
        if self._model is not None and (force_device is None or str(self._device) == force_device):
            return self._model, self._device, self._load_info

        # Determine default checkpoint path
        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(base_dir, "checkpoints", "animalclap_epoch020.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"AnimalCLAP checkpoint not found at: {checkpoint_path}")

        # Determine target device
        if force_device:
            target_device = torch.device(force_device)
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading AnimalCLAP onto target device: {target_device}...")
        t0 = time.time()

        try:
            model = HFCLAPContrastive("laion/clap-htsat-unfused").to(target_device)
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            load_result = model.load_state_dict(sd, strict=False)
            model.eval()
            actual_device = target_device
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if target_device.type == "cuda":
                logger.warning(f"CUDA memory or initialization error: {e}. Falling back gracefully to CPU.")
                torch.cuda.empty_cache()
                actual_device = torch.device("cpu")
                model = HFCLAPContrastive("laion/clap-htsat-unfused").to(actual_device)
                sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                load_result = model.load_state_dict(sd, strict=False)
                model.eval()
            else:
                raise e

        load_duration = time.time() - t0

        self._model = model
        self._device = actual_device
        self._load_info = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_size_mb": round(os.path.getsize(checkpoint_path) / (1024**2), 2),
            "device": str(actual_device),
            "device_name": torch.cuda.get_device_name(0) if actual_device.type == "cuda" else "CPU",
            "load_time_seconds": round(load_duration, 2),
            "missing_keys_count": len(load_result.missing_keys),
            "unexpected_keys_count": len(load_result.unexpected_keys),
            "initial_vram_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2) if actual_device.type == "cuda" else 0.0
        }

        logger.info(
            f"AnimalCLAP successfully loaded in {load_duration:.2f}s on {actual_device} "
            f"(missing: {len(load_result.missing_keys)}, unexpected: {len(load_result.unexpected_keys)})"
        )

        return self._model, self._device, self._load_info

