"""
AnimalCLAP Zero-Shot Wildlife Species Predictor.

Executes official taxonomy-aware contrastive inference, computing cosine similarity
between audio and biological text embeddings with GPU memory tracking and Top-5 ranking.
"""

import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

from .model_loader import AnimalCLAPManager, HFCLAPContrastive
from .audio_processor import AudioProcessor
from .taxonomy import DEFAULT_CANDIDATE_SPECIES, format_taxonomy_prompt


class WildlifePredictor:
    """Orchestrates zero-shot acoustic recognition using the official AnimalCLAP model."""

    def __init__(self, model_manager: Optional[AnimalCLAPManager] = None):
        self.manager = model_manager or AnimalCLAPManager.get_instance()
        self._cached_candidates_key: Optional[str] = None
        self._cached_text_proj: Optional[torch.Tensor] = None
        self._cached_candidate_metadata: Optional[List[Dict[str, str]]] = None

    def _ensure_text_embeddings(
        self,
        model: HFCLAPContrastive,
        device: torch.device,
        candidates: List[Dict[str, str]],
        prompt_style: str = "tax_com"
    ) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
        """Encodes and caches candidate species text embeddings."""
        cache_key = f"{prompt_style}_{len(candidates)}_" + "_".join([c.get("common_name", "") for c in candidates[:5]])
        if self._cached_candidates_key == cache_key and self._cached_text_proj is not None:
            if self._cached_text_proj.device == device:
                return self._cached_text_proj, self._cached_candidate_metadata

        # Format prompts according to AnimalCLAP taxonomy style
        prompts = [format_taxonomy_prompt(c, style=prompt_style) for c in candidates]

        with torch.no_grad():
            t_feat = model.encode_text(prompts)
            t_proj = model.text_head(t_feat)
            t_proj = nn.functional.normalize(t_proj, dim=-1)

        self._cached_candidates_key = cache_key
        self._cached_text_proj = t_proj
        self._cached_candidate_metadata = candidates

        return self._cached_text_proj, candidates

    def predict(
        self,
        file_source: Union[str, bytes],
        candidates: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        prompt_style: str = "tax_com",
        force_device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes zero-shot classification on an audio input.
        
        Returns:
            Dictionary containing Top-K ranked species, raw similarity scores,
            timing, device, and memory metadata.
        """
        if candidates is None:
            candidates = DEFAULT_CANDIDATE_SPECIES

        # 1. Preprocess audio
        prep_t0 = time.time()
        audio_chunks, audio_meta = AudioProcessor.load_and_preprocess(file_source)
        prep_time = time.time() - prep_t0

        # Check silence
        if audio_meta.get("is_silent", False):
            return {
                "success": False,
                "error": "Silence detected: The uploaded recording contains no discernible audio signal (RMS amplitude near 0.0).",
                "audio_metadata": audio_meta,
                "preprocessing_time_sec": round(prep_time, 4)
            }

        # 2. Load model
        model, device, load_info = self.manager.load_model(force_device=force_device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            vram_before = torch.cuda.memory_allocated(device) / (1024**2)
        else:
            vram_before = 0.0

        # 3. Encode text prompts (using cache if available)
        text_proj, candidate_meta = self._ensure_text_embeddings(
            model=model,
            device=device,
            candidates=candidates,
            prompt_style=prompt_style
        )

        # 4. Run audio inference
        inf_t0 = time.time()
        audio_chunks = audio_chunks.to(device)

        with torch.no_grad():
            a_feat = model.encode_audio(audio_chunks, sample_rate=48000)
            a_proj = model.audio_head(a_feat)
            a_proj = nn.functional.normalize(a_proj, dim=-1)

            # Cosine similarity matrix: [num_chunks, num_candidates]
            similarity_matrix = a_proj @ text_proj.t()

            # If multi-chunk long recording, aggregate across chunks
            if similarity_matrix.shape[0] > 1:
                # Average similarity across all 10s windows
                agg_scores = similarity_matrix.mean(dim=0)
            else:
                agg_scores = similarity_matrix[0]

            # Top-K ranking
            k = min(top_k, len(candidates))
            top_scores, top_indices = torch.topk(agg_scores, k=k)

        inf_time = time.time() - inf_t0

        if device.type == "cuda":
            vram_after = torch.cuda.memory_allocated(device) / (1024**2)
            vram_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            vram_after = 0.0
            vram_peak = 0.0

        # 5. Build structured predictions
        predictions = []
        for rank, (score, idx) in enumerate(zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()), start=1):
            c = candidate_meta[idx]
            predictions.append({
                "rank": rank,
                "common_name": c.get("common_name", "Unknown"),
                "scientific_name": c.get("scientific_name", "Unknown"),
                "category": c.get("category", "General"),
                "family": c.get("family", ""),
                "order": c.get("order", ""),
                "class": c.get("class", ""),
                "similarity_score": round(float(score), 4),
                "habitat": c.get("habitat", ""),
                "vocalization_type": c.get("vocalization", "")
            })

        return {
            "success": True,
            "top_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
            "num_candidates_evaluated": len(candidates),
            "prompt_style_used": prompt_style,
            "audio_metadata": audio_meta,
            "hardware_metrics": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "inference_time_sec": round(inf_time, 4),
                "preprocessing_time_sec": round(prep_time, 4),
                "total_processing_time_sec": round(prep_time + inf_time, 4),
                "initial_vram_mb": round(vram_before, 2),
                "current_vram_mb": round(vram_after, 2),
                "peak_vram_mb": round(vram_peak, 2),
            },
            "scientific_note": "Similarity scores represent cosine similarity in AnimalCLAP joint multimodal embedding space [-1.0, 1.0]. They indicate acoustic-taxonomic alignment and are NOT calibrated Bayesian probabilities."
        }
