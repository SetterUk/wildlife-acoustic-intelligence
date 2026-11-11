"""
Export utilities for session history and prediction results.
"""

import json
import pandas as pd
from typing import List, Dict, Any


def export_to_json(records: List[Dict[str, Any]]) -> str:
    """Formats analysis history as formatted JSON string."""
    return json.dumps(records, indent=2, ensure_ascii=False)


def export_to_csv(records: List[Dict[str, Any]]) -> str:
    """Flattens session analysis history into CSV format."""
    flat_rows = []
    for rec in records:
        top1 = rec.get("top_prediction", {}) or {}
        hw = rec.get("hardware_metrics", {}) or {}
        audio = rec.get("audio_metadata", {}) or {}

        row = {
            "timestamp": rec.get("timestamp", ""),
            "filename": rec.get("filename", ""),
            "top1_common_name": top1.get("common_name", ""),
            "top1_scientific_name": top1.get("scientific_name", ""),
            "top1_category": top1.get("category", ""),
            "top1_similarity_score": top1.get("similarity_score", ""),
            "audio_duration_sec": audio.get("original_duration_seconds", ""),
            "sample_rate_hz": audio.get("original_sample_rate", ""),
            "inference_time_sec": hw.get("inference_time_sec", ""),
            "device": hw.get("device", ""),
            "device_name": hw.get("device_name", ""),
            "peak_vram_mb": hw.get("peak_vram_mb", ""),
            "num_candidates": rec.get("num_candidates_evaluated", "")
        }

        # Include Top-5 breakdown
        predictions = rec.get("predictions", [])
        for i in range(5):
            if i < len(predictions):
                p = predictions[i]
                row[f"rank_{i+1}_species"] = f"{p.get('common_name')} ({p.get('scientific_name')})"
                row[f"rank_{i+1}_score"] = p.get("similarity_score")
            else:
                row[f"rank_{i+1}_species"] = ""
                row[f"rank_{i+1}_score"] = ""

        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    return df.to_csv(index=False)

