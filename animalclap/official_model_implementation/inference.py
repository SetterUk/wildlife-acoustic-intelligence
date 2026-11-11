#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimalCLAP zero-shot inference script.

Evaluates Top-K species classification accuracy on the test split.
Requires the HuggingFace dataset test.csv and the downloaded audio files.

Usage:
    python inference.py \
        --ckpt /path/to/checkpoint.pth \
        --test_csv /path/to/test.csv \
        --data_dir /path/to/hf_dataset_root \
        --ks 1 5 10
"""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
from torchaudio.functional import resample
from transformers import ClapProcessor, ClapModel


# ===== Args =====
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--test_csv",   required=True, help="Path to test.csv from HF dataset")
    p.add_argument("--data_dir",   required=True,
                   help="HF dataset root directory (audio paths in test.csv are relative to this)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--ks",         type=int, nargs="+", default=[1, 5, 10],
                   help="Top-K values to evaluate")
    return p.parse_args()


# ===== Dataset =====
class TestDataset(Dataset):
    SAMPLE_RATE = 48000
    CLIP_LEN    = 10.0

    def __init__(self, csv_path: str, data_dir: str):
        df = pd.read_csv(csv_path, dtype=str)
        for c in ["scientific_name", "common_name", "file_name"]:
            df[c] = df.get(c, pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        df = df[df["file_name"] != ""].reset_index(drop=True)

        self.df       = df
        self.data_dir = Path(data_dir)
        self.n_samples = int(self.SAMPLE_RATE * self.CLIP_LEN)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = str(self.data_dir / row["file_name"])
        audio = self._load_audio(path)
        label = row["common_name"] or row["scientific_name"] or "unknown"
        return audio, label

    def _load_audio(self, path: str) -> torch.Tensor:
        try:
            si = torchaudio.info(path)
            total = si.num_frames
            if total and total > self.n_samples:
                start = random.randint(0, max(0, total - self.n_samples))
                wf, sr = torchaudio.load(path, frame_offset=start, num_frames=self.n_samples)
            else:
                wf, sr = torchaudio.load(path)
            if wf.dim() == 2: wf = wf.mean(dim=0)
        except Exception:
            try:
                wf, sr = torchaudio.load(path)
                if wf.dim() == 2: wf = wf.mean(dim=0)
            except Exception:
                try:
                    y, sr = librosa.load(path, sr=None, mono=True)
                    wf = torch.from_numpy(y)
                except Exception:
                    return torch.zeros(self.n_samples, dtype=torch.float32)

        wf = wf.to(torch.float32)
        sr = int(sr) if sr and int(sr) > 0 else self.SAMPLE_RATE
        if sr != self.SAMPLE_RATE and wf.numel() > 1:
            try: wf = resample(wf, sr, self.SAMPLE_RATE)
            except Exception: return torch.zeros(self.n_samples, dtype=torch.float32)
        if wf.numel() >= self.n_samples: return wf[:self.n_samples]
        return torch.nn.functional.pad(wf, (0, self.n_samples - wf.numel()))


# ===== Model =====
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


class HFCLAPContrastive(nn.Module):
    def __init__(self, model_id="laion/clap-htsat-unfused",
                 proj_hidden_dim=512, proj_out_dim=512):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.backbone  = ClapModel.from_pretrained(model_id, use_safetensors=True)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))
        feat_dim = getattr(getattr(self.backbone, "config", object()), "projection_dim", 512)
        self.audio_head = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.text_head  = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.processor.feature_extractor.do_resample = False
        self.processor.feature_extractor.return_attention_mask = False

    def _dev(self): return next(self.parameters()).device

    def encode_audio(self, audio: torch.Tensor, sample_rate=48000) -> torch.Tensor:
        inputs = self.processor(audios=[a.cpu().numpy() for a in audio],
                                sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        return self.backbone.get_audio_features(**inputs)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        return self.backbone.get_text_features(**inputs)


# ===== Evaluation =====
@torch.no_grad()
def evaluate_topk(model: HFCLAPContrastive, loader: DataLoader,
                  class_texts: List[str], device: torch.device, ks=(1, 5, 10)):
    model.eval()
    class_to_idx = {name: i for i, name in enumerate(class_texts)}

    text_feats = model.encode_text(class_texts)
    text_proj  = model.text_head(text_feats)
    text_proj  = nn.functional.normalize(text_proj, dim=-1)

    ks      = sorted(ks)
    correct = {k: 0 for k in ks}
    total   = 0

    for audio, labels in tqdm(loader, desc="Evaluating"):
        audio  = audio.to(device)
        label_idx = torch.tensor(
            [class_to_idx.get(l, class_to_idx.get("unknown", 0)) for l in labels],
            device=device, dtype=torch.long
        )

        a_feat = model.encode_audio(audio, sample_rate=TestDataset.SAMPLE_RATE)
        a_proj = model.audio_head(a_feat)
        a_proj = nn.functional.normalize(a_proj, dim=-1)

        logits    = a_proj @ text_proj.t()
        max_k     = ks[-1]
        _, topk_idx = torch.topk(logits, k=min(max_k, len(class_texts)), dim=1)

        for k in ks:
            hit = (topk_idx[:, :k] == label_idx.unsqueeze(1)).any(dim=1)
            correct[k] += hit.long().sum().item()
        total += label_idx.numel()

    return {k: correct[k] / max(1, total) for k in ks}


# ===== Main =====
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if "sox_io" in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend("sox_io")
    except Exception:
        pass

    # Dataset
    dataset     = TestDataset(args.test_csv, args.data_dir)
    class_texts = sorted(set(dataset.df["common_name"].tolist()))
    loader      = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(dataset)}, Classes: {len(class_texts)}")

    # Model
    model = HFCLAPContrastive("laion/clap-htsat-unfused").to(device)
    sd    = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate
    results = evaluate_topk(model, loader, class_texts, device, ks=args.ks)
    print("\n===== Results =====")
    for k, acc in sorted(results.items()):
        print(f"  Top-{k}: {acc:.4f}")


if __name__ == "__main__":
    main()
