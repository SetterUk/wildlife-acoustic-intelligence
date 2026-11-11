#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimalCLAP Traits inference script.

Evaluates a trained traits classifier on the zero-shot test split.

Traits data (species_traits.csv) is available at:
  https://huggingface.co/datasets/risashinoda/animalclap-dataset

Usage:
    python inference_traits.py \
        --ckpt        /path/to/traits_checkpoint.pth \
        --test_csv    /path/to/test.csv \
        --traits_csv  /path/to/species_traits.csv \
        --data_dir    /path/to/hf_dataset_root \
        --target_col  diet_type \
        --task_type   multiclass

For binary tasks:
    python inference_traits.py \
        --ckpt           /path/to/ckpt.pth \
        --test_csv       /path/to/test.csv \
        --traits_csv     /path/to/species_traits.csv \
        --data_dir       /path/to/hf_dataset_root \
        --target_col     migratory \
        --task_type      binary \
        --positive_value True
"""

import argparse
import random
from pathlib import Path
from typing import Optional, List

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
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--test_csv",       required=True)
    p.add_argument("--traits_csv",     required=True,
                   help="species_traits.csv from HF dataset")
    p.add_argument("--data_dir",       required=True,
                   help="HF dataset root; file_name in CSV is relative to this")
    p.add_argument("--target_col",     required=True)
    p.add_argument("--task_type",      required=True, choices=["multiclass", "binary"])
    p.add_argument("--positive_value", default="True")
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--num_workers",    type=int, default=4)
    return p.parse_args()


# ===== Dataset =====
class TraitsTestDataset(Dataset):
    SAMPLE_RATE = 48000
    CLIP_LEN    = 10.0

    def __init__(self, csv_path: str, traits_csv: str, data_dir: str,
                 target_col: str, task_type: str, positive_value: str,
                 class_to_idx: Optional[dict] = None):
        df = pd.read_csv(csv_path, dtype=str)

        # Merge traits info
        df_traits = pd.read_csv(traits_csv, dtype=str)
        df_traits["scientific_name"] = df_traits["scientific_name"].fillna("").str.strip()
        df = pd.merge(df, df_traits, on="scientific_name", how="left")

        for c in ["scientific_name", "common_name", "file_name", target_col]:
            df[c] = df.get(c, pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        df = df[df[target_col] != ""].reset_index(drop=True)

        self.df            = df
        self.data_dir      = Path(data_dir)
        self.n_samples     = int(self.SAMPLE_RATE * self.CLIP_LEN)
        self.target_col    = target_col
        self.task_type     = task_type
        self.positive_value = positive_value

        if task_type == "multiclass":
            if class_to_idx is None:
                classes = sorted(df[target_col].unique().tolist())
                self.classes      = classes
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.classes      = list(class_to_idx.keys())
                self.class_to_idx = class_to_idx
        else:
            self.classes      = None
            self.class_to_idx = None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = str(self.data_dir / row["file_name"]) if row["file_name"] else ""
        audio = self._load_audio(path) if path else torch.zeros(self.n_samples, dtype=torch.float32)
        y_raw = row[self.target_col]

        if self.task_type == "multiclass":
            y = self.class_to_idx.get(y_raw, -1)
            target = torch.tensor(int(y), dtype=torch.long)
        else:
            true_set = {self.positive_value, "True", "true", "1", "yes"}
            target   = torch.tensor(1.0 if y_raw in true_set else 0.0, dtype=torch.float32)

        return audio, target, y_raw

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


class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder: HFCLAPContrastive, num_classes: int, proj_out_dim: int = 512):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(proj_out_dim, num_classes)

    def forward(self, audio, sample_rate=48000):
        feat = self.encoder.encode_audio(audio, sample_rate=sample_rate)
        emb  = self.encoder.audio_head(feat)
        emb  = nn.functional.normalize(emb, dim=-1)
        return self.classifier(emb)


# ===== Evaluation =====
@torch.no_grad()
def evaluate(model: EncoderWithClassifier, loader: DataLoader,
             device: torch.device, task_type: str):
    model.eval()
    total = correct = tp = fp = fn = 0

    for audio, target, _ in tqdm(loader, desc="Evaluating"):
        audio = audio.to(device)

        if task_type == "multiclass":
            target = target.to(device)
            mask   = target >= 0
            if mask.sum() == 0: continue
            logits = model(audio[mask], sample_rate=TraitsTestDataset.SAMPLE_RATE)
            pred   = logits.argmax(dim=1)
            correct += (pred == target[mask]).sum().item()
            total   += mask.sum().item()
        else:
            target = target.to(device).view(-1, 1)
            logits = model(audio, sample_rate=TraitsTestDataset.SAMPLE_RATE)
            preds  = (torch.sigmoid(logits) > 0.5).float()
            tp += (preds * target).sum().item()
            fp += (preds * (1 - target)).sum().item()
            fn += ((1 - preds) * target).sum().item()
            total += target.numel()

    if task_type == "multiclass":
        return {"acc": correct / max(1, total), "n": total}
    precision = tp / max(1.0, tp + fp)
    recall    = tp / max(1.0, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall, "n": total}


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
    dataset = TraitsTestDataset(
        args.test_csv, args.traits_csv, args.data_dir,
        target_col=args.target_col, task_type=args.task_type,
        positive_value=args.positive_value,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    num_classes = len(dataset.classes) if args.task_type == "multiclass" else 1
    print(f"Test samples: {len(dataset)}")
    if dataset.classes:
        print(f"Classes ({len(dataset.classes)}): {dataset.classes[:10]}{'...' if len(dataset.classes) > 10 else ''}")

    # Model
    encoder = HFCLAPContrastive("laion/clap-htsat-unfused")
    proj_out_dim = getattr(getattr(encoder.backbone, "config", object()), "projection_dim", 512)
    model   = EncoderWithClassifier(encoder, num_classes=num_classes,
                                    proj_out_dim=proj_out_dim).to(device)

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate
    results = evaluate(model, loader, device, args.task_type)
    print("\n===== Results =====")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
