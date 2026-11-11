#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimalCLAP Traits Classifier training script.

Trains a linear classifier on top of a frozen AnimalCLAP encoder
for binary or multi-class trait prediction (e.g. diet_type, migratory).

Traits data (species_traits.csv) is available at:
  https://huggingface.co/datasets/risashinoda/animalclap-dataset

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 train_traits.py \
        --train_csv   /path/to/train.csv \
        --test_csv    /path/to/test.csv \
        --traits_csv  /path/to/species_traits.csv \
        --audio_dir   /path/to/audio \
        --encoder_ckpt /path/to/animalclap.pth \
        --target_col  diet_type \
        --task_type   multiclass \
        --save_dir    ./models/traits \
        --run_name    diet_type_run

For binary tasks, also pass:
    --positive_value True
"""

import os
import argparse
import random
import json
import socket
import warnings
from pathlib import Path
from datetime import timedelta
from time import perf_counter as now
from typing import List, Optional

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from tqdm import tqdm

import wandb
import torchaudio
from torchaudio.functional import resample
from transformers import ClapProcessor, ClapModel


# ===== Args =====
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",    required=True)
    p.add_argument("--test_csv",     required=True, help="Zero-shot test split CSV")
    p.add_argument("--traits_csv",   required=True,
                   help="species_traits.csv from HF dataset (scientific_name + trait columns)")
    p.add_argument("--audio_dir",    required=True)
    p.add_argument("--encoder_ckpt", required=True, help="Pre-trained AnimalCLAP checkpoint")
    p.add_argument("--target_col",   required=True, help="Trait column name to predict")
    p.add_argument("--task_type",    required=True, choices=["multiclass", "binary"])
    p.add_argument("--positive_value", default="True",
                   help="Positive label string for binary tasks")
    p.add_argument("--save_dir",     default="./models/traits")
    p.add_argument("--run_name",     default=None)
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--per_species",  type=int, default=30)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    return p.parse_args()


# ===== DDP utils =====
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30),
                                init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        print(f"[rank {dist.get_rank()}] setup done, host={socket.gethostname()}", flush=True)

def is_dist():  return dist.is_initialized()
def is_rank0(): return (not is_dist()) or (dist.get_rank() == 0)


# ===== Dataset =====
def resolve_audio_path(row, audio_dir: Path) -> str:
    file_name = str(row.get("file_name", "") or "").strip()
    if file_name:
        return str(audio_dir / file_name)
    source = str(row.get("source", "")).strip()
    rec_id = str(row.get("recording_id", "")).strip()
    matches = list(audio_dir.glob(f"{source}_{rec_id}_{rec_id}_0.*"))
    if matches: return str(matches[0])
    matches = list(audio_dir.rglob(f"{source}_{rec_id}_*"))
    if matches: return str(matches[0])
    return ""


class TraitsDataset(Dataset):
    SAMPLE_RATE = 48000
    CLIP_LEN    = 10.0

    def __init__(self, csv_path: str, traits_csv: str, audio_dir: str,
                 target_col: str, task_type: str, positive_value: str,
                 split: str = "train", per_species: int = 30,
                 class_to_idx: Optional[dict] = None):
        df = pd.read_csv(csv_path, dtype=str)
        if "split" in df.columns:
            df = df[df["split"] == split].copy()

        # Merge traits info from species_traits.csv
        df_traits = pd.read_csv(traits_csv, dtype=str)
        df_traits["scientific_name"] = df_traits["scientific_name"].fillna("").str.strip()
        df = pd.merge(df, df_traits, on="scientific_name", how="left")

        for c in ["scientific_name", "common_name", target_col]:
            df[c] = df.get(c, pd.Series(dtype=str)).fillna("").astype(str).str.strip()

        # Drop rows with empty target
        n_before = len(df)
        df = df[df[target_col] != ""].copy()
        if is_rank0() and n_before - len(df) > 0:
            print(f"[data] dropped {n_before - len(df)} rows with empty {target_col}", flush=True)

        # Build class vocab (multiclass)
        if task_type == "multiclass":
            if class_to_idx is None:
                classes = sorted(df[target_col].unique().tolist())
                self.classes     = classes
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.classes      = list(class_to_idx.keys())
                self.class_to_idx = class_to_idx
        else:
            self.classes      = None
            self.class_to_idx = None

        # Per-species sampling for train
        if split == "train" and "scientific_name" in df.columns:
            df = (
                df.groupby("scientific_name", group_keys=False)
                  .apply(lambda g: g.sample(n=per_species, replace=True, random_state=None))
                  .reset_index(drop=True)
            )
        df = df.sample(frac=1.0, random_state=None).reset_index(drop=True)

        self.df            = df
        self.audio_dir     = Path(audio_dir)
        self.n_samples     = int(self.SAMPLE_RATE * self.CLIP_LEN)
        self.target_col    = target_col
        self.task_type     = task_type
        self.positive_value = positive_value

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = resolve_audio_path(row, self.audio_dir)
        audio = self._load_audio(path) if path else torch.zeros(self.n_samples, dtype=torch.float32)
        y_raw = row[self.target_col]

        if self.task_type == "multiclass":
            y = self.class_to_idx.get(y_raw, -1)
            target = torch.tensor(int(y), dtype=torch.long)
        else:
            true_set = {self.positive_value, "True", "true", "1", "yes"}
            target = torch.tensor(1.0 if y_raw in true_set else 0.0, dtype=torch.float32)

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
        _dist  = dist.is_initialized
        _rank  = lambda: dist.get_rank() if _dist() else 0
        _rank0 = lambda: not _dist() or _rank() == 0

        if _rank0():
            self.processor = ClapProcessor.from_pretrained(model_id)
            self.backbone  = ClapModel.from_pretrained(model_id, use_safetensors=True)
        if _dist(): dist.barrier()
        if not _rank0():
            self.processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
            self.backbone  = ClapModel.from_pretrained(model_id, local_files_only=True, use_safetensors=True)
        if _dist(): dist.barrier()

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
    def __init__(self, encoder: HFCLAPContrastive, num_classes: int,
                 proj_out_dim: int = 512, freeze_encoder: bool = True):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(proj_out_dim, num_classes)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode_audio(self, audio, sample_rate=48000):
        feat = self.encoder.encode_audio(audio, sample_rate=sample_rate)
        emb  = self.encoder.audio_head(feat)
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, audio, sample_rate=48000):
        return self.classifier(self.encode_audio(audio, sample_rate))


# ===== Checkpoint loader =====
def load_encoder_ckpt(encoder: HFCLAPContrastive, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if is_rank0():
        print(f"[ckpt] loaded {ckpt_path}")
        if missing:     print(f"  missing ({len(missing)}): {missing[:3]}")
        if unexpected:  print(f"  unexpected ({len(unexpected)}): {unexpected[:3]}")


# ===== Evaluation =====
@torch.no_grad()
def evaluate(model, loader, device, task_type):
    m = model.module if isinstance(model, DDP) else model
    m.eval()
    total = tp = fp = fn = correct = 0

    for audio, target, _ in loader:
        audio = audio.to(device, non_blocking=True)
        if task_type == "multiclass":
            target = target.to(device)
            mask   = target >= 0
            if mask.sum() == 0: continue
            logits = m(audio[mask], sample_rate=TraitsDataset.SAMPLE_RATE)
            pred   = logits.argmax(dim=1)
            correct += (pred == target[mask]).sum().item()
            total   += mask.sum().item()
        else:
            target = target.to(device).view(-1, 1)
            logits = m(audio, sample_rate=TraitsDataset.SAMPLE_RATE)
            preds  = (torch.sigmoid(logits) > 0.5).float()
            tp += (preds * target).sum().item()
            fp += (preds * (1 - target)).sum().item()
            fn += ((1 - preds) * target).sum().item()
            total += target.numel()

    if is_dist():
        if task_type == "multiclass":
            t = torch.tensor([total, correct], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total, correct = t.tolist()
        else:
            v = torch.tensor([tp, fp, fn], device=device, dtype=torch.float32)
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            tp, fp, fn = v.tolist()

    m.train()
    if task_type == "multiclass":
        return {"acc": correct / max(1, total)}
    precision = tp / max(1.0, tp + fp)
    recall    = tp / max(1.0, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


# ===== Main =====
def main():
    args = parse_args()
    setup_distributed()

    run_name = args.run_name or f"traits_{args.target_col}_{args.task_type}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = save_dir / "train_log.json"

    _seed = 1337 + (dist.get_rank() if dist.is_initialized() else 0)
    random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

    DEVICE = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available() else "cpu"
    )

    if is_rank0():
        wandb.init(project="animalclap", name=run_name,
                   config={**vars(args), "run_name": run_name})

    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

    # Datasets
    train_dataset = TraitsDataset(
        args.train_csv, args.traits_csv, args.audio_dir,
        target_col=args.target_col, task_type=args.task_type,
        positive_value=args.positive_value, split="train",
        per_species=args.per_species,
    )
    test_dataset = TraitsDataset(
        args.test_csv, args.traits_csv, args.audio_dir,
        target_col=args.target_col, task_type=args.task_type,
        positive_value=args.positive_value, split="test",
        class_to_idx=train_dataset.class_to_idx,
    )

    num_classes = len(train_dataset.classes) if args.task_type == "multiclass" else 1
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if is_dist():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False, drop_last=False)
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **loader_kwargs)
        test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, sampler=test_sampler,  **loader_kwargs)
    else:
        train_sampler = test_sampler = None
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
        test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Model
    encoder = HFCLAPContrastive("laion/clap-htsat-unfused")
    if is_dist(): dist.barrier()
    load_encoder_ckpt(encoder, args.encoder_ckpt)

    proj_out_dim = getattr(getattr(encoder.backbone, "config", object()), "projection_dim", 512)
    model = EncoderWithClassifier(encoder, num_classes=num_classes,
                                  proj_out_dim=proj_out_dim,
                                  freeze_encoder=args.freeze_encoder).to(DEVICE)
    if is_dist():
        ddp_kwargs = dict(device_ids=[DEVICE.index], output_device=DEVICE.index) \
                     if DEVICE.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=False,
                    broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kwargs)
    model.train()

    criterion = nn.CrossEntropyLoss() if args.task_type == "multiclass" else nn.BCEWithLogitsLoss()
    learnable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.AdamW(learnable, lr=args.lr, weight_decay=0.01)
    scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    logs       = []

    for epoch in range(1, args.epochs + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_rank0())

        for audio, target, _ in pbar:
            audio = audio.to(DEVICE, non_blocking=True)
            target = (target.to(DEVICE) if args.task_type == "multiclass"
                      else target.to(DEVICE).view(-1, 1))
            cm = torch.amp.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
            with cm:
                logits = model(audio, sample_rate=TraitsDataset.SAMPLE_RATE)
                loss   = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            total_loss += float(loss.detach().cpu())
            if is_rank0():
                pbar.set_postfix(loss=f"{total_loss / max(1, pbar.n + 1):.4f}")

        loss_t = torch.tensor(total_loss / max(1, len(train_loader)), device=DEVICE)
        if is_dist(): dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        train_loss = float(loss_t.item())

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            if isinstance(test_sampler, DistributedSampler):
                test_sampler.set_epoch(epoch)
            metrics = evaluate(model, test_loader, DEVICE, args.task_type)
            if is_rank0():
                print(f"[Epoch {epoch}] loss={train_loss:.4f} eval={metrics}")
                wandb.log({"train/loss": train_loss, "epoch": epoch,
                           **{f"eval/{k}": v for k, v in metrics.items()}})
                ckpt  = save_dir / f"epoch{epoch:03d}.pth"
                state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(state, ckpt)
                logs.append({"epoch": epoch, "train_loss": train_loss, **metrics})
                with open(log_json_path, "w") as f: json.dump(logs, f, indent=2)
        else:
            if is_rank0():
                wandb.log({"train/loss": train_loss, "epoch": epoch})
                print(f"[Epoch {epoch}] loss={train_loss:.4f}")

    if is_rank0():
        final = save_dir / f"final_{args.target_col}_{args.task_type}.pth"
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state, final)
        print(f"Final model saved: {final}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
