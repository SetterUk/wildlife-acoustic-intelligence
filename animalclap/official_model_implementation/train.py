#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimalCLAP training script.

Expects the HuggingFace dataset CSV format (single CSV with all columns).
Audio for train/val must be pre-downloaded locally; set --audio_dir accordingly.

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --train_csv /path/to/train.csv \
        --details_csv /path/to/details.csv \
        --audio_dir /path/to/audio \
        --save_dir ./models/animalclap \
        --run_name exp1
"""

import os
import argparse
import random
from pathlib import Path
from datetime import timedelta
from time import perf_counter as now
import socket
import json
import warnings
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
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--audio_dir",  required=True,
                   help="Base directory where audio files are stored. "
                        "Audio path is resolved as audio_dir/file_name for test split, "
                        "or audio_dir/{source}_{recording_id}_* for train split.")
    p.add_argument("--save_dir",   default="./models")
    p.add_argument("--run_name",   default="animalclap")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--per_species",type=int, default=30)
    p.add_argument("--num_workers",type=int, default=2)
    # Text mode for ablations (env var also accepted)
    # TEXT_MODE: "" | "tax_perm_order" | "tax_bow" | "tax_struct" | "tax_struct_rand"
    # TAX_PAYLOAD: "tax" | "tax+com" | "mix"  (used with tax_perm_order)
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
    """Resolve local audio path from a CSV row."""
    file_name = str(row.get("file_name", "") or "").strip()
    if file_name:
        return str(audio_dir / file_name)
    # Train/val: no file_name — search by source + recording_id
    source = str(row.get("source", "")).strip()
    rec_id = str(row.get("recording_id", "")).strip()
    pattern = f"{source}_{rec_id}_{rec_id}_0.*"
    matches = list(audio_dir.glob(pattern))
    if matches:
        return str(matches[0])
    # Fallback: flat search
    matches = list(audio_dir.rglob(f"{source}_{rec_id}_*"))
    if matches:
        return str(matches[0])
    return ""


class AnimalDataset(Dataset):
    """
    Loads a single HF-format CSV (train.csv / test.csv).
    Columns expected: scientific_name, common_name, class, order, family, genus,
                      source, split, recording_id, file_name, download_url
    """
    SAMPLE_RATE = 48000
    CLIP_LEN    = 10.0

    def __init__(self, csv_path: str, audio_dir: str,
                 split: str = "train", per_species: int = 30):
        df = pd.read_csv(csv_path, dtype=str)

        # Filter to requested split
        if "split" in df.columns:
            df = df[df["split"] == split].copy()

        for c in ["scientific_name", "common_name", "class", "order", "family", "genus"]:
            df[c] = df.get(c, pd.Series(dtype=str)).fillna("").astype(str).str.strip()

        # Per-species sampling for train
        if split == "train" and "scientific_name" in df.columns:
            df = (
                df.groupby("scientific_name", group_keys=False)
                  .apply(lambda g: g.sample(n=per_species, replace=True, random_state=None))
                  .reset_index(drop=True)
            )
        df = df.sample(frac=1.0, random_state=None).reset_index(drop=True)

        self.df          = df
        self.audio_dir   = Path(audio_dir)
        self.split       = split
        self.n_samples   = int(self.SAMPLE_RATE * self.CLIP_LEN)

        self.ranks = ["class", "order", "family", "genus", "scientific_name"]
        self._order_perm: Optional[List[int]] = None

        # Ablation: randomize taxonomy order (TEXT_MODE=tax_perm_order)
        if os.getenv("TEXT_MODE", "").strip() == "tax_perm_order":
            seed = int(os.getenv("SHUFFLE_SEED")) if os.getenv("SHUFFLE_SEED") else 0
            rng  = random.Random(seed)
            self._order_perm = list(range(5))
            rng.shuffle(self._order_perm)
            if dist.is_initialized():
                obj = [self._order_perm]
                dist.broadcast_object_list(obj, src=0)
                self._order_perm = obj[0]

    # ---- text helpers ----
    def _nz(self, x): return (x or "").strip() or "unknown"

    def _row_tax_nodes(self, row) -> List[str]:
        nodes = [self._nz(row.get(c, "")) for c in self.ranks]
        if self._order_perm is not None:
            nodes = [nodes[i] for i in self._order_perm]
        return nodes

    def _make_tax_string(self, row) -> str:
        return " ".join(x for x in self._row_tax_nodes(row) if x)

    def _bioclio_text(self, row) -> str:
        sci = self._nz(row.get("scientific_name", ""))
        com = self._nz(row.get("common_name", "")) or sci
        cls = (row.get("class",  "") or "").strip()
        ord_= (row.get("order",  "") or "").strip()
        fam = (row.get("family", "") or "").strip()
        gen = (row.get("genus",  "") or "").strip()
        tax = " ".join(x for x in [cls, ord_, fam, gen, sci] if x)
        candidates = []
        if com:         candidates.append("{com}")
        if sci:         candidates.append("{sci}")
        if tax:         candidates.append("{tax}")
        if sci and com: candidates.append("{sci} with common name {com}")
        if tax and com: candidates.append("{tax} with common name {com}")
        if not candidates:
            candidates = ["{com}"]
        return random.choice(candidates).format(com=com, sci=sci, tax=tax)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform = None
        try:
            si = torchaudio.info(path)
            total = si.num_frames
            if total and total > self.n_samples:
                start = random.randint(0, max(0, total - self.n_samples))
                wf, sr = torchaudio.load(path, frame_offset=start, num_frames=self.n_samples)
            else:
                wf, sr = torchaudio.load(path)
            if wf.dim() == 2:
                wf = wf.mean(dim=0)
            waveform, orig_sr = wf, sr
        except Exception:
            try:
                wf, sr = torchaudio.load(path)
                if wf.dim() == 2: wf = wf.mean(dim=0)
                waveform, orig_sr = wf, sr
            except Exception:
                try:
                    y, sr = librosa.load(path, sr=None, mono=True)
                    waveform, orig_sr = torch.from_numpy(y), sr
                except Exception:
                    return torch.zeros(self.n_samples, dtype=torch.float32)

        waveform = waveform.to(torch.float32)
        orig_sr  = int(orig_sr) if orig_sr and int(orig_sr) > 0 else self.SAMPLE_RATE
        if orig_sr != self.SAMPLE_RATE and waveform.numel() > 1:
            try:
                waveform = resample(waveform, orig_sr, self.SAMPLE_RATE)
            except Exception:
                return torch.zeros(self.n_samples, dtype=torch.float32)
        if waveform.numel() >= self.n_samples:
            return waveform[:self.n_samples]
        return torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.numel()))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = resolve_audio_path(row, self.audio_dir)
        audio = self._load_audio(path) if path else torch.zeros(self.n_samples, dtype=torch.float32)
        com   = self._nz(row.get("common_name", ""))

        if self.split != "train":
            return audio, com

        if self._order_perm is not None:
            text = self._make_tax_string(row)
        else:
            text = self._bioclio_text(row)

        return audio, text


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
                 proj_hidden_dim=512, proj_out_dim=512, freeze_backbone=False):
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

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))
        feat_dim = getattr(getattr(self.backbone, "config", object()), "projection_dim", 512)
        self.audio_head = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.text_head  = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)

        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.processor.feature_extractor.do_resample = False
        self.processor.feature_extractor.return_attention_mask = False
        self.backbone.train()

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

    def forward(self, audio, texts, sample_rate=48000):
        a  = self.encode_audio(audio, sample_rate)
        t  = self.encode_text(texts)
        s  = self.logit_scale.exp()
        return a, t, s


# ===== Loss =====
def gather_embeddings(emb: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized(): return emb
    outs = [torch.zeros_like(emb) for _ in range(dist.get_world_size())]
    dist.all_gather(outs, emb)
    outs[dist.get_rank()] = emb
    return torch.cat(outs, dim=0)

def clip_loss(audio_emb, text_emb, logit_scale):
    audio_emb = gather_embeddings(audio_emb)
    text_emb  = gather_embeddings(text_emb)
    audio_emb = nn.functional.normalize(audio_emb, dim=-1)
    text_emb  = nn.functional.normalize(text_emb,  dim=-1)
    logits    = audio_emb @ text_emb.T * logit_scale
    labels    = torch.arange(len(logits), device=logits.device)
    loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)) / 2
    return loss


# ===== Main =====
def main():
    args = parse_args()
    setup_distributed()

    _seed = 1337 + (dist.get_rank() if dist.is_initialized() else 0)
    random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed)

    DEVICE = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available() else "cpu"
    )

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = save_dir / "train_log.json"

    if is_rank0():
        wandb.init(project="animalclap", name=args.run_name,
                   config=vars(args))

    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    try:
        if "sox_io" in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend("sox_io")
    except Exception:
        pass

    # Dataset / DataLoader
    train_dataset = AnimalDataset(args.train_csv, args.audio_dir,
                                  split="train", per_species=args.per_species)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if dist.is_initialized():
        sampler     = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        sampler     = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, **loader_kwargs)

    # Model
    model = HFCLAPContrastive("laion/clap-htsat-unfused").to(DEVICE)
    if dist.is_initialized():
        dist.barrier()
        ddp_kwargs = dict(device_ids=[DEVICE.index], output_device=DEVICE.index) \
                     if DEVICE.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=True,
                    broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kwargs)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    logs      = []

    for epoch in range(1, args.epochs + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        total_loss = 0.0
        steps      = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_rank0())

        for audio, texts in pbar:
            audio = audio.to(DEVICE, non_blocking=True)
            texts = list(texts)
            cm    = torch.amp.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
            with cm:
                a_emb, t_emb, scale = model(audio, texts)
                loss = clip_loss(a_emb, t_emb, scale)

            optimizer.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            total_loss += float(loss.detach().cpu())
            steps += 1
            if is_rank0():
                pbar.set_postfix(loss=f"{total_loss/steps:.4f}")

        epoch_loss_t = torch.tensor(total_loss / max(1, steps), device=DEVICE)
        if dist.is_initialized():
            dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.AVG)
        global_avg = float(epoch_loss_t.item())

        if is_rank0():
            ckpt  = save_dir / f"epoch{epoch:03d}.pth"
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state, ckpt)
            wandb.save(str(ckpt))
            wandb.log({"train/epoch_loss": global_avg, "epoch": epoch})
            logs.append({"epoch": epoch, "loss": global_avg})
            with open(log_json_path, "w") as f:
                json.dump(logs, f, indent=2)
            print(f"Epoch {epoch}: loss={global_avg:.4f}")

    if is_rank0():
        final = save_dir / "final.pth"
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state, final)
        print(f"Final model saved: {final}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
