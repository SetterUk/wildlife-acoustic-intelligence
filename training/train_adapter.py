"""
Task-Specific Fine-Tuning & Adapter Training Pipeline for AnimalCLAP.

Implements Parameter-Efficient Transfer Learning:
- Freezes the HTS-AT transformer backbone to preserve general acoustic representations.
- Pre-extracts 512-dim multimodal embeddings with bioacoustic data augmentation.
- Trains a custom non-linear Taxonomy Adapter & Projection Head.
- Uses AdamW optimizer, Cosine Annealing schedule, and Contrastive InfoNCE Loss.
- Saves checkpoint to checkpoints/fine_tuned_wildlife_adapter.pth.
- Generates loss/accuracy curves in results/training_curves.png.
"""

import os
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.model_loader import AnimalCLAPManager
from inference.audio_processor import TARGET_SAMPLE_RATE, CLIP_NUM_SAMPLES
from inference.taxonomy import DEFAULT_CANDIDATE_SPECIES, format_taxonomy_prompt


# --- Custom Adapter Neural Architecture ---
class WildlifeTaxonomicAdapter(nn.Module):
    """
    Task-specific Adapter Head trained on top of frozen AnimalCLAP embeddings.
    Features residual projection, LayerNorm, Dropout, and learned contrastive scaling.
    """
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512, dropout=0.2):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        proj = self.adapter(x)
        return nn.functional.normalize(proj, dim=-1)


# --- Contrastive InfoNCE Loss Function ---
def contrastive_infonce_loss(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Computes symmetric InfoNCE contrastive loss and Top-1 accuracy."""
    scale = logit_scale.exp()
    logits = scale * (audio_emb @ text_emb.t())  # [batch_size, num_classes]

    loss = nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()

    return loss, acc


def generate_augmented_clips(
    raw_audio: np.ndarray,
    num_variants: int = 25
) -> List[torch.Tensor]:
    """Generates augmented 10s audio variants from raw waveform."""
    clips = []
    num_samples = len(raw_audio)

    for _ in range(num_variants):
        if num_samples >= CLIP_NUM_SAMPLES:
            max_start = max(0, num_samples - CLIP_NUM_SAMPLES)
            start = random.randint(0, max_start) if max_start > 0 else 0
            clip = raw_audio[start:start + CLIP_NUM_SAMPLES].copy()
        else:
            repeats = int(np.ceil(CLIP_NUM_SAMPLES / num_samples))
            looped = np.tile(raw_audio, repeats)[:CLIP_NUM_SAMPLES]
            clip = looped.copy()

        # Amplitude scaling
        gain = np.random.uniform(0.75, 1.25)
        clip = clip * gain

        # Additive Gaussian background noise (SNR 18 - 32 dB)
        snr_db = np.random.uniform(18.0, 32.0)
        sig_power = np.mean(clip**2) + 1e-12
        noise_power = sig_power / (10**(snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(clip)).astype(np.float32)
        clip = clip + noise

        # Peak normalization
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip = clip / peak

        clips.append(torch.from_numpy(clip).float())
    return clips


def run_adapter_training(epochs: int = 20, batch_size: int = 16, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================================================================================", flush=True)
    print("ANIMALCLAP TASK-SPECIFIC ADAPTER TRAINING (DL II)", flush=True)
    print("================================================================================", flush=True)
    print(f"Hardware Accelerator: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Pretrained Foundation Backbone
    print("Loading pretrained foundation AnimalCLAP model...", flush=True)
    manager = AnimalCLAPManager.get_instance()
    foundation_model, _, load_info = manager.load_model()
    foundation_model.eval()

    # Freeze entire backbone
    for param in foundation_model.parameters():
        param.requires_grad = False

    # 2. Target Species Catalog
    target_species = [
        s for s in DEFAULT_CANDIDATE_SPECIES
        if s["common_name"] in [
            "Bengal tiger", "Indian rhinoceros", "Indian peafowl",
            "Indian grey wolf", "House cricket", "Asian elephant", "Midwife toad / Tree frog"
        ]
    ]

    species_files = {
        "Bengal tiger": "bengal_tiger_growl.wav",
        "Indian rhinoceros": "indian_rhinoceros_call.wav",
        "Indian peafowl": "indian_peafowl.wav",
        "Indian grey wolf": "grey_wolf_howl.wav",
        "House cricket": "house_cricket_chirp.wav",
        "Asian elephant": "asian_elephant.wav",
        "Midwife toad / Tree frog": "midwife_toad_call.wav"
    }

    # 3. Precompute Text Embeddings for Target Taxonomy
    print("Encoding taxonomy strings into semantic text representations...", flush=True)
    text_prompts = [format_taxonomy_prompt(s, style="tax_com") for s in target_species]
    with torch.no_grad():
        t_feats = foundation_model.encode_text(text_prompts)
        t_proj = nn.functional.normalize(foundation_model.text_head(t_feats), dim=-1).to(device)

    # 4. Extract Bioacoustic Embeddings with Augmentation
    print("Extracting bioacoustic representations from real recordings with augmentation...", flush=True)
    all_embeddings = []
    all_labels = []

    for class_idx, (common_name, filename) in enumerate(species_files.items()):
        filepath = os.path.join(samples_dir, filename)
        if not os.path.exists(filepath):
            continue

        raw_audio, sr = sf.read(filepath, dtype="float32")
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)

        # Generate 25 augmented clips per species
        clips = generate_augmented_clips(raw_audio, num_variants=25)
        stacked_clips = torch.stack(clips, dim=0).to(device)

        with torch.no_grad():
            feats = foundation_model.encode_audio(stacked_clips, sample_rate=TARGET_SAMPLE_RATE)
            all_embeddings.append(feats.cpu())
            all_labels.extend([class_idx] * len(clips))

        print(f"  Processed {common_name:25s}: 25 augmented clips extracted.", flush=True)

    X = torch.cat(all_embeddings, dim=0)  # [175, 512]
    y = torch.tensor(all_labels, dtype=torch.long)

    # 80/20 Train/Val Split
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(0.8 * len(X))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"\nDataset Ready: {len(X_train)} Train samples, {len(X_val)} Validation samples.", flush=True)

    # 5. Instantiate Trainable Adapter Head
    adapter = WildlifeTaxonomicAdapter(in_dim=512, hidden_dim=512, out_dim=512, dropout=0.2).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    ckpt_path = os.path.join(checkpoints_dir, "fine_tuned_wildlife_adapter.pth")

    print(f"\nTraining Adapter Head for {epochs} Epochs on {device}...", flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'LR':<10} | {'Time'}", flush=True)
    print("-" * 75, flush=True)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train Step
        adapter.train()
        train_losses, train_accs = [], []
        for feat_batch, label_batch in train_loader:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device)

            adapted_emb = adapter(feat_batch)
            loss, acc = contrastive_infonce_loss(adapted_emb, t_proj, label_batch, adapter.logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc)

        scheduler.step()

        # Validation Step
        adapter.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for v_feat, v_label in val_loader:
                v_feat = v_feat.to(device)
                v_label = v_label.to(device)

                v_adapted = adapter(v_feat)
                v_loss, v_acc = contrastive_infonce_loss(v_adapted, t_proj, v_label, adapter.logit_scale)
                val_losses.append(v_loss.item())
                val_accs.append(v_acc)

        epoch_train_loss = float(np.mean(train_losses))
        epoch_train_acc = float(np.mean(train_accs))
        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_acc = float(np.mean(val_accs))
        cur_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["learning_rate"].append(round(cur_lr, 7))

        print(
            f"{epoch:<6} | {epoch_train_loss:<12.4f} | {epoch_train_acc*100:<9.1f}% | "
            f"{epoch_val_loss:<10.4f} | {epoch_val_acc*100:<9.1f}% | {cur_lr:<10.1e} | {epoch_time:.2f}s",
            flush=True
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "target_species": target_species
            }, ckpt_path)

    total_training_sec = time.time() - start_total_time
    print("-" * 75, flush=True)
    print(f"Training completed in {total_training_sec:.2f}s! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    print(f"Saved custom trained weights to: {ckpt_path}", flush=True)

    # 6. Save Training History JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 7. Generate Loss and Accuracy Curve Plot
    plot_path = os.path.join(results_dir, "training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss (InfoNCE)", color="#ef4444", lw=2)
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#3b82f6", lw=2, linestyle="--")
    ax1.set_title("Contrastive InfoNCE Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Top-1 Acc", color="#10b981", lw=2)
    ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Top-1 Acc", color="#8b5cf6", lw=2, linestyle="--")
    ax2.set_title("Top-1 Species Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Generated training curve plot at: {plot_path}", flush=True)


if __name__ == "__main__":
    run_adapter_training(epochs=20, batch_size=16, lr=1e-4)
