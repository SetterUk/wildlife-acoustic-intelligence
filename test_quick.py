import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
import torch
import torch.nn as nn
from transformers import ClapProcessor, ClapModel

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def _to_tensor(x):
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
    def __init__(self, model_id="laion/clap-htsat-unfused",
                 proj_hidden_dim=512, proj_out_dim=512):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        self.backbone  = ClapModel.from_pretrained(model_id, use_safetensors=True, local_files_only=True)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        feat_dim = getattr(getattr(self.backbone, "config", object()), "projection_dim", 512)
        self.audio_head = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.text_head  = ProjectionMLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.processor.feature_extractor.do_resample = False
        self.processor.feature_extractor.return_attention_mask = False

    def _dev(self):
        return next(self.parameters()).device

    def encode_audio(self, audio: torch.Tensor, sample_rate=48000) -> torch.Tensor:
        audio_list = [a.cpu().numpy() for a in audio]
        try:
            inputs = self.processor(
                audio=audio_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
        except (TypeError, ValueError):
            inputs = self.processor(
                audios=audio_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        res = self.backbone.get_audio_features(**inputs)
        return _to_tensor(res)

    def encode_text(self, texts: list) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._dev()) for k, v in inputs.items()}
        res = self.backbone.get_text_features(**inputs)
        return _to_tensor(res)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("Active Device:", device)
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Initial VRAM Allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)

    t0 = time.time()
    model = HFCLAPContrastive("laion/clap-htsat-unfused").to(device)
    ckpt_path = "C:/Users/Acer/.gemini/antigravity/scratch/wildlife-acoustic-intelligence/checkpoints/animalclap_epoch020.pth"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    load_res = model.load_state_dict(sd, strict=False)
    model.eval()
    t1 = time.time()
    print(f"Model load time: {t1 - t0:.2f}s")
    print(f"Load state dict missing keys: {len(load_res.missing_keys)}, unexpected: {len(load_res.unexpected_keys)}")

    # Test inference
    dummy_audio = torch.randn(1, 480000).to(device)
    dummy_texts = ["Indian peafowl", "Bengal tiger", "Asian elephant", "Cricket", "Dolphin"]
    
    t_start = time.time()
    with torch.no_grad():
        a_feat = model.encode_audio(dummy_audio)
        print(f"a_feat shape: {a_feat.shape}, device: {a_feat.device}")
        a_proj = nn.functional.normalize(model.audio_head(a_feat), dim=-1)
        t_feat = model.encode_text(dummy_texts)
        print(f"t_feat shape: {t_feat.shape}, device: {t_feat.device}")
        t_proj = nn.functional.normalize(model.text_head(t_feat), dim=-1)
        logits = a_proj @ t_proj.t()
    inf_time = time.time() - t_start
    print(f"Inference time: {inf_time:.4f}s")
    print(f"Logits shape: {logits.shape}")
    print(f"Similarity scores: {logits.cpu().numpy()[0]}")
    if torch.cuda.is_available():
        print(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Current VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
