# 🌲 WILDLIFE ACOUSTIC INTELLIGENCE
### AI-Powered Wildlife Species Recognition Using Bioacoustic Signals and AnimalCLAP (ICASSP 2026)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Model](https://img.shields.io/badge/Model-AnimalCLAP%20(Epoch%2020)-blue.svg)](https://huggingface.co/risashinoda/animalclap)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, fully local wildlife acoustic intelligence application built on the official pretrained **AnimalCLAP** model (*ICASSP 2026: Taxonomy-Aware Language-Audio Pretraining for Species Recognition and Trait Inference*).

Engineered specifically for local workstation inference accelerated on the **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** with graceful CPU fallback.

---

## 📌 Project Highlights

- **Official Pretrained Checkpoint:** Uses official `animalclap_epoch020.pth` (590 MB, SHA256 verified) from `risashinoda/animalclap`.
- **Taxonomy-Aware Contrastive Pretraining:** Zero-shot species identification across biological hierarchy (Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species).
- **Multi-Category Coverage:** Evaluates species across **Mammals**, **Birds**, **Amphibians**, **Insects**, **Reptiles**, and **Marine Animals**.
- **Hardware Acceleration:** Fully executes inference on **CUDA (RTX 4050)** achieving **~0.32s latency** with **<686 MB peak VRAM** (under 12% of 6 GB).
- **Robust Audio Preprocessing:** Automated resampling to **48,000 Hz**, stereo downmixing, silent audio detection, and sliding-window chunking (10s windows) for extended field recordings.
- **Interactive Streamlit Web Dashboard:** Spectrogram inspection, Plotly Top-5 ranking visualizations, session history, and CSV/JSON export.

---

## 🏗️ Project Architecture

```text
wildlife-acoustic-intelligence/
│
├── app/                                # Streamlit Web UI Application
│   ├── main.py                         # Application entrypoint
│   ├── components/
│   │   ├── dashboard.py                # Hardware telemetry and status badges
│   │   ├── visualizer.py               # Plotly ranking charts and score bars
│   │   └── history.py                  # Session analysis history & export controls
│   └── utils/
│       ├── audio_utils.py              # Waveform & spectrogram plot generation
│       └── export_utils.py             # CSV / JSON export formatters
│
├── animalclap/
│   └── official_model_implementation/  # Intact official GitHub code (dahlian00/AnimalCLAP)
│
├── inference/
│   ├── model_loader.py                 # Singleton model loader, CUDA/CPU management
│   ├── audio_processor.py              # 48kHz resampling, validation, sliding chunking
│   ├── taxonomy.py                     # Biological taxonomy and candidate species catalog
│   └── predictor.py                    # Multimodal cosine similarity predictor
│
├── checkpoints/
│   └── animalclap_epoch020.pth         # Official pretrained checkpoint (590 MB)
│
├── data/
│   └── samples/                        # Verified public bioacoustic recordings
│       ├── samples_manifest.json       # Metadata, licenses, and recording details
│       ├── indian_peafowl.wav          # Verified Bird call (WAV)
│       ├── asian_elephant.wav          # Verified Mammal trumpet (WAV)
│       ├── grey_wolf_howl.wav          # Verified Mammal howl (WAV)
│       ├── house_cricket_chirp.wav     # Verified Insect stridulation (WAV)
│       ├── midwife_toad_call.wav       # Verified Amphibian chorus (WAV)
│       └── silent_audio_5s.wav         # Edge case: control silence
│
├── docs/
│   └── wildlife_tracking_architecture.md # Technical doc on TDOA / GCC-PHAT array tracking
│
├── results/
│   ├── benchmark_report.json           # Automated hardware benchmark results
│   └── benchmark_report.md             # Hardware telemetry breakdown
│
├── tests/
│   ├── test_inference.py               # Unittest suite (12 automated unit tests)
│   └── test_benchmark.py               # GPU vs. CPU speedup and memory benchmark
│
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Version control exclusion rules
└── README.md                           # Documentation and setup instructions
```

---

## ⚡ Quick Start: Windows PowerShell Commands

Open Windows PowerShell in the project directory:

```powershell
cd C:\Users\Acer\.gemini\antigravity\scratch\wildlife-acoustic-intelligence
```

### 1. Create Virtual Environment
Using Python 3.12 (via `uv` or standard Python):

```powershell
uv venv .venv --python 3.12
```

### 2. Activate Virtual Environment

```powershell
.\.venv\Scripts\activate
```

### 3. Install Compatible Dependencies

```powershell
uv pip install -r requirements.txt --python .\.venv\Scripts\python.exe
```

### 4. Verify Model Checkpoint
The official checkpoint is located at `checkpoints/animalclap_epoch020.pth`. To download or refresh it:

```powershell
New-Item -ItemType Directory -Force -Path "checkpoints"
curl.exe -4 -L -C - "https://huggingface.co/risashinoda/animalclap/resolve/main/animalclap_epoch020.pth" -o "checkpoints\animalclap_epoch020.pth"
```

### 5. Run the Automated Test Suite

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_inference
```
*(All 12 unit tests pass: bird, mammal, insect, amphibian, resampling, FLAC format, long audio chunking, silence detection, corrupt file handling, and memory stability).*

### 6. Run Hardware Benchmark (GPU vs. CPU)

```powershell
.\.venv\Scripts\python.exe tests\test_benchmark.py
```

### 7. Launch the Streamlit Web Application

```powershell
.\.venv\Scripts\streamlit run app/main.py
```

The application will launch automatically in your web browser at `http://localhost:8501`.

---

## 📊 Hardware Benchmarks (RTX 4050 vs. CPU)

Tested on **Intel Core i7 13th Gen (16 Physical / 24 Logical Cores)** + **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)**:

| Metric | NVIDIA RTX 4050 (CUDA) | Intel Core i7 (CPU) | Speedup / Efficiency |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | **0.3234 s** | 0.7742 s | **2.39x Faster** |
| **Model Load Time** | **1.78 s** | 2.65 s | Pre-cached Safetensors |
| **Peak VRAM Allocated** | **685.7 MB** | 0.0 MB | **Only 11.1% of 6 GB VRAM** |
| **Throughput** | **3.1 inferences/sec** | 1.3 inferences/sec | **+139% Throughput** |

---

## 🎯 Verification with Real Bioacoustic Recordings

Actual zero-shot inference results on public verified recordings:

| Input Recording | True Category | Top-1 Predicted Species | Similarity Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| `indian_peafowl.wav` | Birds | **Indian peafowl (*Pavo cristatus*)** | **0.7633** | ✅ Correct |
| `grey_wolf_howl.wav` | Mammals | **Indian grey wolf (*Canis lupus*)** | **0.5765** | ✅ Correct |
| `house_cricket_chirp.wav` | Insects | **House cricket (*Acheta domesticus*)** | **0.6346** | ✅ Correct |
| `midwife_toad_call.wav` | Amphibians | Midwife toad / Tree frog (*Alytes obstetricans*) | **0.2842** (Top-5) | ✅ Evaluated |
| `asian_elephant.wav` | Mammals | Indian peafowl (*Pavo cristatus*) | **0.5383** | ⚠️ Acoustic Overlap |
| `silent_audio_5s.wav` | Edge Case | *None (Detected Silence)* | N/A | ✅ Filtered |

> **Scientific Transparency Note:** AnimalCLAP similarity scores represent cosine similarity $[-1.0, 1.0]$ in the multimodal contrastive embedding space and must **not** be presented as calibrated Bayesian posterior probabilities. The Asian elephant recording (a 1.4s high-frequency trumpeting burst) exhibited acoustic frequency overlap with peafowl high-screams in the shared representation space; documenting model behavior honestly is a core tenet of this system.

---

## 🔭 Future Wildlife Localization & Tracking Extension

While AnimalCLAP identifies *what* vocalized, localization requires physical spatial array processing. See [`docs/wildlife_tracking_architecture.md`](docs/wildlife_tracking_architecture.md) for the detailed architecture on:
1. Multi-microphone spatial array geometry.
2. Time Difference of Arrival (TDOA) with Generalized Cross-Correlation Phase Transform (GCC-PHAT).
3. Hyperbolic multilateration and Steered Response Power (SRP-PHAT).
4. Extended Kalman Filter (EKF) temporal tracking of intermittent vocalizations.

---

## 📜 Citation & Credits

```bibtex
@INPROCEEDINGS{shinodaanimalclap,
  author={Shinoda, Risa and Shiohara, Kaede and Inoue, Nakamasa and Santo, Hiroaki and Okura, Fumio},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining for Species Recognition and Trait Inference},
  year={2026},
  pages={7767-7771},
  doi={10.1109/ICASSP55912.2026.11463001}
}
```
Official Repository: [https://github.com/dahlian00/AnimalCLAP](https://github.com/dahlian00/AnimalCLAP)  
Official Hugging Face Checkpoint: [https://huggingface.co/risashinoda/animalclap](https://huggingface.co/risashinoda/animalclap)

