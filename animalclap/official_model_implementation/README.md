# AnimalCLAP

Official implementation of the ICASSP 2026 paper:

**AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining for Species Recognition and Trait Inference**  
Risa Shinoda, Kaede Shiohara, Nakamasa Inoue, Hiroaki Santo, Fumio Okura  
ICASSP 2026 [[Paper](https://doi.org/10.1109/ICASSP55912.2026.11463001)]

---

## Overview

AnimalCLAP is a CLAP-based audio-language model pretrained on animal sounds with taxonomy-aware text representations. It supports:

- Zero-shot species classification
- Trait inference (diet, habitat, locomotion, etc.)

---

## Dataset

The dataset (701,020 recordings from iNaturalist and Xeno-Canto) is available on Hugging Face:

- Audio + metadata: [risashinoda/animalclap-dataset](https://huggingface.co/datasets/risashinoda/animalclap-dataset)
- Species traits: `species_traits.csv` in the same repository

---

## Requirements

```bash
pip install torch torchaudio transformers librosa wandb
```

---

## Training

### 1. Download audio

Train and validation splits contain `download_url` for each recording. Download audio locally before training.

### 2. Run pretraining

```bash
bash run_train.sh
```

Or directly:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --train_csv  /path/to/train.csv \
    --audio_dir  /path/to/audio \
    --save_dir   ./models \
    --run_name   animalclap_exp1
```

To reproduce the ablation experiment (randomized taxonomy order):

```bash
TEXT_MODE=tax_perm_order torchrun --nnodes=1 --nproc_per_node=8 train.py ...
```

### 3. Train traits classifier

```bash
bash run_train_traits.sh
```

Or directly:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_traits.py \
    --train_csv    /path/to/train.csv \
    --test_csv     /path/to/test.csv \
    --traits_csv   /path/to/species_traits.csv \
    --audio_dir    /path/to/audio \
    --encoder_ckpt /path/to/animalclap_final.pth \
    --target_col   diet_type \
    --task_type    multiclass
```

---

## Inference

### Zero-shot species classification

```bash
python inference.py \
    --ckpt      /path/to/checkpoint.pth \
    --test_csv  /path/to/test.csv \
    --data_dir  /path/to/hf_dataset_root \
    --ks 1 5 10
```

### Trait prediction

```bash
python inference_traits.py \
    --ckpt       /path/to/traits_checkpoint.pth \
    --test_csv   /path/to/test.csv \
    --traits_csv /path/to/species_traits.csv \
    --data_dir   /path/to/hf_dataset_root \
    --target_col diet_type \
    --task_type  multiclass
```

---

## Citation

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
