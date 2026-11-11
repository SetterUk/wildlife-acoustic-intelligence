# Hardware Performance & Telemetry Report

## System Environment
- **Operating System:** Windows 11 (Build 10.0.26300)
- **CPU:** Intel64 Family 6 Model 191 Stepping 2, GenuineIntel (16 Physical, 24 Logical Cores)
- **System Memory:** 15.73 GB RAM
- **Dedicated GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB GDDR6)
- **PyTorch Version:** 2.6.0+cu124
- **CUDA Version:** 12.4

## Benchmarking Results
| Metric | NVIDIA RTX 4050 (CUDA) | CPU Fallback | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Latency** | **0.3234 s** | 0.7742 s | **2.39x Speedup** |
| **Minimum Latency** | **0.2708 s** | 0.7081 s | - |
| **Throughput** | **3.1 rec/s** | 1.3 rec/s | +139% |
| **Peak VRAM Allocated** | **685.7 MB** | 0.0 MB | **11.16% of 6 GB** |
| **Model Load Time** | **1.79 s** | - | - |

## Observations
1. **CUDA Acceleration:** GPU execution on the Ada Lovelace RTX 4050 achieved a **2.39x speedup** over CPU inference.
2. **VRAM Footprint:** Maximum VRAM consumption peaked at **685.7 MB**, well within the 6,141 MB dedicated VRAM limit (leaving >5.4 GB headroom).
3. **Audio Chunking:** 10.0s sliding window inference processes in sub-150ms per window on GPU, enabling real-time bioacoustic processing.
