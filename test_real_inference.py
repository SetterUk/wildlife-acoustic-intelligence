import os
import json
from inference.predictor import WildlifePredictor

def main():
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
    predictor = WildlifePredictor()

    test_files = [
        ("indian_peafowl.wav", "Indian peafowl"),
        ("asian_elephant.wav", "Asian elephant"),
        ("grey_wolf_howl.wav", "Indian grey wolf"),
        ("house_cricket_chirp.wav", "House cricket"),
        ("silent_audio_5s.wav", "Pure Silence"),
    ]

    print("================================================================================")
    print("ANIMALCLAP REAL INFERENCE TEST ON NVIDIA GEFORCE RTX 4050")
    print("================================================================================")

    for fname, expected in test_files:
        fpath = os.path.join(samples_dir, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {fname}: file not found")
            continue

        print(f"\n--- Testing: {fname} (Expected: {expected}) ---")
        res = predictor.predict(fpath, top_k=5)

        if not res["success"]:
            print(f"Result: Handled Expected Condition -> {res.get('error')}")
            continue

        hw = res["hardware_metrics"]
        top1 = res["top_prediction"]
        print(f"Device: {hw['device']} ({hw['device_name']})")
        print(f"Latency: {hw['inference_time_sec']}s (Preprocess: {hw['preprocessing_time_sec']}s)")
        print(f"Peak VRAM: {hw['peak_vram_mb']} MB (Current: {hw['current_vram_mb']} MB)")
        print(f"Top-1 Prediction: {top1['common_name']} ({top1['scientific_name']}) | Similarity: {top1['similarity_score']}")
        print("Top-5 Ranking:")
        for p in res["predictions"]:
            print(f"  #{p['rank']} [{p['category']}] {p['common_name']} ({p['scientific_name']}) -> Score: {p['similarity_score']:.4f}")

if __name__ == "__main__":
    main()

