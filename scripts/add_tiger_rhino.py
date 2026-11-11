import os
import json
import socket
import urllib.request
import io
import soundfile as sf
import librosa
import numpy as np

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig_getaddrinfo(*a, **k) if r[0] == socket.AF_INET] or _orig_getaddrinfo(*a, **k)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
MANIFEST_PATH = os.path.join(SAMPLES_DIR, "samples_manifest.json")

NEW_RECORDINGS = [
    {
        "id": "bengal_tiger_growl",
        "category": "Mammals",
        "common_name": "Bengal tiger",
        "scientific_name": "Panthera tigris",
        "filename": "bengal_tiger_growl.wav",
        "url": "https://static.inaturalist.org/sounds/2048025.mp3",
        "source": "iNaturalist (Observation Sound 2048025)",
        "license": "Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)",
        "description": "Deep guttural territorial growl and resonance of wild Panthera tigris."
    },
    {
        "id": "indian_rhinoceros_call",
        "category": "Mammals",
        "common_name": "Indian rhinoceros",
        "scientific_name": "Rhinoceros unicornis",
        "filename": "indian_rhinoceros_call.wav",
        "url": "https://static.inaturalist.org/sounds/1732965.mp3",
        "source": "iNaturalist (Observation Sound 1732965)",
        "license": "iNaturalist / Open Access Bioacoustics",
        "description": "Characteristic low-frequency nasal snort and contact vocalization of Greater one-horned rhino."
    },
    {
        "id": "white_rhinoceros_snort",
        "category": "Mammals",
        "common_name": "White rhinoceros",
        "scientific_name": "Ceratotherium simum",
        "filename": "white_rhinoceros_snort.wav",
        "url": "https://static.inaturalist.org/sounds/1423510.mp3",
        "source": "iNaturalist (Observation Sound 1423510)",
        "license": "Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)",
        "description": "Resonant alarm snort and huff of Ceratotherium simum."
    }
]

def main():
    print("Downloading and processing Tiger and Rhinoceros audio recordings...")

    # Load existing manifest
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    # Map existing IDs
    existing_ids = {item.get("id"): idx for idx, item in enumerate(manifest)}

    for item in NEW_RECORDINGS:
        out_path = os.path.join(SAMPLES_DIR, item["filename"])
        print(f"\nProcessing {item['common_name']} ({item['filename']})...")
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "WildlifeAudioResearch/1.0"})
            raw_data = urllib.request.urlopen(req, timeout=20).read()
            
            # Load with librosa for robust mp3 decoding
            y, sr = librosa.load(io.BytesIO(raw_data), sr=48000, mono=True)
            duration = round(len(y) / 48000.0, 2)
            
            # Save standard 16-bit PCM WAV
            sf.write(out_path, y.astype(np.float32), 48000, subtype="PCM_16")
            print(f" Saved {item['filename']} | Duration: {duration}s | Sample Rate: 48,000 Hz")

            entry = dict(item)
            entry["duration_seconds"] = duration
            entry["sample_rate"] = 48000
            entry["channels"] = 1
            entry["format"] = "WAV"
            entry["file_size_bytes"] = os.path.getsize(out_path)

            if item["id"] in existing_ids:
                manifest[existing_ids[item["id"]]] = entry
            else:
                manifest.insert(0, entry) # Add to top of list!

        except Exception as e:
            print(f" Error processing {item['common_name']}: {e}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully updated with {len(manifest)} total items at {MANIFEST_PATH}")

if __name__ == "__main__":
    main()

