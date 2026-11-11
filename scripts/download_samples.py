import os
import json
import time
import socket
import urllib.request
import soundfile as sf
import numpy as np

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig_getaddrinfo(*a, **k) if r[0] == socket.AF_INET] or _orig_getaddrinfo(*a, **k)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

USER_AGENT = "WildlifeAcousticIntelligence/1.0 (https://github.com/dahlian00/AnimalCLAP; student-research@edu.local) Python-urllib/3.12"

PUBLIC_RECORDINGS = [
    {
        "id": "indian_peafowl",
        "category": "Birds",
        "common_name": "Indian peafowl",
        "scientific_name": "Pavo cristatus",
        "filename": "indian_peafowl.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Pavo%2Bcristatus.wav",
        "source": "Wikimedia Commons / Bioacoustics Archive",
        "license": "Creative Commons Attribution-ShareAlike (CC BY-SA 3.0)",
        "description": "Loud, resonant territorial call of a wild male Indian Peafowl."
    },
    {
        "id": "asian_elephant",
        "category": "Mammals",
        "common_name": "Asian elephant",
        "scientific_name": "Elephas maximus",
        "filename": "asian_elephant.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/40/Elephant_voice_-_trumpeting.ogg",
        "source": "Wikimedia Commons (Elephant voice - trumpeting)",
        "license": "Creative Commons CC0 1.0 Universal Public Domain",
        "description": "High-intensity trumpeting vocalization of an Asian elephant."
    },
    {
        "id": "grey_wolf",
        "category": "Mammals",
        "common_name": "Indian grey wolf",
        "scientific_name": "Canis lupus",
        "filename": "grey_wolf_howl.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/52/11wolf.wav",
        "source": "Wikimedia Commons / US Fish & Wildlife Service",
        "license": "Public Domain",
        "description": "Sustained long-range harmonic pack howl of Canis lupus."
    },
    {
        "id": "cricket",
        "category": "Insects",
        "common_name": "House cricket",
        "scientific_name": "Acheta domesticus",
        "filename": "house_cricket_chirp.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2a/Acheta-domesticus-Stridulation.ogg",
        "source": "Wikimedia Commons (Acheta-domesticus-Stridulation)",
        "license": "Creative Commons Attribution-ShareAlike (CC BY-SA 3.0)",
        "description": "Rhythmic high-frequency tegminal stridulation chirps of a cricket."
    },
    {
        "id": "toad_frog",
        "category": "Amphibians",
        "common_name": "Midwife toad / Tree frog",
        "scientific_name": "Alytes obstetricans",
        "filename": "midwife_toad_call.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/33/1_vroedmeesterpad_koor_Alytes_obstetricans_choir.wav",
        "source": "Wikimedia Commons (Bioacoustics Field Choir)",
        "license": "Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)",
        "description": "Clear resonant piping bell-like nocturnal mating call chorus."
    },
    {
        "id": "dolphin_whale",
        "category": "Marine animals",
        "common_name": "Dolphin",
        "scientific_name": "Delphinidae",
        "filename": "dolphin_marine_vocal.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4b/Whales_and_Dolphins_whale_nature_sounds_songs_nueva_esparta.ogg",
        "source": "Wikimedia Commons (Whales and Dolphins Nueva Esparta)",
        "license": "Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)",
        "description": "Hydrophone acoustic recording of marine mammal whistles and tonal clicks."
    }
]

def download_and_process():
    manifest = []
    print("Downloading verified public wildlife audio recordings...")

    for item in PUBLIC_RECORDINGS:
        out_path = os.path.join(SAMPLES_DIR, item["filename"])
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f" Existing verified file: {item['filename']}")
            data, sr = sf.read(out_path)
            duration = round(len(data) / sr, 2)
            entry = dict(item)
            entry["duration_seconds"] = duration
            entry["sample_rate"] = sr
            entry["channels"] = 1
            entry["format"] = "WAV"
            entry["file_size_bytes"] = os.path.getsize(out_path)
            manifest.append(entry)
            continue

        print(f"Downloading {item['common_name']} ({item['filename']})...")
        time.sleep(2.0)  # Wikimedia polite rate limiting
        try:
            req = urllib.request.Request(
                item["url"],
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*"
                }
            )
            raw_data = urllib.request.urlopen(req, timeout=30).read()
            temp_path = out_path + ".download"
            with open(temp_path, "wb") as f:
                f.write(raw_data)

            data, sr = sf.read(temp_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(out_path, data.astype(np.float32), sr, subtype="PCM_16")
            if os.path.exists(temp_path):
                os.remove(temp_path)

            duration = round(len(data) / sr, 2)
            print(f" Saved {item['filename']} | SR: {sr} Hz | Duration: {duration}s | Channels: 1")

            entry = dict(item)
            entry["duration_seconds"] = duration
            entry["sample_rate"] = sr
            entry["channels"] = 1
            entry["format"] = "WAV"
            entry["file_size_bytes"] = os.path.getsize(out_path)
            manifest.append(entry)
        except Exception as e:
            print(f" Error downloading {item['common_name']}: {e}")

    # Synthesize edge cases
    print("\nEnsuring edge-case test files exist...")
    
    # 1. Pure silence
    silent_path = os.path.join(SAMPLES_DIR, "silent_audio_5s.wav")
    if not os.path.exists(silent_path):
        silence = np.zeros(48000 * 5, dtype=np.float32)
        sf.write(silent_path, silence, 48000, subtype="PCM_16")
    manifest.append({
        "id": "test_silence",
        "category": "Edge Cases",
        "common_name": "Pure Silence",
        "scientific_name": "N/A (Control Test)",
        "filename": "silent_audio_5s.wav",
        "source": "Generated synthetically (zeros array)",
        "license": "CC0 Public Domain",
        "description": "5.00s of absolute silence (amplitude 0.0) to test silence/energy detection.",
        "duration_seconds": 5.0,
        "sample_rate": 48000,
        "channels": 1,
        "format": "WAV",
        "file_size_bytes": os.path.getsize(silent_path)
    })

    # 2. Corrupted file
    corrupted_path = os.path.join(SAMPLES_DIR, "corrupted_audio_file.wav")
    if not os.path.exists(corrupted_path):
        with open(corrupted_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPT_INVALID_HEADER_DATA_FAIL")
    manifest.append({
        "id": "test_corrupted",
        "category": "Edge Cases",
        "common_name": "Corrupted Audio",
        "scientific_name": "N/A (Error Handling Test)",
        "filename": "corrupted_audio_file.wav",
        "source": "Generated synthetically (corrupted byte sequence)",
        "license": "CC0 Public Domain",
        "description": "Audio file with broken headers to test robust exception handling.",
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "format": "Corrupted WAV",
        "file_size_bytes": os.path.getsize(corrupted_path)
    })

    # 3. Unsupported text file
    unsupported_path = os.path.join(SAMPLES_DIR, "unsupported_document.txt")
    if not os.path.exists(unsupported_path):
        with open(unsupported_path, "w", encoding="utf-8") as f:
            f.write("This is a plain text file, not a biological audio recording.")
    manifest.append({
        "id": "test_unsupported",
        "category": "Edge Cases",
        "common_name": "Unsupported Format",
        "scientific_name": "N/A (Validation Test)",
        "filename": "unsupported_document.txt",
        "source": "Generated synthetically (text document)",
        "license": "CC0 Public Domain",
        "description": "Non-audio text file to verify upload validation filter.",
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "format": "TXT",
        "file_size_bytes": os.path.getsize(unsupported_path)
    })

    # 4. Multi-samplerate (16 kHz version of Peafowl)
    peafowl_path = os.path.join(SAMPLES_DIR, "indian_peafowl.wav")
    if os.path.exists(peafowl_path):
        sr16_path = os.path.join(SAMPLES_DIR, "indian_peafowl_16khz.wav")
        if not os.path.exists(sr16_path):
            data, sr = sf.read(peafowl_path)
            import scipy.signal
            num_samples_16k = int(len(data) * 16000 / sr)
            resampled_16k = scipy.signal.resample(data, num_samples_16k).astype(np.float32)
            sf.write(sr16_path, resampled_16k, 16000, subtype="PCM_16")
        manifest.append({
            "id": "test_sr_16k",
            "category": "Birds (Resampled)",
            "common_name": "Indian peafowl (16 kHz)",
            "scientific_name": "Pavo cristatus",
            "filename": "indian_peafowl_16khz.wav",
            "source": "Resampled from Pavo+cristatus.wav",
            "license": "CC BY-SA 3.0",
            "description": "Standard 16 kHz sample rate variant to test torchaudio functional resampling to 48 kHz.",
            "duration_seconds": round(os.path.getsize(sr16_path) / (16000 * 2), 2),
            "sample_rate": 16000,
            "channels": 1,
            "format": "WAV",
            "file_size_bytes": os.path.getsize(sr16_path)
        })

    # 5. Long audio (>25s)
    elephant_path = os.path.join(SAMPLES_DIR, "asian_elephant.wav")
    if os.path.exists(elephant_path):
        long_path = os.path.join(SAMPLES_DIR, "asian_elephant_long_28s.wav")
        if not os.path.exists(long_path):
            e_data, e_sr = sf.read(elephant_path)
            repeats = int(np.ceil(28.0 * e_sr / len(e_data)))
            long_data = np.tile(e_data, repeats)[:int(28.0 * e_sr)]
            sf.write(long_path, long_data.astype(np.float32), e_sr, subtype="PCM_16")
        manifest.append({
            "id": "test_long_audio",
            "category": "Mammals (Long Audio)",
            "common_name": "Asian elephant (28s recording)",
            "scientific_name": "Elephas maximus",
            "filename": "asian_elephant_long_28s.wav",
            "source": "Extended loop of Elephant voice - trumpeting.ogg",
            "license": "CC0 1.0",
            "description": "28.00s extended duration recording to verify sliding window audio chunking.",
            "duration_seconds": 28.0,
            "sample_rate": 44100,
            "channels": 1,
            "format": "WAV",
            "file_size_bytes": os.path.getsize(long_path)
        })

    # 6. FLAC Peafowl
    if os.path.exists(peafowl_path):
        flac_path = os.path.join(SAMPLES_DIR, "indian_peafowl.flac")
        if not os.path.exists(flac_path):
            data, sr = sf.read(peafowl_path)
            sf.write(flac_path, data.astype(np.float32), sr, format="FLAC")
        manifest.append({
            "id": "test_flac_peafowl",
            "category": "Birds (FLAC format)",
            "common_name": "Indian peafowl (FLAC)",
            "scientific_name": "Pavo cristatus",
            "filename": "indian_peafowl.flac",
            "source": "Converted from Pavo+cristatus.wav",
            "license": "CC BY-SA 3.0",
            "description": "Lossless FLAC format encoding of Indian Peafowl vocalization.",
            "duration_seconds": 14.73,
            "sample_rate": 48000,
            "channels": 1,
            "format": "FLAC",
            "file_size_bytes": os.path.getsize(flac_path)
        })

    manifest_file = os.path.join(SAMPLES_DIR, "samples_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n Manifest updated: {manifest_file} ({len(manifest)} items).")

if __name__ == "__main__":
    download_and_process()
