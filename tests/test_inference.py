"""
Comprehensive Automated Test Suite for AnimalCLAP Wildlife Acoustic Intelligence.
Tests multi-species recognition, audio formats, resampling, and edge cases.
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predictor import WildlifePredictor
from inference.audio_processor import AudioProcessor, AudioValidationError
from inference.taxonomy import DEFAULT_CANDIDATE_SPECIES
from inference.model_loader import AnimalCLAPManager


class TestAnimalCLAPInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = WildlifePredictor()
        cls.samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")

    def test_01_cuda_availability(self):
        """Verify CUDA is enabled and detects RTX 4050 GPU."""
        status = AnimalCLAPManager.get_gpu_status()
        self.assertTrue(status["cuda_available"], "CUDA should be available on RTX 4050")
        self.assertIn("RTX 4050", status["device_name"])
        self.assertGreater(status["total_vram_gb"], 5.0)

    def test_02_bird_recognition(self):
        """Verify Indian Peafowl vocalization recognition."""
        path = os.path.join(self.samples_dir, "indian_peafowl.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["top_prediction"]["common_name"], "Indian peafowl")
        self.assertEqual(res["top_prediction"]["scientific_name"], "Pavo cristatus")
        self.assertGreater(res["top_prediction"]["similarity_score"], 0.6)
        self.assertEqual(res["hardware_metrics"]["device"], "cuda")

    def test_03_mammal_recognition(self):
        """Verify Indian Grey Wolf vocalization recognition."""
        path = os.path.join(self.samples_dir, "grey_wolf_howl.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["top_prediction"]["common_name"], "Indian grey wolf")
        self.assertGreater(res["top_prediction"]["similarity_score"], 0.45)
        self.assertEqual(res["hardware_metrics"]["device"], "cuda")

    def test_04_insect_recognition(self):
        """Verify House Cricket stridulation recognition."""
        path = os.path.join(self.samples_dir, "house_cricket_chirp.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["top_prediction"]["common_name"], "House cricket")
        self.assertGreater(res["top_prediction"]["similarity_score"], 0.5)

    def test_05_amphibian_recognition(self):
        """Verify Midwife Toad / Frog chorus inference."""
        path = os.path.join(self.samples_dir, "midwife_toad_call.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        # Check that predictions are populated and top candidate has positive similarity
        self.assertEqual(len(res["predictions"]), 5)
        self.assertGreater(res["top_prediction"]["similarity_score"], 0.2)

    def test_06_different_samplerate_resampling(self):
        """Verify 16 kHz recording is resampled to 48 kHz and classified."""
        path = os.path.join(self.samples_dir, "indian_peafowl_16khz.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["audio_metadata"]["original_sample_rate"], 16000)
        self.assertEqual(res["audio_metadata"]["target_sample_rate"], 48000)
        self.assertEqual(res["top_prediction"]["common_name"], "Indian peafowl")

    def test_07_flac_format(self):
        """Verify FLAC format input support."""
        path = os.path.join(self.samples_dir, "indian_peafowl.flac")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["top_prediction"]["common_name"], "Indian peafowl")

    def test_08_long_audio_chunking(self):
        """Verify long audio (>25s) is partitioned into sliding window chunks."""
        path = os.path.join(self.samples_dir, "asian_elephant_long_28s.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertTrue(res["success"])
        self.assertGreater(res["audio_metadata"]["num_chunks"], 1)
        self.assertGreaterEqual(res["audio_metadata"]["original_duration_seconds"], 25.0)

    def test_09_silence_detection(self):
        """Verify silence detection prevents false predictions on pure silence."""
        path = os.path.join(self.samples_dir, "silent_audio_5s.wav")
        res = self.predictor.predict(path, top_k=5)
        self.assertFalse(res["success"])
        self.assertIn("Silence detected", res["error"])

    def test_10_corrupted_file_handling(self):
        """Verify corrupted audio files raise clean AudioValidationError."""
        path = os.path.join(self.samples_dir, "corrupted_audio_file.wav")
        with self.assertRaises(AudioValidationError):
            AudioProcessor.load_and_preprocess(path)

    def test_11_unsupported_format_handling(self):
        """Verify non-audio documents raise clean AudioValidationError."""
        path = os.path.join(self.samples_dir, "unsupported_document.txt")
        with self.assertRaises(AudioValidationError):
            AudioProcessor.load_and_preprocess(path)

    def test_12_repeated_inference_memory_stability(self):
        """Verify repeated inference maintains stable VRAM without memory leaks."""
        path = os.path.join(self.samples_dir, "indian_peafowl.wav")
        vram_readings = []
        for _ in range(5):
            res = self.predictor.predict(path, top_k=5)
            self.assertTrue(res["success"])
            vram_readings.append(res["hardware_metrics"]["current_vram_mb"])

        # VRAM should not balloon monotonically across calls
        delta = max(vram_readings) - min(vram_readings)
        self.assertLess(delta, 50.0, f"VRAM fluctuated by {delta} MB across 5 runs")


if __name__ == "__main__":
    unittest.main()

