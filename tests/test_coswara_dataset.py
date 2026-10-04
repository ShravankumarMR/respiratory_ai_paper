import math
import os
import wave

import numpy as np
import pytest

from respiratory_ai.data.dataset import CoswaraDataset
from respiratory_ai.training.pipeline import accuracy_and_f1


def _write_wav(path, sample_rate=16000, duration=1.0, freq=440.0):
    num_frames = int(sample_rate * duration)
    import numpy as np

    t = np.linspace(0, duration, num_frames, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        pcm = (samples * 32767).astype("<i2")
        wav_file.writeframes(pcm.tobytes())


@pytest.fixture
def audio_dir(tmp_path):
    root = tmp_path / "coswara"
    root.mkdir()

    _write_wav(root / "subject_001_breathing_deep.wav", duration=1.0)
    _write_wav(root / "subject_002_cough_heavy.wav", duration=2.0)
    _write_wav(root / "subject_003_vowel_a.wav", duration=0.8)

    return root


def test_coswara_dataset_builds_mel_spectrograms_and_labels(audio_dir):
    dataset = CoswaraDataset(
        data_dir=str(audio_dir),
        sound_types=["breathing_deep", "cough_heavy"],
        sr=16000,
        n_mels=64,
        max_duration=5.0,
    )

    assert len(dataset) == 2
    mel, label = dataset[0]

    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 64
    assert mel.shape[2] > 0
    assert label in {0, 1}

    mel_2, label_2 = dataset[1]
    assert label_2 in {0, 1}
    assert mel_2.shape[1] == 64


def test_coswara_dataset_supports_sound_type_filtering(audio_dir):
    dataset = CoswaraDataset(
        data_dir=str(audio_dir),
        sound_types=["vowel_a"],
        sr=16000,
        n_mels=64,
        max_duration=5.0,
    )

    assert len(dataset) == 1
    mel, label = dataset[0]
    assert label == 0
    assert mel.shape[1] == 64


def test_coswara_dataset_skips_empty_wav_files(tmp_path):
    root = tmp_path / "coswara"
    root.mkdir()

    with wave.open(str(root / "subject_001_breathing_deep.wav"), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"")

    num_frames = int(16000 * 1.0)
    t = np.linspace(0, 1.0, num_frames, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    with wave.open(str(root / "subject_002_cough_heavy.wav"), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        pcm = (samples * 32767).astype("<i2")
        wav_file.writeframes(pcm.tobytes())

    dataset = CoswaraDataset(
        data_dir=str(root),
        sound_types=["breathing_deep", "cough_heavy"],
        sr=16000,
        n_mels=64,
        max_duration=1.0,
    )

    assert len(dataset) == 1
    mel, label = dataset[0]
    assert mel.shape[1] == 64
    assert label == 1


def test_accuracy_and_f1_handles_missing_validation_classes():
    y_true = np.array([0, 1, 0, 1, 5, 5], dtype=int)
    y_pred = np.array([0, 1, 0, 1, 5, 5], dtype=int)
    y_score = np.array(
        [
            [0.80, 0.10, 0.02, 0.01, 0.01, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.75, 0.04, 0.02, 0.02, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],
            [0.78, 0.11, 0.03, 0.01, 0.01, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.06, 0.82, 0.03, 0.01, 0.01, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00],
            [0.01, 0.02, 0.01, 0.02, 0.01, 0.88, 0.01, 0.01, 0.00, 0.00, 0.00],
            [0.02, 0.01, 0.01, 0.01, 0.01, 0.90, 0.01, 0.01, 0.00, 0.00, 0.00],
        ],
        dtype=float,
    )

    metrics = accuracy_and_f1(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auc"] > 0.0
