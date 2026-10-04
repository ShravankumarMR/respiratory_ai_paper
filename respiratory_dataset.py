from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    from scipy.io import wavfile
except Exception:  # pragma: no cover
    wavfile = None


def _normalize_sound_type(value: str) -> str:
    """Normalize names such as 'cough-heavy' or 'Breathing Deep' to a stable token."""
    if value is None:
        return ""
    normalized = str(value).lower().strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def trim_silence(
    audio: torch.Tensor,
    sr: int,
    threshold_db: float = 30.0,
    frame_ms: float = 32.0,
    hop_ms: float = 16.0,
    pad_ms: float = 100.0,
) -> torch.Tensor:
    """Trim leading/trailing near-silence so short bursts (e.g. coughs) aren't diluted by padding."""
    if audio.numel() == 0:
        return audio

    abs_audio = audio.abs()
    peak = abs_audio.max()
    if peak < 1e-6:
        return audio

    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    num_frames = max(1, (audio.numel() - frame_length) // hop_length + 1)

    threshold = peak * (10 ** (-threshold_db / 20.0))
    active_frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = abs_audio[start:start + frame_length]
        if frame.numel() == 0:
            break
        if frame.max() > threshold:
            active_frames.append(i)

    if not active_frames:
        return audio

    pad_samples = int(sr * pad_ms / 1000.0)
    start_sample = max(0, active_frames[0] * hop_length - pad_samples)
    end_sample = min(audio.numel(), active_frames[-1] * hop_length + frame_length + pad_samples)
    return audio[start_sample:end_sample]


class CoswaraDataset(torch.utils.data.Dataset):
    """Coswara respiratory sound dataset that returns a log Mel-spectrogram and label."""

    SOUND_TYPE_ALIASES = {
        "breathing_deep": "breathing_deep",
        "breathing": "breathing",
        "breath": "breathing",
        "deep_breath": "breathing_deep",
        "breathing_shallow": "breathing_shallow",
        "shallow_breath": "breathing_shallow",
        "cough_heavy": "cough_heavy",
        "cough_shallow": "cough_shallow",
        "cough": "cough",
        "coughing": "cough",
        "vowel_a": "vowel_a",
        "vowel_i": "vowel_i",
        "vowel_o": "vowel_o",
        "vowel": "vowel",
        "counting_fast": "counting_fast",
        "counting_normal": "counting_normal",
        "counting": "counting_normal",
        "speech": "speech",
        "voice": "voice",
    }

    def __init__(
        self,
        data_dir: str | Path,
        sound_types: Optional[Sequence[str]] = None,
        sr: int = 16000,
        n_mels: int = 64,
        max_duration: float = 5.0,
        audio_extensions: Optional[Sequence[str]] = None,
        vad_trim: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.max_duration = float(max_duration)
        self.target_samples = int(self.max_duration * self.sr)
        self.vad_trim = bool(vad_trim)
        extensions = audio_extensions or (".wav", ".flac", ".mp3", ".ogg")
        self.audio_extensions = tuple(ext.lower() for ext in extensions)

        raw_types = [] if sound_types is None else list(sound_types)
        self.sound_types = [_normalize_sound_type(item) for item in raw_types]
        self.file_records = self._build_file_records()

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels,
            f_min=50,
            f_max=8000,
        )
        self.log_mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def _collect_audio_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []

        files: List[Path] = []
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.audio_extensions:
                files.append(path)
        return files

    def _infer_sound_type(self, path: Path) -> str:
        text = " ".join([path.name, path.stem, *[part.lower() for part in path.parts]])
        normalized = _normalize_sound_type(text)

        for alias in self.SOUND_TYPE_ALIASES:
            if alias in normalized:
                return self.SOUND_TYPE_ALIASES[alias]

        return "unknown"

    def _build_file_records(self) -> List[Tuple[Path, int]]:
        candidates = self._collect_audio_files()
        if not candidates:
            return []

        detected = []
        for path in candidates:
            sound_type = self._infer_sound_type(path)
            if sound_type != "unknown":
                detected.append(sound_type)

        unique_types = sorted(set(detected)) if not self.sound_types else self.sound_types
        self.class_names = unique_types
        label_map = {name: idx for idx, name in enumerate(unique_types)}

        records: List[Tuple[Path, int]] = []
        seen = set()
        for path in candidates:
            sound_type = self._infer_sound_type(path)
            if sound_type == "unknown":
                continue
            if self.sound_types and sound_type not in self.sound_types:
                continue
            if sound_type not in label_map:
                continue
            path_key = str(path.resolve())
            if path_key in seen:
                continue
            seen.add(path_key)
            records.append((path, label_map[sound_type]))

        return records

    def _load_waveform(self, path: Path) -> torch.Tensor:
        try:
            if sf is not None:
                audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
                if isinstance(audio, list):
                    audio = torch.tensor(audio, dtype=torch.float32)
                else:
                    audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
                if audio.dim() > 1:
                    audio = audio.mean(dim=1)
            elif path.suffix.lower() == ".wav" and wavfile is not None:
                sample_rate, data = wavfile.read(str(path))
                audio = torch.from_numpy(np.asarray(data, dtype=np.float32))
                if audio.dim() > 1:
                    audio = audio.mean(dim=1)
            elif path.suffix.lower() == ".wav":
                with wave.open(str(path), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    frames = wav_file.readframes(wav_file.getnframes())
                    # Convert PCM to float32 in the range [-1, 1].
                    sample_width = wav_file.getsampwidth()
                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    dtype = dtype_map.get(sample_width, np.int16)
                    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    if channels > 1:
                        samples = samples.reshape(-1, channels)
                        audio = samples.mean(axis=1)
                    else:
                        audio = samples
                audio = torch.from_numpy(audio.astype(np.float32))
            else:
                audio, sample_rate = torchaudio.load(str(path))
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)
                audio = audio.float()

            if sf is not None or wavfile is not None or path.suffix.lower() == ".wav":
                audio = audio.float()
                if sample_rate != self.sr:
                    audio = torchaudio.functional.resample(audio, sample_rate, self.sr)
        except Exception:
            try:
                audio, sample_rate = torchaudio.load(str(path))
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)
                audio = audio.float()
                if sample_rate != self.sr:
                    audio = torchaudio.functional.resample(audio, sample_rate, self.sr)
            except Exception:
                # Last-resort fallback for WAV files when even torchaudio fails.
                with wave.open(str(path), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_width = wav_file.getsampwidth()
                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    dtype = dtype_map.get(sample_width, np.int16)
                    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    samples = samples / float(np.iinfo(dtype).max)
                    samples = samples.reshape(-1)
                    audio = torch.from_numpy(samples)
                    if sample_rate != self.sr:
                        audio = torchaudio.functional.resample(audio, sample_rate, self.sr)

        if audio.numel() == 0:
            return torch.zeros(self.target_samples, dtype=torch.float32)

        if self.vad_trim:
            audio = trim_silence(audio, self.sr)

        if audio.numel() < self.target_samples:
            padding = torch.zeros(self.target_samples - audio.numel(), dtype=audio.dtype)
            audio = torch.cat([audio, padding], dim=0)
        else:
            audio = audio[: self.target_samples]

        return audio

    def __len__(self) -> int:
        return len(self.file_records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.file_records[index]
        waveform = self._load_waveform(path)
        mel = self.mel_extractor(waveform.unsqueeze(0)).squeeze(0)
        log_mel = self.log_mel(mel)
        return log_mel.unsqueeze(0), int(label)


__all__ = ["CoswaraDataset", "_normalize_sound_type"]
