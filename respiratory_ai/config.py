from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DEFAULT_SOUND_TYPES = [
    "breathing_deep",
    "breathing_shallow",
    "cough_heavy",
    "cough_shallow",
    "vowel_a",
    "vowel_i",
    "vowel_o",
    "counting_normal",
    "counting_fast",
    "speech",
    "voice",
]


@dataclass
class AppConfig:
    data_dir: str | Path = "coswara-data"
    sound_types: List[str] = field(default_factory=lambda: DEFAULT_SOUND_TYPES.copy())
    sr: int = 16000
    n_mels: int = 64
    max_duration: float = 5.0
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 2e-2
    test_size: float = 0.2
    random_seed: int = 42
    output_dir: str | Path = "artifacts"
    save_name: str = "respiro_attn_best.pth"
    num_workers: int = 0
    vad_trim: bool = True
    use_specaugment: bool = True
    label_smoothing: float = 0.1
    cough_class_weight: float = 1.5
    # Down-weights counting_fast / up-weights counting_normal to fix the precision/recall imbalance between them.
    counting_class_weight: float = 1.3
    use_weighted_sampler: bool = True
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    dropout: float = 0.3

    def as_dict(self):
        return {
            "data_dir": str(self.data_dir),
            "sound_types": self.sound_types,
            "sr": self.sr,
            "n_mels": self.n_mels,
            "max_duration": self.max_duration,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "test_size": self.test_size,
            "random_seed": self.random_seed,
            "output_dir": str(self.output_dir),
            "save_name": self.save_name,
            "num_workers": self.num_workers,
            "vad_trim": self.vad_trim,
            "use_specaugment": self.use_specaugment,
            "label_smoothing": self.label_smoothing,
            "cough_class_weight": self.cough_class_weight,
            "counting_class_weight": self.counting_class_weight,
            "use_weighted_sampler": self.use_weighted_sampler,
            "warmup_epochs": self.warmup_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "dropout": self.dropout,
        }
