"""Explainable respiratory sound classification pipeline for the Coswara dataset.

This script is designed to be run step-by-step in a research workflow:
1. Load the downloaded Coswara audio data.
2. Create train/validation splits.
3. Train a CNN with CBAM attention and a plain CNN baseline.
4. Evaluate using accuracy, macro-F1, and AUC.
5. Visualize attention maps.
6. Perform attention stability analysis under noisy conditions.
7. Run standalone inference on a single wav file.

Usage examples:
    python respiratory_attention_pipeline.py --data_dir coswara-data --sound_types cough_heavy,breathing_deep,vowel_a --epochs 15
    python respiratory_attention_pipeline.py --mode evaluate --checkpoint respiro_attn_best.pth
    python respiratory_attention_pipeline.py --mode predict --model_path respiro_attn_best.pth --wav_path path/to/file.wav
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

try:
    from respiratory_dataset import CoswaraDataset
except Exception:  # pragma: no cover
    # Fallback import in case the file is copied or the module path is different.
    from src.respiratory_dataset import CoswaraDataset  # type: ignore


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
class TrainingConfig:
    """Training and data configuration for the respiratory pipeline."""

    data_dir: str = "coswara-data"
    sound_types: List[str] = field(default_factory=lambda: DEFAULT_SOUND_TYPES.copy())
    sr: int = 16000
    n_mels: int = 64
    max_duration: float = 5.0
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    test_size: float = 0.2
    random_seed: int = 42
    output_dir: str = "artifacts"
    save_name: str = "respiro_attn_best.pth"
    num_workers: int = 0


class ChannelAttention(nn.Module):
    """SE-like channel attention with average + max pooling."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.shared_mlp_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(x)
        max_out = self.shared_mlp_max(x)
        return self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention built from average and max feature responses."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_map, max_map], dim=1)
        return self.conv(combined)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.conv(x)
        attention = self.channel_attention(residual) * residual
        spatial_attention = self.spatial_attention(attention)
        out = attention * spatial_attention
        out = self.relu(out)
        return out, spatial_attention


class ConvBlock(nn.Module):
    """Plain convolution block used in both the attention and baseline model."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RespiroAttnNet(nn.Module):
    """CNN backbone with CBAM attention for respiratory sound classification.

    Forward returns both logits and the spatial attention map. The attention map is
    extracted from the final CBAM block and resized to the input Mel shape in the
    visualization pipeline.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.cbam = CBAMBlock(256, 256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features, spatial_attention = self.cbam(features)
        logits = self.classifier(features)
        return logits, spatial_attention


class PlainCNN(nn.Module):
    """Plain CNN baseline without attention for comparison."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        activations = features.mean(dim=1, keepdim=True)
        attention = activations / (activations.abs().max().clamp_min(1e-8))
        return logits, attention


def parse_sound_types(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    return tokens or None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mel_from_wav(wav_path: str | Path, sr: int = 16000, n_mels: int = 64, max_duration: float = 5.0) -> torch.Tensor:
    """Convert raw .wav audio into a log Mel-spectrogram tensor."""
    wav_path = Path(wav_path)

    if wav_path.suffix.lower() == ".wav":
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sample_width, np.int16)
            samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if channels > 1:
                samples = samples.reshape(-1, channels)
                waveform = samples.mean(axis=1)
            else:
                waveform = samples
            waveform = waveform / float(np.iinfo(dtype).max)
    else:
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.float()

    waveform = torch.from_numpy(waveform.astype(np.float32))
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)

    target_samples = int(max_duration * sr)
    if waveform.numel() < target_samples:
        pad = torch.zeros(target_samples - waveform.numel(), dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad], dim=0)
    else:
        waveform = waveform[:target_samples]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        f_min=50,
        f_max=8000,
    )(waveform.unsqueeze(0))
    log_mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)
    return log_mel.squeeze(0).unsqueeze(0)


def create_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train and validation dataloaders from the local Coswara folder."""
    dataset = CoswaraDataset(
        data_dir=config.data_dir,
        sound_types=config.sound_types,
        sr=config.sr,
        n_mels=config.n_mels,
        max_duration=config.max_duration,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No labeled audio files found in '{config.data_dir}'. "
            "Check that the Coswara data directory is correct and that the selected sound types are valid."
        )

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=config.test_size,
        stratify=labels,
        random_state=config.random_seed,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, dataset.class_names


def accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if y_score.ndim == 1 or y_score.shape[1] == 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "auc": float(auc)}


def resize_attention(attention_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Resize a single attention map to the Mel-spectrogram shape."""
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(0)
    if attention_map.shape[1] != 1:
        attention_map = attention_map.mean(dim=1, keepdim=True)
    return F.interpolate(attention_map, size=target_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_true: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, _ = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_true.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(all_true)
    y_score = np.concatenate(all_logits, axis=0)
    y_pred = y_score.argmax(axis=1)
    metrics = accuracy_and_f1(y_true, y_pred, y_score)
    return metrics, y_true, y_pred, y_score


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, criterion: nn.Module):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def train_pipeline(config: TrainingConfig) -> Dict[str, object]:
    """Train the attention model and save the best checkpoint."""
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = create_dataloaders(config)
    num_classes = len(class_names)

    model = RespiroAttnNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
        val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(model, val_loader, device)
        current_score = val_metrics["macro_f1"] + 0.5 * val_metrics["auc"]

        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['macro_f1']:.4f} | val_auc={val_metrics['auc']:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = val_metrics.copy()

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    best_path = output_dir / config.save_name
    torch.save({
        "model_state": best_state,
        "class_names": class_names,
        "config": asdict(config),
        "best_metrics": best_metrics,
    }, best_path)
    print(f"Best model saved to: {best_path}")

    val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(model, val_loader, device)
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    return {
        "model": model,
        "class_names": class_names,
        "val_metrics": val_metrics,
        "best_path": str(best_path),
    }


def plot_attention_example(mel: np.ndarray, attention_map: np.ndarray, label: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """Plot a Mel-spectrogram, attention map, and overlay."""
    mel = mel.squeeze(0) if mel.ndim == 3 else mel
    attention_map = attention_map.squeeze() if attention_map.ndim > 2 else attention_map

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.set_facecolor("white")

    ax = axes[0]
    im0 = ax.imshow(mel, cmap="magma", aspect="auto", origin="lower")
    ax.set_title("Mel-spectrogram")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(attention_map, cmap="jet", aspect="auto", origin="lower")
    ax.set_title("Attention map")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    overlay = ax.imshow(mel, cmap="magma", aspect="auto", origin="lower")
    attention_resized = attention_map
    if attention_resized.shape != mel.shape:
        attention_resized = np.array(plt.cm.jet((attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)))[:, :, :3]
        attention_resized = np.transpose(attention_resized, (0, 1, 2))
    ax.imshow(attention_map, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    ax.set_title("Overlay")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    if label is not None:
        ax.text(0.02, 0.98, f"Class: {label}", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def visualize_attention_examples(model: nn.Module, loader: DataLoader, class_names: Sequence[str], output_dir: str, max_examples: int = 3, device: Optional[torch.device] = None):
    """Save a small set of high-quality attention visualizations."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    counter = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, attention = model(inputs)
            preds = logits.argmax(dim=1)
            for idx in range(inputs.size(0)):
                if counter >= max_examples:
                    return
                mel = inputs[idx].detach().cpu().numpy()
                attn = attention[idx].detach().cpu().numpy()
                attn = resize_attention(torch.tensor(attn).unsqueeze(0), (mel.shape[1], mel.shape[2]))
                if attn.shape != mel.shape[-2:]:
                    attn = attn[: mel.shape[-2], : mel.shape[-1]]
                label_name = class_names[int(labels[idx].item())]
                fig = plot_attention_example(
                    mel=mel,
                    attention_map=attn.numpy() if hasattr(attn, "numpy") else attn,
                    label=label_name,
                    save_path=str(output_dir / f"attention_example_{counter + 1}.png"),
                )
                plt.close(fig)
                counter += 1


def add_noise_to_mel(mel: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to a Mel-spectrogram at the requested SNR level."""
    signal_power = torch.mean(mel.float() ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(mel, dtype=torch.float32) * math.sqrt(noise_power.item())
    return mel + noise


def spearman_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1D arrays."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("Input vectors must have the same length.")
    if x.size < 2:
        return 1.0
    if spearmanr is not None:
        corr, _ = spearmanr(x, y)
        return float(corr)

    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(corr if not np.isnan(corr) else 1.0)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if np.isclose(denom, 0.0):
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.dot(x, y) / denom)


def extract_attention_map(model: nn.Module, mel: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        mel = mel.to(device).unsqueeze(0)
        _, attention = model(mel)
        if attention is None:
            return np.zeros((64, 64), dtype=np.float32)
        attention = attention[0].detach().cpu()
        attention = attention.mean(dim=0, keepdim=True) if attention.dim() > 2 else attention.unsqueeze(0)
        attn = attention.squeeze(0)
        if attn.dim() == 2:
            return attn.numpy()
        if attn.dim() == 3:
            return attn.mean(dim=0).numpy()
        return attn.reshape(-1, 1).squeeze(-1).numpy()


def attention_stability_analysis(model: nn.Module, baseline_model: nn.Module, dataset: CoswaraDataset, snr_levels: Sequence[float] = (10, 5, 0), max_samples: int = 20):
    """Measure attention robustness under noisy conditions using Spearman and cosine similarity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    baseline_model.to(device)

    results: Dict[str, Dict[str, float]] = {}
    for snr in snr_levels:
        clean_spearman_attention = []
        clean_cosine_attention = []
        baseline_spearman_attention = []
        baseline_cosine_attention = []

        for idx in range(min(len(dataset), max_samples)):
            mel, label = dataset[idx]
            clean = mel.to(device)
            noisy = add_noise_to_mel(clean, snr).to(device)

            clean_attention = extract_attention_map(model, clean, device)
            noisy_attention = extract_attention_map(model, noisy, device)
            clean_spearman_attention.append(spearman_similarity(clean_attention.ravel(), noisy_attention.ravel()))
            clean_cosine_attention.append(cosine_similarity(clean_attention.ravel(), noisy_attention.ravel()))

            baseline_clean = extract_attention_map(baseline_model, clean, device)
            baseline_noisy = extract_attention_map(baseline_model, noisy, device)
            baseline_spearman_attention.append(spearman_similarity(baseline_clean.ravel(), baseline_noisy.ravel()))
            baseline_cosine_attention.append(cosine_similarity(baseline_clean.ravel(), baseline_noisy.ravel()))

        results[f"{int(snr)}dB"] = {
            "attention_model_spearman": float(np.mean(clean_spearman_attention)),
            "attention_model_cosine": float(np.mean(clean_cosine_attention)),
            "baseline_model_spearman": float(np.mean(baseline_spearman_attention)),
            "baseline_model_cosine": float(np.mean(baseline_cosine_attention)),
        }

    return results


def plot_stability_summary(results: Dict[str, Dict[str, float]], save_path: str = "attention_stability.png"):
    """Create a comparative plot of attention stability at different SNR levels."""
    snrs = [int(key.replace("dB", "")) for key in results.keys()]
    att_spear = [results[f"{snr}dB"]["attention_model_spearman"] for snr in snrs]
    att_cos = [results[f"{snr}dB"]["attention_model_cosine"] for snr in snrs]
    base_spear = [results[f"{snr}dB"]["baseline_model_spearman"] for snr in snrs]
    base_cos = [results[f"{snr}dB"]["baseline_model_cosine"] for snr in snrs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].plot(snrs, att_spear, marker="o", label="Attention model")
    axes[0].plot(snrs, base_spear, marker="s", label="Plain CNN")
    axes[0].set_title("Spearman correlation")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Correlation")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(snrs, att_cos, marker="o", label="Attention model")
    axes[1].plot(snrs, base_cos, marker="s", label="Plain CNN")
    axes[1].set_title("Cosine similarity")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Similarity")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_single_prediction(model: nn.Module, wav_path: str | Path, class_names: Sequence[str], device: Optional[torch.device] = None):
    """Predict class on a single .wav file and return label plus attention visualization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = mel_from_wav(wav_path)
    model.eval()
    with torch.no_grad():
        logits, attention = model(mel.to(device).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_label = class_names[pred_idx]
        confidence = float(probs[0, pred_idx].item())

    attention_map = attention[0].detach().cpu().numpy()
    label = f"Predicted: {pred_label} ({confidence:.3f})"
    fig = plot_attention_example(mel.cpu().numpy(), attention_map, label=label, save_path="single_prediction_attention.png")
    plt.close(fig)
    return {"predicted_label": pred_label, "confidence": confidence, "probabilities": probs[0].tolist()}


def run_full_pipeline(config: TrainingConfig):
    """End-to-end training + evaluation + attention analysis."""
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders(config)
    num_classes = len(class_names)

    attention_model = RespiroAttnNet(num_classes=num_classes)
    baseline_model = PlainCNN(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_model.to(device)
    baseline_model.to(device)

    optimizer = torch.optim.AdamW(attention_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}
    for epoch in range(config.num_epochs):
        attention_model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = attention_model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * inputs.size(0)

        val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(attention_model, val_loader, device)
        curr = val_metrics["macro_f1"] + 0.5 * val_metrics["auc"]
        print(f"Epoch {epoch + 1:02d} | train_loss={total_loss / len(train_loader.dataset):.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | val_auc={val_metrics['auc']:.4f}")
        if curr > best_score:
            best_score = curr
            best_state = {k: v.detach().cpu() for k, v in attention_model.state_dict().items()}
            best_metrics = val_metrics.copy()

    if best_state is None:
        raise RuntimeError("No valid model was saved during training.")

    attention_model.load_state_dict(best_state)
    save_path = Path(config.output_dir) / config.save_name
    torch.save({
        "model_state": best_state,
        "class_names": class_names,
        "config": asdict(config),
        "best_metrics": best_metrics,
    }, save_path)
    print(f"Best model saved to: {save_path}")

    val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(attention_model, val_loader, device)
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))
    print(confusion_matrix(true_labels, pred_labels))

    visualize_attention_examples(attention_model, val_loader, class_names, str(Path(config.output_dir) / "attention_examples"), max_examples=3)

    dataset = CoswaraDataset(
        data_dir=config.data_dir,
        sound_types=config.sound_types,
        sr=config.sr,
        n_mels=config.n_mels,
        max_duration=config.max_duration,
    )
    stability = attention_stability_analysis(attention_model, baseline_model, dataset, snr_levels=(10, 5, 0), max_samples=20)
    plot_stability_summary(stability, save_path=str(Path(config.output_dir) / "attention_stability.png"))
    print(json.dumps(stability, indent=2))

    return {
        "attention_model": attention_model,
        "baseline_model": baseline_model,
        "class_names": class_names,
        "metrics": val_metrics,
    }


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description="Explainable respiratory sound classification with attention on Coswara.")
    parser.add_argument("--data_dir", type=str, default="coswara-data", help="Path to the Coswara dataset directory.")
    parser.add_argument("--sound_types", type=str, default=",".join(DEFAULT_SOUND_TYPES), help="Comma-separated sound types, e.g. cough_heavy,breathing_deep,vowel_a")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--model_path", type=str, default=None, help="Optional path to a saved checkpoint for inference.")
    parser.add_argument("--wav_path", type=str, default=None, help="Single wav file for prediction.")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], default="train", help="Pipeline mode to run.")
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        sound_types=parse_sound_types(args.sound_types) or DEFAULT_SOUND_TYPES,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )

    if args.mode in {"train", "evaluate"}:
        train_result = run_full_pipeline(config)
        if args.mode == "evaluate":
            print("Evaluation complete.")

    if args.mode == "predict":
        if args.model_path is None or args.wav_path is None:
            raise ValueError("For single-sample prediction, pass both --model_path and --wav_path.")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        class_names = checkpoint["class_names"]
        model = RespiroAttnNet(num_classes=len(class_names))
        model.load_state_dict(checkpoint["model_state"])
        result = run_single_prediction(model, args.wav_path, class_names)
        print(json.dumps(result, indent=2))

    if args.mode == "train" and args.wav_path is not None and args.model_path is None:
        # A convenient way to run inference immediately after training on a single file.
        pass

    if args.wav_path and args.model_path:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        class_names = checkpoint["class_names"]
        model = RespiroAttnNet(num_classes=len(class_names))
        model.load_state_dict(checkpoint["model_state"])
        result = run_single_prediction(model, args.wav_path, class_names)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    entrypoint()
