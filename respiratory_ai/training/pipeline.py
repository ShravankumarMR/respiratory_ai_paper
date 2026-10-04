from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from respiratory_ai.config import AppConfig
from respiratory_ai.data.dataset import CoswaraDataset
from respiratory_ai.training.models import PlainCNN, RespiroAttnNet, resize_attention


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mel_from_wav(wav_path: str | Path, sr: int = 16000, n_mels: int = 64, max_duration: float = 5.0) -> torch.Tensor:
    import wave

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


def create_dataloaders(config: AppConfig):
    dataset = CoswaraDataset(
        data_dir=config.data_dir,
        sound_types=config.sound_types,
        sr=config.sr,
        n_mels=config.n_mels,
        max_duration=config.max_duration,
        vad_trim=config.vad_trim,
    )
    if len(dataset) == 0:
        raise ValueError(f"No labeled audio files found in '{config.data_dir}'.")

    labels = np.array([label for _, label in dataset.file_records], dtype=np.int64)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=config.test_size,
        stratify=labels,
        random_state=config.random_seed,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    sampler = None
    if config.use_weighted_sampler:
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=len(dataset.class_names)).astype(np.float64)
        class_counts[class_counts == 0.0] = 1.0
        per_class_weight = 1.0 / class_counts
        sample_weights = per_class_weight[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, dataset.class_names


def accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if y_score.ndim == 1 or y_score.shape[1] == 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        present_classes = np.unique(y_true)
        if present_classes.size == 0:
            auc = 0.0
        else:
            class_indices = np.arange(y_score.shape[1])
            if present_classes.size < y_score.shape[1]:
                allowed = class_indices[np.isin(class_indices, present_classes)]
                subset = y_score[:, allowed]
                row_sums = subset.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 0):
                    subset = subset / np.clip(row_sums, 1e-12, None)
                else:
                    subset = subset / row_sums
                try:
                    auc = roc_auc_score(y_true, subset, multi_class="ovr", average="macro", labels=allowed)
                except ValueError:
                    auc = 0.0
            else:
                auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "auc": float(auc)}


def filter_labels_for_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]):
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    if present_labels.size == 0:
        return np.array([], dtype=int), []

    labels = present_labels.astype(int)
    names = [class_names[int(label)] for label in labels if int(label) < len(class_names)]
    return labels, names


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, _ = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_true.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(all_true)
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits.argmax(axis=1)
    # roc_auc_score needs class probabilities (rows summing to 1), not raw logits.
    y_score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = accuracy_and_f1(y_true, y_pred, y_score)
    return metrics, y_true, y_pred, y_score


def build_spec_augment(n_mels: int) -> nn.Module:
    """Time/frequency masking applied to log-mel batches during training only."""
    return nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=max(1, n_mels // 8)),
        torchaudio.transforms.TimeMasking(time_mask_param=30),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    spec_augment: nn.Module | None = None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if spec_augment is not None:
            inputs = spec_augment(inputs)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train_and_evaluate(config: AppConfig):
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders(config)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RespiroAttnNet(num_classes=num_classes, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = max(1, config.num_epochs * len(train_loader))
    warmup_steps = min(total_steps - 1, max(0, config.warmup_epochs * len(train_loader))) if total_steps > 1 else 0
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Upweight classes that are hardest to tell apart: cough_heavy vs cough_shallow, and
    # counting_normal (under-predicted) vs counting_fast (over-predicted).
    def _class_weight(name: str) -> float:
        if "cough" in name:
            return config.cough_class_weight
        if name == "counting_normal":
            return config.counting_class_weight
        return 1.0

    class_weights = torch.tensor(
        [_class_weight(name) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    spec_augment = build_spec_augment(config.n_mels).to(device) if config.use_specaugment else None

    best_score = -1.0
    best_state = None
    best_metrics = {}
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion, spec_augment)
        val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(model, val_loader, device)
        current_score = val_metrics["macro_f1"] + 0.5 * val_metrics["auc"]
        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | val_auc={val_metrics['auc']:.4f}")
        if current_score > best_score:
            best_score = current_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs (patience={config.early_stopping_patience}).")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    save_path = output_dir / config.save_name
    torch.save({
        "model_state": best_state,
        "class_names": class_names,
        "config": config.as_dict(),
        "best_metrics": best_metrics,
    }, save_path)

    val_metrics, true_labels, pred_labels, prob_scores = evaluate_model(model, val_loader, device)
    report_labels, report_names = filter_labels_for_report(true_labels, pred_labels, class_names)
    print(classification_report(true_labels, pred_labels, labels=report_labels, target_names=report_names, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels, labels=report_labels))

    plot_confusion_matrix_image(true_labels, pred_labels, report_names, str(output_dir / "confusion_matrix.png"), labels=report_labels)
    plot_classification_report(true_labels, pred_labels, report_names, str(output_dir / "classification_report.png"), labels=report_labels)
    visualize_attention_examples(model, val_loader, class_names, str(output_dir / "attention_examples"), device=device)
    visualize_misclassified_examples(
        model,
        val_loader,
        class_names,
        str(output_dir / "attention_examples" / "misclassified"),
        focus_classes=[name for name in class_names if "cough" in name],
        device=device,
    )
    plot_attention_stability(model, val_loader, str(output_dir), device=device)

    return {
        "model": model,
        "class_names": class_names,
        "metrics": val_metrics,
        "path": str(save_path),
    }


def plot_confusion_matrix_image(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], save_path: str, labels: np.ndarray | None = None) -> None:
    labels = labels if labels is not None else np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], save_path: str, labels: np.ndarray | None = None) -> None:
    labels = labels if labels is not None else np.arange(len(class_names))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)

    row_labels = list(class_names) + ["accuracy", "macro avg", "weighted avg"]
    col_labels = ["precision", "recall", "f1-score", "support"]
    cell_text = []
    for row in row_labels:
        if row == "accuracy":
            support = int(report["macro avg"]["support"])
            cell_text.append(["", "", f"{report['accuracy']:.4f}", f"{support}"])
        else:
            metrics = report[row]
            cell_text.append([
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                f"{int(metrics['support'])}",
            ])

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(row_labels) + 1.5))
    ax.axis("off")
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Classification Report", pad=20)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_stability(model: nn.Module, loader: DataLoader, output_dir: str, device: torch.device | None = None, max_examples: int = 50) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    maps = []
    target_size = None

    with torch.no_grad():
        for inputs, _ in loader:
            if len(maps) >= max_examples:
                break
            inputs = inputs.to(device)
            _, attention = model(inputs)
            if target_size is None:
                target_size = (inputs.shape[2], inputs.shape[3])
            for idx in range(inputs.size(0)):
                if len(maps) >= max_examples:
                    break
                attn = resize_attention(attention[idx].detach().cpu().unsqueeze(0), target_size)
                attn = attn.numpy() if hasattr(attn, "numpy") else attn
                maps.append(np.squeeze(attn))

    if not maps:
        return

    stacked = np.stack(maps, axis=0)
    mean_map = stacked.mean(axis=0)
    std_map = stacked.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(mean_map, cmap="jet", aspect="auto", origin="lower")
    axes[0].set_title("Mean attention map")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(std_map, cmap="viridis", aspect="auto", origin="lower")
    axes[1].set_title("Attention stability (std across samples)")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    out = Path(output_dir) / "attention_stability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_example(mel: np.ndarray, attention_map: np.ndarray, label: str | None = None, save_path: str | None = None):
    mel = mel.squeeze(0) if mel.ndim == 3 else mel
    attention_map = attention_map.squeeze() if attention_map.ndim > 2 else attention_map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

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
    ax.imshow(mel, cmap="magma", aspect="auto", origin="lower")
    ax.imshow(attention_map, cmap="jet", alpha=0.45, aspect="auto", origin="lower")
    ax.set_title("Overlay")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    if label is not None:
        ax.text(0.02, 0.98, f"Class: {label}", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")

    return fig


def visualize_attention_examples(model: nn.Module, loader: DataLoader, class_names: Sequence[str], output_dir: str, max_examples: int = 3, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    counter = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, attention = model(inputs)
            for idx in range(inputs.size(0)):
                if counter >= max_examples:
                    return
                mel = inputs[idx].detach().cpu().numpy()
                attn = attention[idx].detach().cpu().numpy()
                attn = resize_attention(torch.tensor(attn).unsqueeze(0), (mel.shape[1], mel.shape[2]))
                attn = attn.numpy() if hasattr(attn, "numpy") else attn
                label_name = class_names[int(labels[idx].item())]
                fig = plot_attention_example(mel=mel, attention_map=attn, label=label_name, save_path=str(out_dir / f"attention_example_{counter + 1}.png"))
                plt.close(fig)
                counter += 1


def visualize_misclassified_examples(
    model: nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    output_dir: str,
    focus_classes: Sequence[str],
    max_examples: int = 12,
    device: torch.device | None = None,
) -> None:
    """Save attention maps for validation samples of focus_classes that the model got wrong."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focus_idx = {class_names.index(name) for name in focus_classes if name in class_names}
    if not focus_idx:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    counter = 0
    with torch.no_grad():
        for inputs, labels in loader:
            if counter >= max_examples:
                return
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, attention = model(inputs)
            preds = logits.argmax(dim=1)
            for idx in range(inputs.size(0)):
                if counter >= max_examples:
                    return
                true_idx = int(labels[idx].item())
                pred_idx = int(preds[idx].item())
                if true_idx not in focus_idx or pred_idx == true_idx:
                    continue
                mel = inputs[idx].detach().cpu().numpy()
                attn = attention[idx].detach().cpu().numpy()
                attn = resize_attention(torch.tensor(attn).unsqueeze(0), (mel.shape[1], mel.shape[2]))
                attn = attn.numpy() if hasattr(attn, "numpy") else attn
                label_name = f"true={class_names[true_idx]} pred={class_names[pred_idx]}"
                fig = plot_attention_example(mel=mel, attention_map=attn, label=label_name, save_path=str(out_dir / f"misclassified_{counter + 1}.png"))
                plt.close(fig)
                counter += 1


def predict_single_file(model_path: str | Path, wav_path: str | Path, config: AppConfig):
    checkpoint = torch.load(str(model_path), map_location="cpu")
    class_names = checkpoint["class_names"]
    model = RespiroAttnNet(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mel = mel_from_wav(wav_path, sr=config.sr, n_mels=config.n_mels, max_duration=config.max_duration)
    with torch.no_grad():
        logits, attention = model(mel.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_label = class_names[pred_idx]
        confidence = float(probs[0, pred_idx].item())

    output = {"predicted_label": pred_label, "confidence": confidence, "probabilities": probs[0].tolist()}
    print(json.dumps(output, indent=2))

    attn = attention[0].detach().cpu().numpy()
    fig = plot_attention_example(mel=mel.squeeze(0).numpy(), attention_map=attn, label=pred_label, save_path=str(Path(config.output_dir) / "single_prediction_attention.png"))
    plt.close(fig)
    return output
