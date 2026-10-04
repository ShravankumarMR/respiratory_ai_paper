#!/usr/bin/env python3
"""Production entrypoint for training the respiratory classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from respiratory_ai.training.pipeline import train_and_evaluate, predict_single_file
from respiratory_ai.config import AppConfig, DEFAULT_SOUND_TYPES


def parse_sound_types(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    sound_types = [item.strip() for value in values for item in value.split(",") if item.strip()]
    return sound_types or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run inference for a respiratory sound attention model.")
    parser.add_argument("--data_dir", type=str, default="coswara-data", help="Coswara dataset folder")
    parser.add_argument(
        "--sound_types",
        nargs="+",
        default=DEFAULT_SOUND_TYPES,
        help="Comma-separated sound types; spaces after commas are allowed",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--vad_trim", type=lambda v: v.lower() != "false", default=True, help="Trim leading/trailing silence before framing (true/false)")
    parser.add_argument("--use_specaugment", type=lambda v: v.lower() != "false", default=True, help="Apply SpecAugment during training (true/false)")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cough_class_weight", type=float, default=1.5, help="Loss weight multiplier for cough_heavy/cough_shallow classes")
    parser.add_argument("--counting_class_weight", type=float, default=1.3, help="Loss weight multiplier for counting_normal to offset over-prediction of counting_fast")
    parser.add_argument("--use_weighted_sampler", type=lambda v: v.lower() != "false", default=True, help="Use class-balanced sampling in each training epoch (true/false)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear learning-rate warmup epochs before cosine annealing")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Stop if validation score does not improve for N epochs")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability before the final classifier layer")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--wav_path", type=str, default=None)
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()

    config = AppConfig(
        data_dir=args.data_dir,
        sound_types=parse_sound_types(args.sound_types) or DEFAULT_SOUND_TYPES,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        output_dir=args.output_dir,
        vad_trim=args.vad_trim,
        use_specaugment=args.use_specaugment,
        label_smoothing=args.label_smoothing,
        cough_class_weight=args.cough_class_weight,
        counting_class_weight=args.counting_class_weight,
        use_weighted_sampler=args.use_weighted_sampler,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
    )

    if args.mode == "train":
        train_and_evaluate(config)
        return

    if args.mode == "predict":
        if args.model_path is None or args.wav_path is None:
            raise ValueError("For prediction, provide --model_path and --wav_path.")
        predict_single_file(Path(args.model_path), Path(args.wav_path), config)


if __name__ == "__main__":
    main()
