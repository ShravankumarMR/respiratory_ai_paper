# Respiratory AI Project

Production-style respiratory sound classification pipeline for the Coswara dataset, built for local execution on a personal machine.

## Project overview

This project does the following:

- loads Coswara respiratory audio files
- normalizes and filters sound types
- converts each audio clip into a log Mel-spectrogram
- trains a CNN with CBAM attention
- compares it against a plain CNN baseline
- evaluates using accuracy, macro-F1, and AUC
- saves the best model checkpoint
- visualizes attention maps
- supports single-sample inference from raw .wav files

The actual model logic remains the same as the original pipeline; this project just reorganizes it into a cleaner local workflow.

---

## 1. Prerequisites

You need:

- Python 3.10+
- pip
- a local copy of the Coswara dataset in the project folder

Your folder should look like this:

```text
respiratory_ai_paper/
├── coswara-data/
├── respiratory_ai/
├── tests/
├── config/
├── notebooks/
├── run_training.py
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

## 2. Create a virtual environment

From the project root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

From the project root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer a package-style install:

```powershell
python -m pip install -e .
```

If your virtual environment is not activated, use the venv interpreter directly:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 4. Verify the project structure

Check that the dataset is available:

```powershell
Get-ChildItem .\coswara-data
```

You should see the Coswara data directories and audio files.

---

## 5. Run the EDA notebook

Open the notebook:

```text
notebooks/eda.ipynb
```

This notebook helps you:

- inspect the audio inventory
- review file counts and extensions
- check class balance
- examine sample Mel-spectrogram structure
- understand how the dataset is prepared before training

---

## 6. Train the model

Run:

```powershell
python run_training.py --data_dir coswara-data --sound_types cough_heavy breathing_deep vowel_a --epochs 20 --output_dir artifacts
```

If the environment is not active, use:

```powershell
& .\.venv\Scripts\python.exe  run_training.py --data_dir coswara-data --sound_types breathing_deep breathing_shallow cough_heavy cough_shallow vowel_a vowel_i vowel_o counting_normal counting_fast speech voice --epochs 20 --output_dir artifacts
```

This command will:

- load the selected Coswara audio types
- split data into train/validation sets
- train the CNN + CBAM attention model
- save the best checkpoint as `artifacts/respiro_attn_best.pth`
- print accuracy, macro-F1, and AUC results
- print the classification report and confusion matrix

---

## 7. Check the saved model

After training, you should see:

```text
artifacts/
├── respiro_attn_best.pth
├── attention_examples/
├── attention_stability.png
├── class_distribution.png
└── audio_manifest.csv
```

---

## 8. Run prediction on a single audio file

Use a raw .wav file:

```powershell
python run_training.py --mode predict --model_path artifacts/respiro_attn_best.pth --wav_path .\path\to\sample.wav
```

If the environment is not active, use:

```powershell
& .\.venv\Scripts\python.exe run_training.py --mode predict --model_path artifacts/respiro_attn_best.pth --wav_path .\path\to\sample.wav
```

This prints:

- predicted class
- confidence score
- class probabilities
- a saved attention visualization image

---

## 9. Reuse the config file

The project also includes:

```text
config/training_config.yaml
```

This file stores the default training options such as:

- dataset path
- selected sound types
- sample rate
- Mel settings
- batch size
- epochs
- learning rate
- output directory

You can edit this file as needed before running training.

---

## 10. Useful troubleshooting

### If Python cannot find packages

```powershell
python -m pip install -r requirements.txt
```

### If you want to verify the project tests

```powershell
python -m pytest -q
```

If the virtual environment is not active, use:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

### If the dataset is not found

Check that `coswara-data` exists in the project root:

```powershell
Get-ChildItem .\coswara-data
```

### If a command fails due to PowerShell path issues

Use the call operator for Python explicitly:

```powershell
& .\.venv\Scripts\python.exe run_training.py --help
```

---

## 11. Notes

- This project keeps compatibility with the original functionality in the original pipeline and dataset scripts.
- The training pipeline is intentionally organized for easier local experimentation and reproducibility.
- No core logic was changed during the project cleanup; only the project structure and execution flow were improved for local use.
# respiratory_ai_paper
