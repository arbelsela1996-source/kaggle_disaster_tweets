# Disaster Tweet Classifier (BERT)

This is a small, beginner-friendly NLP project that fine-tunes a pretrained BERT model (`bert-base-uncased`) on the Kaggle **“NLP with Disaster Tweets”** dataset to classify whether a tweet describes a real disaster (`1`) or not (`0`).

The project is designed to be easy to read and extend, and it mirrors a Kaggle Notebook version you can run with GPU acceleration.

## Project structure

- `requirements.txt` – Python dependencies.
- `scripts/create_venv.ps1` – Creates `.venv` when Python is installed (Windows).
- `train_disaster_bert.py` – Main training script (local entry point).
- `src/`
  - `config.py` – Configuration (paths, hyperparameters, model name).
  - `data.py` – Load and split the Kaggle CSVs.
  - `datasets.py` – PyTorch dataset for tweets with BERT tokenization.
  - `modeling.py` – Helpers to create tokenizer and BERT classifier.
  - `train_utils.py` – Training and evaluation loops with metrics.
  - `plots.py` – Simple plotting utilities (loss/metrics/confusion matrix).
  - `predict.py` – Helper for running inference on new tweets.

You can also create a Kaggle Notebook (`kaggle_disaster_bert.ipynb`) that mirrors the same pipeline.

## Git

This folder is a **Git** repository (default branch **`main`**). [Install Git](https://git-scm.com/download/win) if needed, then from the project root:

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

On Kaggle you can `git clone` that URL into `/kaggle/working` and `git pull` for updates.

## Setup (local)

**If `python` is not recognized (Windows):** install [Python 3.10+](https://www.python.org/downloads/) and tick **“Add python.exe to PATH”**, then open a **new** PowerShell window. Alternatively run `scripts\create_venv.ps1` (see below).

1. Create and activate a virtual environment (recommended).

   On **Windows (PowerShell)** — after Python is on PATH:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Or run the helper script from the project folder:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\create_venv.ps1
   ```

   On **macOS / Linux**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies inside the venv:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the **“NLP with Disaster Tweets”** dataset from Kaggle and place the CSV files in a `data/` folder at the project root:

- `data/train.csv`
- `data/test.csv`

4. Run training:

```bash
python train_disaster_bert.py
```

This will:

- Fine-tune `bert-base-uncased` on the training set.
- Evaluate on a validation split.
- Print per-epoch metrics (loss, accuracy, F1).
- Save plots and the best model to an `outputs/` folder.

## Running on Kaggle

- Create a new Kaggle Notebook attached to the **“NLP with Disaster Tweets”** competition.
- Copy the core code from this repo (or use the provided `kaggle_disaster_bert.ipynb` once created).
- Enable a GPU in the Notebook settings.
- Run cells in order to train and evaluate the model.

This project focuses on clarity over cleverness, so it’s a good starting point to learn how BERT-based text classification pipelines are built.

