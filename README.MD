# Protein Secondary Structure Prediction using Deep Learning

A deep learning project that predicts **protein secondary structure** (Q3 and Q8 classifications) from amino acid sequences using three custom architectures — Bi-RNN, Bi-GRU, and CNN-LSTM. Built with PyTorch for the SEP-25 DL Gen AI NPPE-2 Kaggle competition.

---

## Overview

Protein secondary structure prediction is a fundamental bioinformatics task — given a sequence of amino acids, predict the structural class of each residue. This project implements and compares three sequence-to-sequence deep learning models that predict both Q8 (8-class) and Q3 (3-class: Helix, Sheet, Coil) structures simultaneously.

---

## Dataset

- **Source:** [SEP-25 DL Gen AI NPPE-2](https://www.kaggle.com/competitions/sep-25-dl-gen-ai-nppe-2)
- **Input:** Amino acid sequences (20 standard amino acids)
- **Target:** Per-residue structure labels — Q8 (H, G, I, E, B, T, S, C) and Q3 (H, E, C)
- **Split:** 90/10 train-validation (`random_state=42`)

---

## Models

### 1. Bi-RNN (Bidirectional RNN)

| Component | Details |
|-----------|---------|
| Embedding | 128-dim, with PAD + UNK tokens |
| RNN | 3-layer Bidirectional RNN, hidden=256 |
| Regularization | Dropout 0.3, LayerNorm |
| Head | Residual feedforward → Q3 (3) + Q8 (8) outputs |
| Kaggle Score | ~0.482 |

### 2. Bi-GRU (Bidirectional GRU)

| Component | Details |
|-----------|---------|
| Embedding | 128-dim |
| GRU | 3-layer Bidirectional GRU, hidden=256 |
| Regularization | Dropout 0.3/0.4, LayerNorm |
| Head | Residual feedforward → Q3 + Q8 outputs |
| Kaggle Score | ~0.482 |

### 3. CNN-LSTM (Best Model)

| Component | Details |
|-----------|---------|
| Embedding | 128-dim |
| CNN | Multi-scale convolutions (kernel 1, 3, 5, 7) — captures local motifs |
| LSTM | Bidirectional LSTM, hidden=256 — captures long-range dependencies |
| Regularization | Dropout 0.3 |
| Head | Q3 (3) + Q8 (8) dual outputs |
| Kaggle Score | **~0.487** |

> CNN-LSTM outperforms pure recurrent models by combining local pattern extraction (CNN) with sequential context (LSTM).

---

## Training Details

| Parameter | Bi-RNN | Bi-GRU | CNN-LSTM |
|-----------|--------|--------|----------|
| Optimizer | Adam | Adam | AdamW |
| Learning Rate | 0.001 | 0.0005 | 0.001 |
| Weight Decay | — | — | 1e-4 |
| Epochs | 10 | 12 | 10 |
| Batch Size | 16 | 16 | 16 |
| LR Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Loss | CrossEntropyLoss (ignore padding) | CrossEntropyLoss | CrossEntropyLoss |

---

## Results

| Model | Kaggle Score |
|-------|-------------|
| Best on leaderboard | **0.51** |
| **CNN-LSTM (this project)** | **0.49** |
| Bi-GRU | 0.482 |
| Bi-RNN | 0.482 |

> Achieved **0.49** with a custom CNN-LSTM architecture — only 2% behind the top score of 0.51. No pretrained protein language models (like ESM or ProtTrans) were used.

---

## Pretrained Weights

Model weights are hosted on Kaggle Model Hub (too large for GitHub):

| Model | Download Link |
|-------|---------------|
| CNN-LSTM (best) | [protein-cnn-lsm](https://www.kaggle.com/models/jatinnahatajain/protein-cnn-lsm) |
| Bi-RNN + Bi-GRU | [protein-rnn-gru](https://www.kaggle.com/models/jatinnahatajain/protein-rnn-gru) |

To load in code:
```python
import kagglehub
kagglehub.model_download("jatinnahatajain/protein-cnn-lsm/pytorch/v1")
kagglehub.model_download("jatinnahatajain/protein-rnn-gru/pytorch/baseline")
```

---

## Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook
└── README.md               # This file
```

---

## How to Run

1. **On Kaggle / Google Colab (recommended):**
   - Open the [notebook on Kaggle](https://www.kaggle.com/) and click **"Copy & Edit"**
   - Or upload `notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
   - Enable **GPU accelerator** in notebook settings
   - Run all cells

2. **Locally:**
   ```bash
   pip install torch pandas numpy scikit-learn kagglehub
   jupyter notebook notebook.ipynb
   ```

---

## Tech Stack

- Python 3
- PyTorch (models, training, inference)
- Scikit-learn (train/test split, F1 score)
- Pandas / NumPy
- KaggleHub (model upload/download)
- Trackio (experiment tracking)

---

## Key Features

- Three custom architectures compared: Bi-RNN, Bi-GRU, CNN-LSTM
- Multi-scale CNN (kernels 1, 3, 5, 7) captures local amino acid motifs at different scales
- Dual-head prediction: Q3 and Q8 structure simultaneously
- Q8 → Q3 label merging (H/G/I → Helix, E/B → Sheet, T/S/C → Coil)
- Dynamic padding with custom collate function for variable-length sequences
- CosineAnnealingLR for smooth learning rate decay
- Model versioning via Kaggle Model Hub
- GPU-accelerated training

---

## Author

**Jatin Nahata Jain** — [Kaggle Profile](https://www.kaggle.com/jatinnahatajain)

---