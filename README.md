# Comparative ML/DL Approaches for Information Extraction in Real Estate

This project implements a comprehensive comparative study of Machine Learning and Deep Learning approaches for Information Extraction in Real Estate domain.

## Table of Contents


## Overview


## Installation

### Prerequisites


### Quick Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── .gitignore <-- Ignore: venv, pycache, .DS_Store, *.pth, data/
├── README.md <-- Overview, Installation, How to run UI
├── requirements.txt <-- Dependencies (streamlit, transformers, sklearn-crfsuite, etc.)
├── main.py <-- CLI Entry point
│
├── configs/
│ ├── ml_config.yaml <-- Params: Window size, L1/L2 reg for CRF, C for SVM
│ └── dl_config.yaml <-- Params: Model ID (PhoBERT), LR, Epochs, Max_len
│
├── data/
│ ├── 01_raw/
│ ├── 02_intermediate/
│ ├── 03_primary/
│ └── 04_model_input/
│
├── docs/
│ └── annotation_guideline.pdf <-- Create an empty placeholder file
│
├── notebooks/
│ ├── 01_EDA_Data_Analysis.ipynb
│ ├── 02_Exp_Machine_Learning.ipynb
│ ├── 03_Exp_Deep_Learning.ipynb
│ └── 04_Benchmark_Comparison.ipynb
│
├── ui/
│ ├── app.py
│ └── components.py
│
├── outputs/
│ ├── checkpoints/
│ ├── logs/
│ └── predictions/
│
└── src/
├── init.py
├── features/
│ ├── init.py
│ ├── hand_crafted.py
│ └── embeddings.py
│
├── data_loader/
│ ├── dataset.py
│ └── preprocess.py
│
├── models/
│ ├── init.py
│ ├── conventional.py
│ └── deep_learning.py
│
└── utils/
    ├── metrics.py
    ├── visualization.py
    └── post_process.py
```

## How to Run

### CLI
```bash
python main.py train --model both
python main.py evaluate --model both --checkpoint outputs/checkpoints/best.pth
python main.py predict --model dl --checkpoint model.pth --input test.txt
```

### Web UI
```bash
streamlit run ui/app.py
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Models

### Machine Learning
- **CRF** - Conditional Random Field with hand-crafted features
- **SVM** - Support Vector Machine  
- **Logistic Regression** - Linear baseline

### Deep Learning
- **PhoBERT** - Vietnamese BERT (state-of-the-art)
- **mBERT** - Multilingual BERT
- **XLM-R** - Cross-lingual RoBERTa

## 📊 Performance Results



**Key Findings:**


## License

MIT License

## Contributing


---

