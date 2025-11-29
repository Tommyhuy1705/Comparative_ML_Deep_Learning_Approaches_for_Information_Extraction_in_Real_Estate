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
vn-real-estate-ie/          <-- Root Directory
│
├── .gitignore              <-- Ignore: venv, __pycache__, .DS_Store, *.pth, data/
├── README.md               <-- Overview, Installation, How to run UI
├── requirements.txt        <-- Dependencies (streamlit, transformers, sklearn-crfsuite, etc.)
├── main.py                 <-- CLI Entry point (để chạy train full pipeline trên server/terminal)
│
├── configs/                <-- Configuration files
│   ├── ml_config.yaml      <-- Params: Window size, L1/L2 reg for CRF, C for SVM
│   └── dl_config.yaml      <-- Params: Model ID (PhoBERT), LR, Epochs, Max_len
│
├── data/                   <-- Data Pipeline (Không push file nặng lên git)
│   ├── 01_raw/             <-- Raw text files
│   ├── 02_intermediate/    <-- Cleaned, Tokenized (VnCoreNLP/Underthesea)
│   ├── 03_primary/         <-- Golden Dataset (JSONL/CoNLL) đã gán nhãn & review
│   └── 04_model_input/     <-- Train/Dev/Test splits
│
├── docs/                   <-- Documentation
│   └── annotation_guideline.pdf  <-- TÀI LIỆU DUY NHẤT & QUAN TRỌNG NHẤT
│
├── notebooks/              <-- TRUNG TÂM THỰC NGHIỆM (Experimental & Visualization)
│   ├── 01_EDA_Data_Analysis.ipynb       <-- Thống kê phân bố nhãn, độ dài câu, từ vựng
│   ├── 02_Exp_Machine_Learning.ipynb    <-- Train & Tune CRF, SVM, MEMM (GridSearch)
│   ├── 03_Exp_Deep_Learning.ipynb       <-- Train Loop cho PhoBERT/BiLSTM (Loss chart)
│   └── 04_Benchmark_Comparison.ipynb    <-- Load model saved -> Chạy trên Test set -> Vẽ biểu đồ so sánh ML vs DL -> Error Analysis
│
├── ui/                     <-- USER INTERFACE (Web App Demo)
│   ├── app.py              <-- Streamlit/Gradio App (Main UI logic)
│   └── components.py       <-- Các hàm render kết quả (Highlight Entity màu sắc)
│
├── outputs/                <-- Artifacts
│   ├── checkpoints/        <-- Model weights (.pth, .pkl)
│   ├── logs/               <-- Training logs
│   └── predictions/        <-- Kết quả predict ra file .txt/.json để notebook 04 đọc vào
│
└── src/                    <-- Core Logic (Import vào Notebooks để dùng lại)
    ├── __init__.py
    ├── features/
    │   ├── __init__.py
    │   ├── hand_crafted.py <-- Hàm `get_features(sent, i)` cho ML
    │   └── embeddings.py   <-- Hàm `get_bert_embedding(text)` cho Visualization
    │
    ├── data_loader/
    │   ├── dataset.py      <-- Pytorch Dataset Class
    │   └── preprocess.py   <-- Text normalization
    │
    ├── models/
    │   ├── __init__.py
    │   ├── conventional.py <-- Wrapper class cho ML models
    │   └── deep_learning.py <-- Custom Module (PhoBERT + Linear/CRF head)
    │
    └── utils/
        ├── metrics.py      <-- Hàm tính Span-level F1 (dùng seqeval)
        ├── visualization.py <-- Hàm vẽ Confusion Matrix, t-SNE
        └── post_process.py <-- Xử lý output model ra format hiển thị UI
```

## How to Run

### CLI
```bash

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

## Performance Results



**Key Findings:**


## License

MIT License

## Contributing


---

