import os
import glob
import joblib
import kagglehub
from transformers import AutoModel, RobertaTokenizerFast, AutoModelForTokenClassification
import streamlit as st
import numpy as np
import torch

# Kaggle Model Handles
KAGGLE_ML_HANDLE_NER = "donhuyn/ml-ner-er-vietnamese/scikitLearn/ner"
KAGGLE_ML_HANDLE_RE = "donhuyn/ml-ner-er-vietnamese/scikitLearn/re"
KAGGLE_DL_HANDLE_NER = "donhuyn/phobert-ner-er-vietnamese/pyTorch/ner"
KAGGLE_DL_HANDLE_RE = "donhuyn/phobert-ner-er-vietnamese/pyTorch/re"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PHOBERT_NAME = "vinai/phobert-base-v2"
DL_tokenizer = RobertaTokenizerFast.from_pretrained(PHOBERT_NAME, add_prefix_space=True)
phobert_base_model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
phobert_base_model.eval()

RE_MODEL_FILE_MAP = {
    "PCA": "re_pca.pkl",
    "MaxEnt": "re_maxent.pkl",
    "SVM": "re_svm.pkl",
    "Scaler": "re_scaler.pkl",
    "RandomForest": "re_randomforest.pkl"
}

NER_MODEL_FILE_MAP = {
    "CRF": "ner_crf.pkl",
    "MaxEnt": "ner_maxent.pkl",
    "SVM": "ner_svm.pkl",
}

# ================================
# Các hàm load model từ Kaggle Hub
# ================================
@st.cache_resource
def ner_load_ml_model_from_kaggle(model_type):
    """Load model ML NER từ Kaggle Hub"""
    try:
        with st.spinner(f"Đang tải ML model từ {KAGGLE_ML_HANDLE_NER}..."):
            path = kagglehub.model_download(KAGGLE_ML_HANDLE_NER)

        file_name = NER_MODEL_FILE_MAP.get(model_type)
        if not file_name:
            raise ValueError(f"Model không hỗ trợ: {model_type}")

        model_path = find_file_by_pattern(path, file_name)
        if not model_path:
            raise FileNotFoundError(f"Không tìm thấy {file_name}")

        model = joblib.load(model_path)
        st.toast(f"Loaded NER-ML model: {model_type}")
        return model

    except Exception as e:
        st.error(f"Lỗi load RE-ML model: {e}")
        return None

@st.cache_resource
def re_load_ml_model_from_kaggle(model_type):
    """Load model ML RE từ Kaggle Hub"""
    try:
        with st.spinner(f"Đang tải ML model từ {KAGGLE_ML_HANDLE_RE}..."):
            path = kagglehub.model_download(KAGGLE_ML_HANDLE_RE)

        file_name = RE_MODEL_FILE_MAP.get(model_type)
        if not file_name:
            raise ValueError(f"Model không hỗ trợ: {model_type}")

        model_path = find_file_by_pattern(path, file_name)
        if not model_path:
            raise FileNotFoundError(f"Không tìm thấy {file_name}")

        model = joblib.load(model_path)
        st.toast(f"Loaded RE-ML model: {model_type}")
        return model

    except Exception as e:
        st.error(f"Lỗi load RE-ML model: {e}")
        return None

@st.cache_resource
def ner_load_dl_model_from_kaggle():
    """Load PhoBERT NER model từ Kaggle Hub"""
    try:
        with st.spinner(f"Đang tải DL model từ {KAGGLE_DL_HANDLE_NER}..."):
            path = kagglehub.model_download(KAGGLE_DL_HANDLE_NER)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer từ CHÍNH thư mục model
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Load model NER (KHÔNG torch.load)
        model = AutoModelForTokenClassification.from_pretrained(path)
        model.to(device)
        model.eval()

        # Lấy label mapping từ config (chuẩn HF)
        id2label = model.config.id2label

        st.toast("Đã load PhoBERT NER model")
        return model, tokenizer, id2label

    except Exception as e:
        st.error(f"Lỗi load DL model: {e}")
        return None, None, None

@st.cache_resource
def re_load_dl_model_from_kaggle():
    """Load PhoBERT RE model từ Kaggle Hub"""
    try:
        with st.spinner(f"Đang tải DL model từ {KAGGLE_DL_HANDLE_RE}..."):
            path = kagglehub.model_download(KAGGLE_DL_HANDLE_RE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer từ CHÍNH thư mục model
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Load model NER (KHÔNG torch.load)
        model = AutoModelForTokenClassification.from_pretrained(path)
        model.to(device)
        model.eval()

        # Lấy label mapping từ config (chuẩn HF)
        id2label = model.config.id2label

        st.toast("Đã load PhoBERT RE model")
        return model, tokenizer, id2label

    except Exception as e:
        st.error(f"Lỗi load DL model: {e}")
        return None, None, None

# =======================
# Các hàm run NER/RE/IE
# =======================
def run_ner(tokens, cleaned_text, model_type, model_name=None):
    entities = []

    if model_type == "Machine Learning":
        model = ner_load_ml_model_from_kaggle(model_name)
        if model_name == "CRF":
            features = ner_extract_features(
                cleaned_text=cleaned_text,
                tokens=tokens,
                tokenizer=DL_tokenizer,
                phobert_model=phobert_base_model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                feature_type="crf"
            )
            preds = model.predict([features])[0]

            for token, label in zip(tokens, preds):
                entities.append({"text": token, "label": label})

        else:
            id2label={0: "B-AREA", 1: "B-ATTR", 2: "B-LOC", 3: "B-O", 4: "B-ORG", 5: "B-PER", 6: "B-PRICE", 7: "B-TYPE", 8: "I-AREA",
            9: "I-ATTR", 10: "I-LOC", 11: "I-O", 12: "I-ORG", 13: "I-PER", 14: "I-PRICE", 15: "I-TYPE", 16: "O"}    
                    
            features = ner_extract_features(
                cleaned_text=cleaned_text,
                tokens=tokens,
                tokenizer=DL_tokenizer,
                phobert_model=phobert_base_model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                feature_type="other"
            )
            preds = model.predict(features)
            pred_labels = [id2label[pid] for pid in preds]

            for token, label in zip(tokens, pred_labels):
                entities.append({"text": token, "label": label})

        return entities

    else:
        model, tokenizer, id2label = ner_load_dl_model_from_kaggle()

        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)[0].tolist()

        subtokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [id2label[p] for p in preds]

        # Gom BIO
        entities = []
        current = []
        current_label = None

        for tok, lab in zip(subtokens, labels):
            if tok in ["<s>", "</s>", "<pad>"]:
                continue

            tok = (
                tok.replace("▁", "")
                .replace("</w>", "")
                .strip()
            )

            if lab.startswith("B-"):
                if current:
                    entities.append({"text": " ".join(current), "label": current_label})
                current = [tok]
                current_label = lab[2:]

            elif lab.startswith("I-") and current_label == lab[2:]:
                current.append(tok)

            else:
                if current:
                    entities.append({"text": " ".join(current), "label": current_label})
                    current = []
                    current_label = None

        if current:
            entities.append({"text": " ".join(current), "label": current_label})

        return entities

def run_re(entities, model_type, model_name=None):
    """
    entities: output từ NER (list of {"text", "label"})
    model_type: "Machine Learning" | "Deep Learning"
    model_name: PCA / MaxEnt / SVM / Scaler / RandomForest (với ML)
    """

    if not entities or len(entities) < 2:
        return []

    # Chuẩn hóa entity
    norm_entities = normalize_entities(entities)

    # Sinh tất cả cặp entity
    entity_pairs = build_entity_pairs(norm_entities)

    results = []

    # ML-based RE
    if model_type == "Machine Learning":
        model = re_load_ml_model_from_kaggle(model_name)
        if model is None:
            return []

        for pair in entity_pairs:
            e1 = pair["entity_1"]
            e2 = pair["entity_2"]

            # Feature extraction (đúng với lúc train)
            features = get_relation_features(cleaned_text, e1, e2)

            pred = model.predict([features])[0]

            if pred != "No-Relation":
                results.append({
                    "entity_1": e1["text"],
                    "entity_1_type": e1["type"],
                    "relation": pred,
                    "entity_2": e2["text"],
                    "entity_2_type": e2["type"],
                })

        return results

    # DL-based RE (PhoBERT)
    else:
        model, tokenizer, id2label = re_load_dl_model_from_kaggle()
        if model is None:
            return []

        device = next(model.parameters()).device

        for pair in entity_pairs:
            e1 = pair["entity_1"]
            e2 = pair["entity_2"]

            # Chuẩn input PhoBERT cho RE
            # (phải khớp với lúc bạn train)
            input_text = f"{e1['text']} [SEP] {e2['text']}"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                pred_id = torch.argmax(logits, dim=-1).item()
                pred_label = id2label[pred_id]

            if pred_label != "No-Relation":
                results.append({
                    "entity_1": e1["text"],
                    "entity_1_type": e1["type"],
                    "relation": pred_label,
                    "entity_2": e2["text"],
                    "entity_2_type": e2["type"],
                })

        return results

def run_ie(tokens, cleaned_text):
    # NER tốt nhất = MaxEnt
    ner_entities = run_ner(tokens, cleaned_text, "Machine Learning", "MaxEnt")

    # RE tốt nhất = PhoBERT
    relations = run_re(ner_entities, "Deep Learning")

    return {
        "entities": ner_entities,
        "relations": relations
    }

# =======================
# Các hàm hỗ trợ
# =======================-
def normalize_entities(entities):
    return [
        {
            "text": e["text"],
            "type": e["label"]
        }
        for e in entities
    ]

def build_entity_pairs(entities):
    pairs = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i == j:
                continue
            pairs.append({
                "entity_1": e1,
                "entity_2": e2
            })
    return pairs

def find_file_by_pattern(directory, pattern):
    """Hàm tìm kiếm file trong thư mục dựa trên pattern"""
    search_path = os.path.join(directory, "**", pattern)
    files = glob.glob(search_path, recursive=True)
    return files[0] if files else None

def ner_extract_features(cleaned_text, tokens, tokenizer, phobert_model, device, feature_type="crf"):
    """
    feature_type: "crf" | "vector"

    Returns:
        - CRF: List[Dict]
        - vector: np.ndarray (seq_len, hidden_dim)
    """
    if feature_type == "other":
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = phobert_model(**model_inputs)
            hidden_states = outputs.last_hidden_state[0]

        word_ids = encoding.word_ids()
        X = hidden_states.cpu().numpy()

        X_tokens = []
        current_word = None
        current_vectors = []
        for vec, wid in zip(X, word_ids):
            if wid is None:
                continue

            if wid != current_word:
                if current_vectors:
                    X_tokens.append(np.mean(current_vectors, axis=0))
                current_vectors = [vec]
                current_word = wid
            else:
                current_vectors.append(vec)

        if current_vectors:
            X_tokens.append(np.mean(current_vectors, axis=0))

        return np.array(X_tokens)

    if feature_type == "crf":
        encoding = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = phobert_model(**model_inputs)
            X = outputs.last_hidden_state[0].cpu().numpy()

        sent_features = []
        for vec in X:
            feat = {f"d_{i}": float(v) for i, v in enumerate(vec)}
            feat["bias"] = 1.0
            sent_features.append(feat)

        return sent_features

    raise ValueError(f"Unknown feature_type: {feature_type}")