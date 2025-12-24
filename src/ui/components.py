import os
import glob
import joblib
import kagglehub
from transformers import AutoModel, AutoTokenizer, RobertaTokenizerFast, AutoModelForTokenClassification, AutoModelForSequenceClassification
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
    "RandomForest": "re_randomforest.pkl",
    "MaxEnt": "re_maxent.pkl",
    "SVM": "re_svm.pkl",
    "Scaler": "re_scaler.pkl",
    "PCA": "re_pca.pkl"
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
def ner_load_ml_model(model_type):
    """Load model ML NER từ Kaggle Hub"""
    try:
        path = kagglehub.model_download(KAGGLE_ML_HANDLE_NER)

        file_name = NER_MODEL_FILE_MAP.get(model_type)
        if not file_name:
            raise ValueError(f"Model không hỗ trợ: {model_type}")

        model_path = find_file_by_pattern(path, file_name)
        if not model_path:
            raise FileNotFoundError(f"Không tìm thấy {file_name}")

        model = joblib.load(model_path)
        return model

    except Exception as e:
        return None

@st.cache_resource
def re_load_ml_model(model_type):
    """Load model ML RE từ Kaggle Hub"""
    try:
        path = kagglehub.model_download(KAGGLE_ML_HANDLE_RE)

        file_name = RE_MODEL_FILE_MAP.get(model_type)
        if not file_name:
            raise ValueError(f"Model không hỗ trợ: {model_type}")

        model_path = find_file_by_pattern(path, file_name)
        if not model_path:
            raise FileNotFoundError(f"Không tìm thấy {file_name}")

        model = joblib.load(model_path)
        return model

    except Exception as e:
        return None

@st.cache_resource
def ner_load_dl_model():
    """Load PhoBERT NER model từ Kaggle Hub"""
    try:
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
        return model, tokenizer, id2label

    except Exception as e:
        return None, None, None

@st.cache_resource
def re_load_dl_model():
    """Load PhoBERT RE model từ Kaggle Hub"""
    try:
        path = kagglehub.model_download(KAGGLE_DL_HANDLE_RE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer từ CHÍNH thư mục model
        tokenizer = AutoTokenizer.from_pretrained(path)

        # Load model NER (KHÔNG torch.load)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        model.eval()

        # Lấy label mapping từ config (chuẩn HF)
        id2label = model.config.id2label
        return model, tokenizer, id2label

    except Exception as e:
        return None, None, None

def re_load_ml_model_from_kaggle(model_type):
    try:
        with st.spinner(f"Đang tải RE-ML model ({model_type})..."):
            model = re_load_ml_model(model_type)

        st.toast(f"Đã tải RE-ML model: {model_type}")
        return model

    except Exception as e:
        st.error(f"Lỗi load RE-ML model: {e}")
        return None
    
def re_load_dl_model_from_kaggle():
    try:
        with st.spinner(f"Đang tải RE-DL model Phobert..."):
            model = re_load_dl_model()

        st.toast(f"Đã tải RE-DL model: Phobert")
        return model

    except Exception as e:
        st.error(f"Lỗi load RE-DL model: {e}")
        return None

def ner_load_ml_model_from_kaggle(model_type):
    try:
        with st.spinner(f"Đang tải NER-ML model ({model_type})..."):
            model = ner_load_ml_model(model_type)

        st.toast(f"Đã tải NER-ML model: {model_type}")
        return model

    except Exception as e:
        st.error(f"Lỗi load NER-ML model: {e}")
        return None
    
def ner_load_dl_model_from_kaggle():
    try:
        with st.spinner(f"Đang tải NER-DL model Phobert..."):
            model = ner_load_dl_model()

        st.toast(f"Đã tải NER-DL model: Phobert")
        return model

    except Exception as e:
        st.error(f"Lỗi load NER-DL model: {e}")
        return None

# =======================
# Các hàm run NER/RE/IE
# =======================
def run_ner(tokens, cleaned_text, model_type, model_name=None, phobert_model=phobert_base_model, tokenizer=DL_tokenizer, device=device):
    if model_type == "Machine Learning":
        model = ner_load_ml_model_from_kaggle(model_name)
        if model_name == "CRF":
            features = ner_extract_features(
                cleaned_text=cleaned_text,
                tokens=tokens,
                tokenizer=DL_tokenizer,
                phobert_model=phobert_model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                feature_type="crf"
            )
            preds = model.predict([features])[0]

            entities = []
            cursor = 0
            for token, label in zip(tokens, preds):
                token_surface = token.replace("_", " ")

                start = cleaned_text.find(token_surface, cursor)

                if start == -1:
                    continue

                end = start + len(token_surface)

                entities.append({
                    "text": cleaned_text[start:end],
                    "label": label,
                    "start": start,
                    "end": end
                })

                cursor = end

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

            entities = []
            cursor = 0
            for token, label in zip(tokens, pred_labels):
                token_surface = token.replace("_", " ")

                start = cleaned_text.find(token_surface, cursor)

                if start == -1:
                    continue

                end = start + len(token_surface)

                entities.append({
                    "text": cleaned_text[start:end],
                    "label": label,
                    "start": start,
                    "end": end
                })

                cursor = end

        return entities

    else:
        model, tokenizer, id2label = ner_load_dl_model_from_kaggle()

        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True
        )

        offsets = inputs.pop("offset_mapping")[0].tolist()

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)[0].tolist()

        subtokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [id2label[p] for p in preds]

        entities = []
        current_label = None
        start = end = None

        for tok, lab, (s, e) in zip(subtokens, labels, offsets):
            if tok in ["<s>", "</s>", "<pad>"] or s == e:
                continue

            if lab.startswith("B-"):
                if current_label:
                    entities.append({
                        "text": cleaned_text[start:end],
                        "label": current_label,
                        "start": start,
                        "end": end
                    })
                current_label = lab[2:]
                start, end = s, e

            elif lab.startswith("I-") and current_label == lab[2:]:
                end = e

            else:
                if current_label:
                    entities.append({
                        "text": cleaned_text[start:end],
                        "label": current_label,
                        "start": start,
                        "end": end
                    })
                    current_label = None
                    start = end = None

        # flush cuối
        if current_label:
            entities.append({
                "text": cleaned_text[start:end],
                "label": current_label,
                "start": start,
                "end": end
            })
        return entities

def run_re(cleaned_text, entities, model_type, model_name=None, phobert_model=phobert_base_model, tokenizer=DL_tokenizer, device=device):
    """
    entities: output từ NER (list of {"text", "label", "start", "end"})
    model_type: "Machine Learning" | "Deep Learning"
    model_name: MaxEnt / SVM / RandomForest (với ML)
    """
    if not entities or len(entities) < 2:
        return []

    # ===== 1. Normalize entity =====
    norm_entities = normalize_entities(entities)

    # ===== 2. Build entity pairs =====
    entity_pairs = build_entity_pairs(norm_entities)

    if not entity_pairs:
        return []

    results = []

    # =============== ML-BASED RE ======================
    if model_type == "Machine Learning":
        model = re_load_ml_model_from_kaggle(model_name)
        scaler = re_load_ml_model_from_kaggle("Scaler")
        pca = re_load_ml_model_from_kaggle("PCA")
        if model is None:
            return []
        
        for pair in entity_pairs:
            e1 = pair["entity_1"]
            e2 = pair["entity_2"]

            marked_text = insert_entity_markers(cleaned_text, e1, e2)

            X = re_extract_embedding(
                text=marked_text,
                tokenizer=DL_tokenizer,
                phobert_model=phobert_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            X = scaler.transform(X)                 # (1, 768)
            X = pca.transform(X)    
            pred = model.predict(X)[0]

            id2label={
                0: "HAS_AREA",
                1: "HAS_ATTR",
                2: "HAS_PRICE",
                3: "LOCATED_AT",
                4: "NO_RELATION"
            }   
            pred_label = id2label[pred]

            if pred_label != "NO_RELATION":
                results.append({
                    "entity_1": e1["text"],
                    "entity_1_type": e1["type"],
                    "relation": pred_label,
                    "entity_2": e2["text"],
                    "entity_2_type": e2["type"],
                })

        return results

    # =============== DL-BASED RE ======================
    else:
        model, tokenizer, id2label = re_load_dl_model_from_kaggle()
        if model is None:
            return []
        
        for pair in entity_pairs:
            e1 = pair["entity_1"]
            e2 = pair["entity_2"]

            marked_text = insert_entity_markers(cleaned_text, e1, e2)
            inputs = tokenizer(
                marked_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits      # (1, num_labels)
                pred_id = logits.argmax(dim=-1).item()

            pred_label = id2label[pred_id]

            if pred_label in ["NO_RELATION", "No-Relation", "O"]:
                continue

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
    relations = run_re(cleaned_text, ner_entities, "Deep Learning")

    return {
        "entities": ner_entities,
        "relations": relations
    }

# =======================
# Các hàm hỗ trợ
# =======================-
def normalize_entities(entities):
    norm = []
    for ent in entities:
        label = ent["label"]

        # bỏ prefix B-/I-
        if "-" in label:
            label = label.split("-", 1)[1]

        norm.append({
            "text": ent["text"],
            "type": label,   # TYPE / AREA / PRICE / ATTR
            "start": ent["start"],
            "end": ent["end"]
        })
    return norm


def build_entity_pairs(entities):
    VALID_PAIRS = {
        ("TYPE", "AREA"),
        ("TYPE", "PRICE"),
        ("TYPE", "ATTR"),
        ("TYPE", "LOC"),
    }

    pairs = []

    for e1 in entities:
        for e2 in entities:
            if e1 is e2:
                continue

            # bỏ entity O
            if e1["type"] == "O" or e2["type"] == "O":
                continue

            if (e1["type"], e2["type"]) in VALID_PAIRS:
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

def re_extract_embedding(text, tokenizer, phobert_model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = phobert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]

    return cls_embedding.cpu().numpy()

def insert_entity_markers(text, ent1, ent2):
    entities = [
        (ent1["start"], ent1["end"], "[E1]", "[/E1]"),
        (ent2["start"], ent2["end"], "[E2]", "[/E2]")
    ]

    # Chèn từ entity có start lớn hơn trước
    for start, end, s_tag, e_tag in sorted(entities, key=lambda x: x[0], reverse=True):
        text = (
            text[:start] +
            f"{s_tag} " +
            text[start:end] +
            f" {e_tag}" +
            text[end:]
        )

    return text
