import sys
from pathlib import Path

# Add src to PYTHONPATH
current_file = Path(__file__).resolve()
src_path = current_file.parents[1]
sys.path.append(str(src_path))

import streamlit as st
import torch
import pandas as pd
# ---------------------------------------------------------
# 1. SETUP PATH & IMPORTS
# ---------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root / "src"))

try:
    from data_loader.preprocess import clean_and_format_content
    from data_loader.dataset import tokenize_with_pyvi_offsets
    from components import run_ner, run_re, run_ie
except ImportError as e:
    st.error(f"Lỗi import module: {e}. Hãy đảm bảo cấu trúc thư mục 'src' đúng như trong repo.")
    st.stop()

KAGGLE_ML_HANDLE = "donhuyn/ml-ner-er-vietnamese/scikitLearn/ner"
KAGGLE_DL_HANDLE = "donhuyn/phobert-ner-er-vietnamese/pyTorch/ner"
PHOBERT_NAME = "vinai/phobert-base-v2"

# ---------------------------------------------------------
# CONFIG & KAGGLE HUB HANDLES
# ---------------------------------------------------------
st.set_page_config(page_title="Real Estate NER-RE Pipeline", layout="wide")

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("Real Estate Info Extraction")
st.markdown(f"Các mô hình được tải từ **Kaggle Hub**:\n- ML: `{KAGGLE_ML_HANDLE}`\n- DL: `{KAGGLE_DL_HANDLE}`")

# Sidebar
with st.sidebar:
    st.header("Cấu hình")

    task_type = st.radio(
        "Chọn bài toán:",
        ["NER", "RE", "IE"]
    )

    if task_type == "NER":
        model_type = st.radio(
            "Chọn loại mô hình:",
            ["Machine Learning", "Deep Learning"]
        )
        if model_type == "Machine Learning":
                model_ML = st.radio(
                "Chọn loại mô hình ML cho NER:",
                ["CRF", "MaxEnt", "SVM"]
            )
        else:
            model_DL = st.radio(
            "Chọn loại mô hình DL cho NER:",
            ["PhoBERT"]
        )
    elif task_type == "RE":
        model_type = st.radio(
            "Chọn loại mô hình:",
            ["Machine Learning", "Deep Learning"]
        )
        if model_type == "Machine Learning":
                model_ML = st.radio(
                "Chọn loại mô hình ML cho RE:",
                ["PCA", "MaxEnt", "SVM", "Scaler", "RandomForest"]
            )
        else:
            model_DL = st.radio(
            "Chọn loại mô hình DL cho RE:",
            ["PhoBERT"]
        )
    else:
        model_type = None

    st.divider()
    st.caption("Pipeline IE sử dụng NER (MaxEnt) + RE (PhoBERT).")

# Input Area
input_text = st.text_area("Nhập tin đăng bất động sản:", height=150, 
                        placeholder="Bán nhà hẻm xe hơi 50m2 đường Cách Mạng Tháng 8, Quận 10, giá 6.5 tỷ, sổ hồng đầy đủ")

if st.button("Thực hiện Trích xuất", type="primary"):
    if not input_text:
        st.warning("Vui lòng nhập văn bản.")
        st.stop()
    else:        
        # 1. Preprocess
        st.subheader("1. Preprocessing")
        cleaned_text = clean_and_format_content(input_text)
        with st.expander("Kết quả làm sạch", expanded=False):
            st.text(cleaned_text)

        # 2. Tokenize
        st.subheader("2. Tokenize (PyVi)")
        tokens_map = tokenize_with_pyvi_offsets(cleaned_text)
        tokens = [t["text"].replace("</w>", "") for t in tokens_map]
        st.write(f"Tokens: `{' '.join(tokens)}`") # Hiển thị tokens

        # 3. Kết quả trích xuất
        st.subheader("3. Kết quả trích xuất")
        
        entities = []
        relations = []

        if task_type == "NER":
            if model_type == "Machine Learning":
                model_name = model_ML # ML
            else:
                model_name = None # DL

            entities = run_ner(
                tokens=tokens,
                cleaned_text=cleaned_text,
                model_type=model_type,
                model_name=model_name
            )

            if entities:
                st.success(f"Phát hiện {len(entities)} thực thể")
                st.dataframe(pd.DataFrame(entities), use_container_width=True)

                st.markdown("### JSON Output (NER)")
                st.json(entities)
            else:
                st.warning("Không phát hiện thực thể nào")

        elif task_type == "RE":
            if model_type == "Machine Learning":
                model_name = model_ML
            else:
                model_name = None

            entities = run_ner(
                tokens = tokens, 
                cleaned_text = cleaned_text, 
                model_type = "Machine Learning", 
                model_name = "MaxEnt") # Tự động chọn ML cho task NER

            relations = run_re(
                entities=entities,
                model_type=model_type,
                model_name=model_name
            )

            if relations:
                st.success(f"Phát hiện {len(relations)} quan hệ")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)

                st.markdown("### JSON Output (RE)")
                st.json(relations)
            else:
                st.warning("Không phát hiện quan hệ nào")
        
        else:
            ie_result = run_ie(tokens, cleaned_text)

            st.success("Hoàn tất pipeline IE")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Thực thể (Entities)")
                st.dataframe(pd.DataFrame(ie_result["entities"]), use_container_width=True)

            with col2:
                st.markdown("### Quan hệ (Relations)")
                st.dataframe(pd.DataFrame(ie_result["relations"]), use_container_width=True)

            st.markdown("### JSON Output (IE)")
            st.json(ie_result)