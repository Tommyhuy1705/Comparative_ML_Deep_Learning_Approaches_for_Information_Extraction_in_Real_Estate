import pandas as pd
import os
import sys
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_loader.preprocess import clean_and_format_content, clean_title

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
RAW_DATA_DIR = os.path.join(project_root, "data/01_raw")
INPUT_FILE_NAME = "BDS_Mien Nam_cleaned.csv"

OUTPUT_DIR = os.path.join(project_root, "data/02_intermediate")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "official_dataset_mien_nam.json")

def process_pipeline():
    input_path = os.path.join(RAW_DATA_DIR, INPUT_FILE_NAME)
    
    if not os.path.exists(input_path):
        # Thử fallback sang CSV
        input_path_csv = input_path.replace(".xlsx", ".csv")
        if os.path.exists(input_path_csv):
            input_path = input_path_csv
        else:
            print(f"Lỗi: Không tìm thấy file input '{INPUT_FILE_NAME}' hoặc bản .csv của nó trong '{RAW_DATA_DIR}'")
            return

    # 2. Đọc dữ liệu
    print(f"-> Đang đọc file: {input_path} ...")
    try:
        if input_path.endswith(".xlsx"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Chuẩn hóa tên cột
    df.columns = [c.lower() for c in df.columns]
    
    if 'content' not in df.columns:
        print("Lỗi: File không có cột 'content'.")
        return

    total_rows = len(df)
    print(f"Tổng số dòng trong file gốc: {total_rows}")
    json_output = []
    valid_content_count = 0

    print(f"-> Đang xử lý dữ liệu...")
    
    for index, row in df.iterrows():
        raw_content = str(row['content'])
        
        if not raw_content or raw_content.lower() == 'nan': continue
        
        cleaned_text = clean_and_format_content(raw_content)
        one_block_text = re.sub(r'[\n\r]+', ' ', cleaned_text)
        
        one_block_text = re.sub(r'\s+', ' ', one_block_text).strip()
        
        if len(one_block_text) > 20:
            item = {
                "text": one_block_text
            }
            json_output.append(item)

    # 4. Lưu file JSON kết quả
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)

    final_count = len(json_output)
    
    print("\n" + "="*40)
    print("HOÀN TẤT QUÁ TRÌNH XỬ LÝ")
    print("="*40)
    print(f"1. Tổng số dòng dữ liệu (Raw):   {total_rows:,}")
    print(f"2. Số dòng có nội dung (Valid):  {valid_content_count:,}")
    print(f"3. Số lượng Text kết quả (Final): {final_count:,}")
    print("-" * 40)
    print(f"Đã loại bỏ: {valid_content_count - final_count} bài (do quá ngắn hoặc lỗi)")
    print(f"File output: {OUTPUT_JSON}")
    print("="*40 + "\n")

if __name__ == "__main__":
    process_pipeline()