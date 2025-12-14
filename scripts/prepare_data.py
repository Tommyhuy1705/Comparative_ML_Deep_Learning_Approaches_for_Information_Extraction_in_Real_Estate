import pandas as pd
import os
import sys
import json
import re

from sklearn.model_selection import train_test_split
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_loader.preprocess import clean_and_format_content, clean_title

RAW_DATA_DIR = os.path.join(project_root, "data/01_raw")
INPUT_FILE_NAME = "BDS_Mien_Trung_Cleaned.xlsx"

OUTPUT_DIR = os.path.join(project_root, "data/02_intermediate")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "official_dataset_mien_trung.json")

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
    
def merge_labeled_datasets(input_files, output_file):
    print(f"\n Đang gộp {len(input_files)} file dataset...")
    
    merged_data = []
    current_id = 1
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Cảnh báo: Không tìm thấy file {file_path}")
            continue
            
        print(f"   -> Đọc file: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Duyệt qua từng task trong file
            for task in data:
                # 1. Reset ID của Task (quan trọng để không bị trùng)
                task['id'] = current_id
                
                # (Tùy chọn) Nếu muốn lưu lại tên file gốc vào meta
                if 'meta' not in task: task['meta'] = {}
                task['meta']['source_file'] = os.path.basename(file_path)
                
                merged_data.append(task)
                current_id += 1
                
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")

    # Lưu file tổng
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
        
    print(f"Đã gộp xong! Tổng cộng: {len(merged_data)} bài đăng.")
    print(f"File kết quả: {output_file}")

def split_dataset(input_file, train_file, test_file, dev_file, test_size=0.2, dev_size=0.1, random_state=42):
    print(f"\n Đang chia dữ liệu từ file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data, temp_data = train_test_split(data, test_size=(test_size + dev_size), random_state=random_state)
    relative_dev_size = dev_size / (test_size + dev_size)
    dev_data, test_data = train_test_split(temp_data, test_size=relative_dev_size, random_state=random_state)
    
    # Lưu các file kết quả
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
        
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
        
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
        
    print(f"Chia dữ liệu xong:")
    print(f" - Train: {len(train_data)} bài -> {train_file}")
    print(f" - Dev:   {len(dev_data)} bài -> {dev_file}")
    print(f" - Test:  {len(test_data)} bài -> {test_file}")

if __name__ == "__main__":
    # process_pipeline()
    files_to_merge = [
        "data/03_primary/label_data_mien_trung.json",
        "data/03_primary/label_data_mien_tay.json",
        "data/03_primary/label_data_mien_nam.json",
        "data/03_primary/label_data_mien bac.json"
    ]
    output_master = "data/03_primary/final_labeled_dataset.json"
    #merge_labeled_datasets(files_to_merge, output_master)

    input_file = "data/03_primary/final_labeled_dataset.json"
    train_file = "data/04_model_input/train_dataset.json"
    test_file = "data/04_model_input/test_dataset.json"
    dev_file = "data/04_model_input/dev_dataset.json"

    split_dataset(input_file, train_file, test_file, dev_file, test_size=0.2, dev_size=0.1, random_state=42)
