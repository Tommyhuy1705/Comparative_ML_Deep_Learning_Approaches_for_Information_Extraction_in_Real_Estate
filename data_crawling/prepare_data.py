import pandas as pd
import os
import re
from src.data_loader.preprocess import clean_and_format_content, clean_title

RAW_DATA_DIR = "data/01_raw"
OUTPUT_FILE = "data/02_intermediate/official_bds_data.csv"

def process_pipeline():
    files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

    print(f" Tìm thấy {len(files)} file dữ liệu thô: {files}")

    all_data = []
    for f in files:
        try:
            print(f"-> Đang xử lý: {f} ...")
            df = pd.read_csv(f)
            
            df.columns = [c.lower() for c in df.columns]
            
            if 'content' not in df.columns:
                print(f"   [SKIP] File {f} không có cột 'content'.")
                continue
            
            # Tạo DataFrame tạm
            temp_df = pd.DataFrame()
            
            # Xử lý Title (nếu có)
            if 'title' in df.columns:
                temp_df['title'] = df['title'].fillna("No Title").apply(clean_title)
            else:
                temp_df['title'] = "No Title"
                
            # Xử lý Content (Dùng hàm từ src/data_loader/preprocess.py)
            temp_df['content'] = df['content'].apply(clean_and_format_content)
            
            # Bỏ dòng trống
            temp_df = temp_df[temp_df['content'].str.strip() != ""]
            
            all_data.append(temp_df)
            
        except Exception as e:
            print(f"   [ERROR] Lỗi file {f}: {e}")

    # 2. Gộp và đánh ID
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Tạo ID tự tăng
        final_df.insert(0, 'id', range(1, len(final_df) + 1))
        
        # 3. Lưu file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n XONG! Đã lưu {len(final_df)} dòng vào: {OUTPUT_FILE}")
        print("Cấu trúc file:", final_df.columns.tolist())
    else:
        print("\n Không có dữ liệu nào được xử lý.")

if __name__ == "__main__":
    process_pipeline()
    df = pd.read_csv('data/02_intermediate/official_bds_data.csv')
    df.drop_duplicates(subset=['content'], inplace=True)

    # Lưu file chốt
    df.to_csv('data/02_intermediate/final_dataset_ready_for_labeling.csv', index=False, encoding='utf-8-sig')
    print("✅ Đã xử lý xong! File sẵn sàng để gán nhãn nằm ở: data/02_intermediate/final_dataset_ready_for_labeling.csv")


