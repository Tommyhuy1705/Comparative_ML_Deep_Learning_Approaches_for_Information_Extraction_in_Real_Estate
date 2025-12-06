import pandas as pd
import json
import os
import math

INPUT_CSV = "D:\\STUDY\\XỬ LÝ NGÔN NGỮ TỰ NHIÊN\\Đồ Án\\Clean\\Comparative_ML_Deep_Learning_Approaches_for_Information_Extraction_in_Real_Estate-main\\data\\01_raw\\[NLP] BĐS - Miền Tây.csv"
OUTPUT_JSON = "D:\\STUDY\\XỬ LÝ NGÔN NGỮ TỰ NHIÊN\\Đồ Án\\Clean\\Comparative_ML_Deep_Learning_Approaches_for_Information_Extraction_in_Real_Estate-main\\official_dataset.json"
OUTPUT_DIR = "D:\\STUDY\\XỬ LÝ NGÔN NGỮ TỰ NHIÊN\\Đồ Án\\Clean\\Comparative_ML_Deep_Learning_Approaches_for_Information_Extraction_in_Real_Estate-main\\data\\02_intermediate\\data_label"

def convert_csv_to_sentence_json():
    if not os.path.exists(INPUT_CSV):
        if os.path.exists("official_dataset_ready_for_labeling.csv"):
            INPUT_PATH = "official_dataset_ready_for_labeling.csv"
        else:
            print(f"Lỗi: Không tìm thấy file input tại {INPUT_CSV}")
            return
    else:
        INPUT_PATH = INPUT_CSV

    print(f"-> Đang đọc file CSV: {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    json_output = []
    total_posts = 0

    for index, row in df.iterrows():
        content = str(row['content'])
        if not content or content.lower() == 'nan':
            continue

        content = content.strip()
        if len(content) <= 5:
            continue

        item = {
            "text": content      
        }
        json_output.append(item)
        total_posts += 1

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)

    print("XONG! Đã chuyển đổi thành công.")
    print(f"   - Số bài đăng (số ô content hợp lệ): {total_posts}")
    print(f"   - File kết quả: {OUTPUT_JSON}")

    NUM_PARTS = 4
    chunk_size = math.ceil(total_posts / NUM_PARTS)
    print(f"\n-> Đang chia thành {NUM_PARTS} phần (Mỗi phần khoảng {chunk_size} bài)...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_PARTS):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        part_data = json_output[start_idx:end_idx]
        file_name = f"dataset_{i+1}.json"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(part_data, f, ensure_ascii=False, indent=4)

        print(f"   - Đã tạo: {file_name} ({len(part_data)} bài)")

    print(f"Dataset label đã được lưu vào '{OUTPUT_DIR}'")

if __name__ == "__main__":
    convert_csv_to_sentence_json()
