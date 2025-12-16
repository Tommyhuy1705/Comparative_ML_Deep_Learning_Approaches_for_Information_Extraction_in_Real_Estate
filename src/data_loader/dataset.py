import json
from tqdm import tqdm
from pyvi import ViTokenizer
import re

def tokenize_with_pyvi_offsets(text):
    tokenized_text = ViTokenizer.tokenize(text)
    raw_tokens = tokenized_text.split()
    
    tokens_map = []
    cursor = 0
    
    for token in raw_tokens:
        # Pyvi nối từ bằng _, ta cần đổi lại thành space để tìm trong text gốc
        # Lưu ý: Có trường hợp từ gốc có chứa _ nhưng rất hiếm
        search_token = token.replace("_", " ")
        
        # Tìm vị trí của từ này trong văn bản gốc bắt đầu từ con trỏ hiện tại
        start = text.find(search_token, cursor)
        
        # Nếu không tìm thấy (do Pyvi chuẩn hóa gì đó lạ), ta thử tìm phiên bản không case-sensitive hoặc bỏ qua
        if start == -1:
            # Fallback: Nếu không tìm thấy chính xác, ta nhảy qua (chấp nhận rủi ro nhỏ)
            # Nhưng thường Pyvi chỉ thay space bằng _ nên rất an toàn
            continue
            
        end = start + len(search_token)
        
        # Lưu token (giữ nguyên dấu _ để làm đặc trưng cho ML) và tọa độ gốc
        tokens_map.append({
            "text": token,
            "start": start,
            "end": end
        })
        
        cursor = end
        
    return tokens_map

def convert_label_studio_to_ner_data(json_data):
    dataset = []
    
    for task in tqdm(json_data, desc="Converting with Pyvi"):
        # Lấy text gốc
        if 'data' in task and 'text' in task['data']:
            raw_text = task['data']['text']
        else:
            continue

        # --- SỬ DỤNG PYVI TOKENIZER ---
        tokens_map = tokenize_with_pyvi_offsets(raw_text)
        
        # Khởi tạo nhãn O
        labels = ["O"] * len(tokens_map)

        # Map annotation vào tokens
        if task.get('annotations'):
            results = task['annotations'][0]['result']
            for item in results:
                if item['type'] == 'labels':
                    label_name = item['value']['labels'][0]
                    start_char = item['value']['start']
                    end_char = item['value']['end']

                    for i, token in enumerate(tokens_map):
                        # Logic giao thoa: Token nằm trong hoặc giao với vùng nhãn
                        # Pyvi gom từ (VD: "Hồ_Chí_Minh"), nếu nhãn chỉ gán cho "Hồ Chí" thì ta vẫn gán cả cụm là LOC
                        if token['start'] >= start_char and token['end'] <= end_char:
                            if token['start'] == start_char or (i > 0 and labels[i-1] == "O"):
                                labels[i] = f"B-{label_name}"
                            else:
                                labels[i] = f"I-{label_name}"
                        # Xử lý trường hợp Pyvi gom từ dài hơn nhãn (Overlap một phần)
                        # VD: Nhãn gán "Quận", Pyvi tách "Quận_7". Ta gán cả "Quận_7" là B-LOC
                        elif token['start'] < end_char and token['end'] > start_char:
                             if labels[i] == "O": # Ưu tiên gán nếu chưa có nhãn
                                labels[i] = f"B-{label_name}"
        
        # Tạo câu
        sentence = [(t['text'], labels[i]) for i, t in enumerate(tokens_map)]
        if len(sentence) > 0:
            dataset.append(sentence)
            
    return dataset