import json
from tqdm import tqdm
from pyvi import ViTokenizer
import re
import itertools

def tokenize_with_pyvi_offsets(text):
    tokenized_text = ViTokenizer.tokenize(text)
    raw_tokens = tokenized_text.split()
    
    tokens_map = []
    cursor = 0
    
    for token in raw_tokens:
        search_token = token.replace("_", " ")
        start = text.find(search_token, cursor)
        if start == -1:
            continue
            
        end = start + len(search_token)
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

def prepare_re_data_from_json(json_data):
    dataset = []
    for task in json_data:
        text = task['data']['text']
        entities = {}
        relations = []
        if not task['annotations']:
            continue
            
        # Duyệt qua từng kết quả trong annotation
        for item in task['annotations'][0]['result']:
            # Lấy Entities
            if item['type'] == 'labels':
                entities[item['id']] = {
                    'id': item['id'],
                    'text': item['value']['text'],
                    'start': item['value']['start'],
                    'end': item['value']['end'],
                    'label': item['value']['labels'][0]
                }
            elif item['type'] == 'relation':
                if 'labels' not in item or not item['labels']:
                    continue
                relations.append({
                    'from': item['from_id'],
                    'to': item['to_id'],
                    'label': item['labels'][0]
                })

        true_relation_map = {}
        for rel in relations:
            true_relation_map[(rel['from'], rel['to'])] = rel['label']
            
        entity_ids = list(entities.keys())
        for id1, id2 in itertools.permutations(entity_ids, 2):
            e1 = entities[id1]
            e2 = entities[id2]

            label = true_relation_map.get((id1, id2), 'NO_RELATION')
            
            dataset.append({
                'text': text,
                'ent1': e1,
                'ent2': e2,
                'label': label
            })
            
    return dataset