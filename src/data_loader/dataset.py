import json
import re
import itertools
import torch

from tqdm import tqdm
from pyvi import ViTokenizer
from torch.utils.data import Dataset

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

        # PYVI TOKENIZER
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
                        if token['start'] >= start_char and token['end'] <= end_char:
                            if token['start'] == start_char or (i > 0 and labels[i-1] == "O"):
                                labels[i] = f"B-{label_name}"
                            else:
                                labels[i] = f"I-{label_name}"
                        elif token['start'] < end_char and token['end'] > start_char:
                             if labels[i] == "O":
                                labels[i] = f"B-{label_name}"
        
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

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_list, label_list = self.data[index]
        
        tokenized_inputs = self.tokenizer(
            text_list,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        prev_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_str = label_list[word_idx]
                label_ids.append(self.label2id.get(label_str, -100))
            else:
                label_ids.append(-100) 
            prev_word_idx = word_idx

        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class REDataset(Dataset):
    def __init__(self, data_pairs, tokenizer, label2id, max_len=256):
        self.data = data_pairs
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        text = item.get('text', '')
        label_str = item.get('label', 'NO_RELATION')
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id.get(label_str, 0), dtype=torch.long)
        }