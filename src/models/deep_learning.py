import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

class PhoBertForNER(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super(PhoBertForNER, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits) if loss is not None else logits

class PhoBertForRE(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super(PhoBertForRE, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return (loss, logits) if loss is not None else logits

class DLTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, id2label):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.id2label = id2label
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=labels)
            
            loss = outputs[0]
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
        return total_loss / len(self.train_loader)

    def evaluate(self, task_type='ner'):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                if isinstance(outputs, tuple): outputs = outputs[1]
                
                if task_type == 'ner':
                    preds = torch.argmax(outputs, dim=2).cpu().numpy()
                    labels = labels.cpu().numpy()
                    
                    for i in range(len(labels)):
                        sent_preds = []
                        sent_labels = []
                        for j in range(len(labels[i])):
                            if labels[i][j] != -100:
                                sent_preds.append(self.id2label[preds[i][j]])
                                sent_labels.append(self.id2label[labels[i][j]])
                        all_preds.extend(sent_preds)
                        all_labels.extend(sent_labels)
                        
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend([self.id2label[p] for p in preds])
                    all_labels.extend([self.id2label[l] for l in labels.cpu().numpy()])

        print(classification_report(all_labels, all_preds))
        return f1_score(all_labels, all_preds, average='macro')
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")