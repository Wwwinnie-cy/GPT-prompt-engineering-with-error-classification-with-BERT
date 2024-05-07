import torch
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        data = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = data['input_ids'][0]
        attention_mask = data['attention_mask'][0]

        if self.labels is not None:
            label = self.labels[idx]
            if not label:
                #print(f"Warning: Empty label at index {idx}")
                label = [0] * 5  # 假设有5个分类标签，给一个默认值
            label_tensor = torch.tensor(label, dtype=torch.float)
            if label_tensor.size(0) == 0:
                print(f"Error: Label tensor is empty at index {idx}")
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_tensor}
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def load_data_old(path):
    df = pd.read_json(path)
    texts = ["Question: " + text['Question'] + " Solution: " + text['Solution'] + " GPT answer: " + text['GPT answer'] for text in df['text'].values]
    labels = [label['Error Type'] for label in df['Label'].values] if 'Label' in df.columns else None
    return texts, labels

def extract_gpt_answers(item):
    # 使用正则表达式查找所有以'gpt_answer_'开头的键
    for key in item:
        #print("this is item")
        #print(item)
        if re.match(r'gpt_answer_\d+', key):
            #print("this is key", key)
            key = str(key)
            gpt_answers= item[key]
            #print(gpt_answers)
            #print(type(gpt_answers))
    # 结合所有找到的GPT答案
    #gpt_answers = " ".join([item[key] for key in gpt_answer_keys])
    return gpt_answers


def load_data(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        print(f"the length of my data is {len(data)}")

    texts = []  # 存储所有文本信息
    labels = []  # 存储所有标签信息

    # 遍历每个项目构建texts和labels
    for item in data:
        gpt_answers = " ".join(item[key] for key in item if "gpt" in key)  # 合并所有gpt开头的答案

        # 构建每个项目的文本信息
        text_entry = {
            "Question": item['problem'],
            "Solution": item['solution'],
            "GPT answer": gpt_answers
        }
        texts.append(text_entry.get("Question") + text_entry.get("Solution") + text_entry.get("GPT answer"))  # 添加到texts列表

        # 构建每个项目的标签信息
        label_entry = {
            "Error Type": item.get("error_type", [])
        }
        labels.append(label_entry.get("Error Type"))  # 添加到labels列表

    return texts, labels
    
def load_data_only_gpt(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        print(f"the length of my data is {len(data)}")

    texts = []  # 存储所有文本信息
    labels = []  # 存储所有标签信息

    # 遍历每个项目构建texts和labels
    for item in data:
        gpt_answers = " ".join(item[key] for key in item if "gpt" in key)  # 合并所有gpt开头的答案

        # 构建每个项目的文本信息
        text_entry = {
            "Question": item['problem'],
            "Solution": item['solution'],
            "GPT answer": gpt_answers
        }
        texts.append(text_entry.get("GPT answer"))  # 添加到texts列表

        # 构建每个项目的标签信息
        label_entry = {
            "Error Type": item.get("error_type", [])
        }
        labels.append(label_entry.get("Error Type"))  # 添加到labels列表

    return texts, labels
    

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def generate_pseudo_labels_fake(model, data_loader, texts_unlabeled, device, threshold):
    model.eval()
    pseudo_labeled_texts = []
    pseudo_labeled_labels = []
    num=0
    with torch.no_grad():
        for batch in data_loader:
            for i, batch in enumerate(data_loader):
                print(i)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                max_probs, preds = torch.max(probs, dim=-1)
                # 只选择置信度高于阈值的样本
                for j in range(len(max_probs)):
                    if max_probs[j] > threshold:
                        pseudo_labeled_texts.append(texts_unlabeled[i * data_loader.batch_size + j])
                        pseudo_labeled_labels.append(preds[j].item())
                    else:
                        print("too high threshold.")
                        num+=1
        print("unlabel:", num)
    return pseudo_labeled_texts, pseudo_labeled_labels

def generate_pseudo_labels_fake(model, data_loader, texts_unlabeled, device, threshold=0.5):
    model.eval()
    pseudo_labeled_texts = []
    pseudo_labeled_labels = []
    m=0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # 使用sigmoid而不是softmax
            # 使用阈值判断每个标签是否激活
            preds = (probs > threshold).int()
            for j in range(len(preds)):
                if any(preds[j]):  # 如果任何标签被激活（即存在大于阈值的概率）
                    pseudo_labeled_texts.append(texts_unlabeled[i * data_loader.batch_size + j])
                    pseudo_labeled_labels.append(preds[j].cpu().numpy().tolist())  # 保存整个标签向量
                    m+=1
    print(f"adding {len(pseudo_labeled_labels)} more samples")      
    print('真正个数', m)
                #else:
                    #print("too high threshold or no labels above threshold.")

    return pseudo_labeled_texts, pseudo_labeled_labels


def generate_pseudo_labels(model, data_loader, texts_unlabeled, device, threshold=0.5):
    model.eval()
    pseudo_labeled_texts = []
    pseudo_labeled_labels = []
    used_indices = set()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # 使用sigmoid而不是softmax
            preds = (probs > threshold).int()
            for j in range(len(preds)):
                index = i * data_loader.batch_size + j
                if index < len(texts_unlabeled) and any(preds[j]):  # 确保索引有效且标签被激活
                    print('this is predicted index:')
                    print(preds[j].cpu().numpy().tolist())
                    pseudo_labeled_texts.append(texts_unlabeled[index])
                    pseudo_labeled_labels.append(preds[j].cpu().numpy().tolist())
                    used_indices.add(index)
    print(f"Added {len(pseudo_labeled_labels)} more samples.")
    print(f'True number added: {len(used_indices)}')
    return pseudo_labeled_texts, pseudo_labeled_labels, used_indices

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

def calculate_accuracy(logits, targets):
    # 使用sigmoid函数将logits转换为0到1之间的概率
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()  # 阈值设置为0.5，大于0.5预测为1，否则为0
    correct = (predictions == targets).float()  # 对预测正确的标签计数
    # 计算所有样本的平均准确率
    accuracy = correct.mean()
    return accuracy.item()