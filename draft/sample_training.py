import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertForSequenceClassification, BertTokenizer
from bert_class_definition import TextDataset, load_data, generate_pseudo_labels,calculate_accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import random_split

# 超参数设置
num_labels = 3
learning_rate = 2e-5
batch_size = 32
epochs = 3
threshold = 0.05  # 置信度阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "D:\\是大学啦\\DDA\\4210\\final proj\\bert_base_uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.to(device)

# 数据集构建
texts, labels = load_data("D:\\是大学啦\\DDA\\4210\\final proj\\sample_dataset.json")
texts_unlabeled, _ = load_data('D:\\是大学啦\\DDA\\4210\\final proj\\unlabel_sample_dataset.json')

labeled_dataset = TextDataset(texts, labels, tokenizer)  # 确保传递tokenizer
unlabeled_dataset = TextDataset(texts_unlabeled, None, tokenizer)  # 对于未标注数据同样确保传递tokenizer


train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
pseudo_texts, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, texts_unlabeled, device, threshold)


# 生成伪标签并创建新的训练集
all_texts = texts + pseudo_texts
all_labels = labels + pseudo_labels

all_dataset = TextDataset(all_texts, all_labels, tokenizer)

n_samples = len(all_dataset)
print(f"There are total {n_samples} samples in all training data")
n_train = int(n_samples * 0.8)
n_val = n_samples - n_train

train_dataset, val_dataset = random_split(all_dataset, [n_train, n_val])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# 早停设置
best_loss = float('inf')
patience = 3
trigger_times = 0

# 训练循环
for epoch in range(10):  # 假设最大 epoch 数为 10
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += calculate_accuracy(outputs.logits, targets) * len(targets)
        total += len(targets)
    
    train_accuracy = correct / total

    # 验证循环
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=inputs, labels=targets)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()
            correct_val += calculate_accuracy(outputs.logits, targets) * len(targets)
            total_val += len(targets)
    
    val_accuracy = correct_val / total_val
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Train Acc: {train_accuracy}, Val Acc: {val_accuracy}")

    scheduler.step(avg_val_loss)

    # 早停判断
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("Training completed.")