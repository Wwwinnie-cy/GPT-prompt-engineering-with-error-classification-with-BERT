import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt
from bert_class_definition import TextDataset, load_data, generate_pseudo_labels, calculate_accuracy, load_data_only_gpt
import json
import numpy as np

epochs = 1000
num_labels = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/workspace/AdBRC/ningzhihan/final_proj/bert_base_uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)

## check gpu
print("Using device:", device)

if device.type == 'cuda':
    print("CUDA is available.")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Memory Allocated:", torch.cuda.memory_allocated(0))
    print("CUDA Device Memory Cached:", torch.cuda.memory_reserved(0))
else:
    print("CUDA is not available, using CPU.")


texts, labels = load_data("/workspace/AdBRC/ningzhihan/final_proj/output/label/labell_all.json")
texts_unlabeled, _ = load_data("/workspace/AdBRC/ningzhihan/final_proj/train_unlabel.json")

labeled_dataset = TextDataset(texts, labels, tokenizer)
unlabeled_dataset = TextDataset(texts_unlabeled, None, tokenizer)

num_data = len(labeled_dataset)
indices = torch.randperm(num_data)

split = int(0.8 * num_data)
train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(labeled_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(labeled_dataset, batch_size=32, sampler=val_sampler)

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
batch_sizes = [32, 63, 128]
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

best_val_loss = float('inf')
best_params = {}

train_size = int(0.8 * len(labeled_dataset))
val_size = len(labeled_dataset) - train_size
train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size])

best_val_loss = float('inf')
patience = 3
patience_counter = 0

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for threshold in thresholds:
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            for epoch in range(30): 
                # 训练阶段
                model.train()
                total_train_loss, total_train_correct, total_train_samples = 0, 0, 0

                train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                for batch in train_loader:
                    inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
                    model.zero_grad()
                    outputs = model(input_ids=inputs, labels=targets)
                    loss = criterion(outputs.logits, targets.float())
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    total_train_correct += calculate_accuracy(outputs.logits, targets) * inputs.size(0)
                    total_train_samples += inputs.size(0)

                avg_train_loss = total_train_loss / len(train_loader)
                train_accuracy = total_train_correct / total_train_samples

                train_losses.append(avg_train_loss)
                train_accs.append(train_accuracy)

                pseudo_texts, pseudo_labels, used_indices = generate_pseudo_labels(model, DataLoader(unlabeled_dataset, batch_size=batch_size), texts_unlabeled, device, threshold)
                if pseudo_texts:
                        texts += pseudo_texts
                        labels += pseudo_labels
                        labeled_dataset = TextDataset(texts, labels, tokenizer)
                        texts_unlabeled = [text for idx, text in enumerate(texts_unlabeled) if idx not in used_indices]
                print("Remaining unlabeled samples:", len(texts_unlabeled))

                model.eval()
                total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
                        outputs = model(input_ids=inputs, labels=targets)
                        val_loss = criterion(outputs.logits, targets.float())
                        total_val_loss += val_loss.item()
                        total_val_correct += calculate_accuracy(outputs.logits, targets) * inputs.size(0)
                        total_val_samples += inputs.size(0)

                avg_val_loss = total_val_loss / len(train_loader)
                val_accuracy = total_val_correct / total_val_samples
                val_losses.append(avg_val_loss)
                val_accs.append(val_accuracy)
                print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'threshold': threshold,
                        'epoch': epoch
                    }
                    print("The best paramters are:")
                    print(best_params)
                        # 早停判断
                    torch.save(model.state_dict(), "/workspace/AdBRC/ningzhihan/final_proj/best_model.pth")
                    print('already save model')
                
                """
                if np.abs(avg_val_loss - best_val_loss) > 0.00005:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                """
                    
                    
print("Best parameters:", best_params)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Accuracy')
plt.plot(range(1, len(val_accs)+1), val_accs, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/workspace/AdBRC/ningzhihan/final_proj/training_performance.png')
print('picture save')
plt.show()
print('finish training!!')

print("start evaluation")

## 加载模型
config_path = "/workspace/AdBRC/ningzhihan/final_proj/bert_base_uncased"
config = BertConfig.from_pretrained(config_path, num_labels=num_labels)
model = BertForSequenceClassification(config)
model_path = "/workspace/AdBRC/ningzhihan/final_proj/best_model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)


texts_test, labels_test = load_data("/workspace/AdBRC/ningzhihan/final_proj/test_unlabel.json")
test_dataset = TextDataset(texts_test, labels_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model.eval()
test_results = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        outputs = model(input_ids=inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        test_results.extend(predictions.tolist())

print(test_results)


output_data = [
    {"text": text, "predicted_label": label}
    for text, label in zip(texts_test, test_results)
]


with open('/workspace/AdBRC/ningzhihan/final_proj/predicted_results.json', 'w', encoding='utf-8') as f:
    print("prediction save")
    json.dump(output_data, f, indent=4)

print("evaluation finish!")
