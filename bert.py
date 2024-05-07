import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt
from bert_class_definition import TextDataset, load_data, generate_pseudo_labels, calculate_accuracy
import json

num_labels = 5
learning_rate = 2e-5
batch_size = 32
epochs = 1000
threshold = 0.5  # 置信度阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "D:\\是大学啦\\DDA\\4210\\final proj\\bert_base_uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
model.to(device)


texts, labels = load_data("D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\labell_all.json")
texts_unlabeled, _ = load_data("D:\\是大学啦\\DDA\\4210\\final proj\\train_unlabel.json")


labeled_dataset = TextDataset(texts, labels, tokenizer)
unlabeled_dataset = TextDataset(texts_unlabeled, None, tokenizer)


optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
criterion = BCEWithLogitsLoss()


train_losses, val_losses = [], []
train_accs, val_accs = [], []

# 早停设置
best_val_loss = float('inf')
patience = 3
patience_counter = 0

# start training
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
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

    # semi-supervise
    pseudo_texts, pseudo_labels = generate_pseudo_labels(model, DataLoader(unlabeled_dataset, batch_size=batch_size), texts_unlabeled, device, threshold)
    if len(pseudo_texts) > 0:
        texts += pseudo_texts
        labels += pseudo_labels
        labeled_dataset = TextDataset(texts, labels, tokenizer)

    # evaluate
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    val_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)
    for batch in val_loader:
        inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=targets)
            val_loss = criterion(outputs.logits, targets.float())
            total_val_loss += val_loss.item()
            total_val_correct += calculate_accuracy(outputs.logits, targets) * inputs.size(0)
            total_val_samples += inputs.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / total_val_samples

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    scheduler.step(avg_val_loss)

    # 早停判断
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Plot loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('D:\\是大学啦\\DDA\\4210\\final proj\\training_performance.png')
plt.show()

torch.save(model.state_dict(), "D:\\是大学啦\\DDA\\4210\\final proj\\model_state_dict.pth")
print('finish training!!')


## start evaluation
print("start evaluation")


config_path = "D:\\是大学啦\\DDA\\4210\\final proj\\bert_base_uncased"
config = BertConfig.from_pretrained(config_path, num_labels=num_labels)
model = BertForSequenceClassification(config)
model_path = "D:\\是大学啦\\DDA\\4210\\final proj\\model_state_dict.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)


texts_test, labels_test = load_data("D:\\是大学啦\\DDA\\4210\\final proj\\test_unlabe.json")
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

# 写入到 JSON 
with open('D:\\是大学啦\\DDA\\4210\\final proj\\predicted_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("evaluation finish!")