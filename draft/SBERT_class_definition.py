from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class MLPClassifier_origin(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        if isinstance(x, dict) and 'sentence_embedding' in x:
            x = x['sentence_embedding']  # 提取句子嵌入
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        if isinstance(x, dict) and 'sentence_embedding' in x:
            x = x['sentence_embedding']
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return {'sentence_embedding': x}  # 确保输出是字典


class ExampleDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples  # 这是一个包含InputExample的列表
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # 对example.texts进行编码
        inputs = self.tokenizer(
            example.texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}  # 移除批次维度，使数据适合单样本处理
        inputs['labels'] = torch.tensor(example.label)
        return inputs

class CustomSoftmaxLoss(nn.Module):
    def __init__(self, num_labels):
        super(CustomSoftmaxLoss, self).__init__()
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # 确保 logits 是一个张量
        if isinstance(logits, list):
            logits = torch.tensor(logits)  # 将列表中的张量堆叠成一个新的张量
        return self.loss_fn(logits, labels)

class CustomClassifierLoss(nn.Module):
    def __init__(self, model, num_labels):
        super(CustomClassifierLoss, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, texts, labels):
        # 确保 texts 是批次中所有句子的列表
        sentence_embeddings = self.model.encode(texts, convert_to_tensor=True)
        return self.loss_fn(sentence_embeddings, labels)