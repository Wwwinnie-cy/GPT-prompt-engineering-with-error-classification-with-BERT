"""
def prepare_data(data):
    texts = []
    labels = []
    for item in data:
        # 合并感兴趣的文本部分，忽略 Low Principle
        text = f"Question: {item['text']['Question']} Solution: {item['text']['Solution']} GPT answer: {item['text']['GPT answer']}"
        texts.append(text)
        labels.append(item['Label']['Error Type'])
    return texts, labels

import pandas as pd

# 示例 JSON 数据路径
data_path = 'your_data.json'

# 加载数据
df = pd.read_json(data_path)

# 合并文本字段
df['text'] = 'Question: ' + df['text']['Question'] + \
             ' Solution: ' + df['text']['Solution'] + \
             ' GPT answer: ' + df['text']['GPT answer']

# 获取标签
df['label'] = df['Label']['Error Type']
"""
import json
import random

# 定义一些示例文本用于生成数据
questions = [
    "What causes rain?",
    "How does the internet work?",
    "What is artificial intelligence?",
    "Why is the sky blue?",
    "What is quantum computing?"
]

solutions = [
    "Rain is caused by moisture condensing in the air.",
    "The internet works by transmitting data through networks.",
    "Artificial intelligence is a branch of computer science dealing with simulation of intelligent behavior.",
    "The sky is blue because of the scattering of sunlight by the atmosphere.",
    "Quantum computing is computing using quantum-mechanical phenomena."
]

gpt_answers = [
    "Rain happens when moisture in the air condenses into water droplets.",
    "The internet is a network that connects computers worldwide.",
    "AI is the field of creating machines that can think like humans.",
    "The sky is blue due to the Rayleigh scattering of sunlight by the Earth's atmosphere.",
    "Quantum computing uses quantum bits to encode information."
]

error_types = [0, 1, 2]  # 假设有三种错误类型

# 生成数据
data = []
for _ in range(50):
    question = random.choice(questions)
    solution = random.choice(solutions)
    gpt_answer = random.choice(gpt_answers)
    error_type = []
    for i in range(5):
        label = random.randint(0,1)
        error_type.append(label)
    
    sample = {
        "text": {
            "Question": question,
            "Solution": solution,
            "GPT answer": gpt_answer
        },
        "Label": {
            'Error Type': error_type
        }
    }
    data.append(sample)

# 将数据保存为 JSON 文件
with open('D:\\是大学啦\\DDA\\4210\\final proj\\label_sample_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Sample dataset has been saved to 'sample_dataset.json'.")
