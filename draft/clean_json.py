"""
import json

with open('D:\\是大学啦\\DDA\\4210\\final proj\\output\\unlabel\\output_zero_shot_geom.json', 'r', encoding='utf-8') as file:
    sample_json = json.load(file)

# 过滤
filtered_json = [item for item in sample_json if item['principle_or_not'] == 0]

filtered_json_path = 'D:\\是大学啦\\DDA\\4210\\final proj\\output\\unlabel\\geometry_unlabel_without1.json'
with open(filtered_json_path, 'w', encoding='utf-8') as outfile:
    json.dump(filtered_json, outfile, indent=4)  
"""

import json
import random

# 文件路径列表

files = [
    'D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\algebra_label.json',
    'D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\clean_output_zero_shot_secondhalf.json',
    'D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\counting_first_half_label.json',
    'D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\interAlg_label.json',
    'D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\interAlg_unlabel_without1.json',
    "D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\NumThr_unlabel_without1.json",
    "D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\prealgebra_unlabel_without1.json",
    "D:\\是大学啦\\DDA\\4210\\final proj\\output\\label\\precalculus_unlabel_without1.json"
]

all_dicts = []
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        all_dicts.extend(data)  

random.shuffle(all_dicts)

"""
split_index = int(0.7 * len(all_dicts))

train_data = all_dicts[:split_index]
test_data = all_dicts[split_index:]

# 将分割后的数据保存到新的JSON文件中
with open('D:\\是大学啦\\DDA\\4210\\final proj\\train_unlabel.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open('D:\\是大学啦\\DDA\\4210\\final proj\\test_unlabel.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=4)
"""

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")