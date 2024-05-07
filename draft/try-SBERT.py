import json

filename = 'D:\\是大学啦\\DDA\\4210\\final proj\\output\\counting\\output_zero_shot_secondhalf.json'
#filename = 'D:\\是大学啦\\DDA\\4210\\final proj\\label_sample_dataset.json'

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

data = load_json(filename)
print(data)
