import json

with open('processed_train_v1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'questions' in item:
        item['conclusions'] = item.pop('questions')

with open('processed_train_v1_ver2.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
