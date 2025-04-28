import json

with open("train_v1.json", "r") as f:
    data = json.load(f)
    
processed = []
for dp in data:
    dp.pop("premises-FOL", None)
    dp.pop("explanation", None)
    
    for q, a, i in zip(dp["questions"], dp["answers"], dp['idx']):
        if a in ('Yes', 'No', 'Uncertain'):
            processed.append({
                "premises": dp["premises-NL"],
                "questions": [q],
                "answers": [a],
                "idx": [i]
            })

with open('preprocessed_train.json', 'w', encoding='utf-8') as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)