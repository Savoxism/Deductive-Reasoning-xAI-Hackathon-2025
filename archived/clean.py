import json

with open("raw_train.json", "r") as f:
    data = json.load(f)
    
processed = []
for dp in data:
    dp.pop("premises-FOL", None)
    dp.pop("explanation", None)
    dp.pop("idx", None)
    
    for q, a in zip(dp["questions"], dp["answers"]):
        if a in ('Yes', 'No', 'Uncertain'):
            processed.append({
                "premises": dp["premises-NL"],
                "hypothesis": q,
                "answers": a,
            })

with open('preprocessed_train.json', 'w', encoding='utf-8') as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)