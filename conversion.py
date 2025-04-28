import json

with open('filtered_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
converted = []

for dp in data:
    premises = dp['premises']
    
    for idx_list, conclusion in zip(dp['idx'], dp['conclusions']):
        lines = [f"sent {i}: {s}" for i, s in enumerate(premises, start=1)]
        
        prompt = "\n".join(lines) + f"\nDoes it imply that {conclusion}?"
        
        if len(idx_list) == 1:
            target = f"sent {idx_list[0]}. Therefore, {conclusion}"
        else:
            first = f"sent {idx_list[0]}."
            others = ", ".join(f"sent {j}" for j in idx_list[1:])
            target = f"{first} We know that {others}. Therefore, {conclusion}"
            
        converted.append({
            "prompt": prompt,
            "target": target,
        })
        
with open('converted_train.json', 'w', encoding='utf-8') as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)
    