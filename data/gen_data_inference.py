import json
import re

def load_selection_data(path="selection_train.json"):
    with open(path, "r") as f:
        return json.load(f)

def convert_to_inference(sel_examples):
    inf_examples = []
    for ex in sel_examples:
        prompt = ex["prompt"]
        target = ex["target"]

        # 1) Build a mapping from sent index to sentence text
        sent_map = {}
        for line in prompt.splitlines():
            m = re.match(r"sent (\d+): (.+)", line)
            if m:
                idx = int(m.group(1))
                sent_map[idx] = m.group(2).strip()

        # 2) Split selection target into segments by ". "
        segments = [seg.strip() for seg in re.split(r"\.\s*", target) if seg.strip()]
        # segments[0] = "sent i"
        # segments[1] = "We know that sent j and sent k"
        # segments[2] = "Therefore, ..."

        # 3) Extract rule index
        rule_idx = int(re.match(r"sent (\d+)", segments[0]).group(1))
        rule_text = sent_map[rule_idx]

        # 4) Extract selected premise indices from segment[1]
        sel_indices = [int(i) for i in re.findall(r"sent (\d+)", segments[1])]
        facts = [sent_map[i] for i in sel_indices]

        # 5) Build inference prompt
        inf_prompt = rule_text + " We know that " + " and ".join(facts)

        # 6) Build inference target
        if len(segments) >= 3:
            inf_t = segments[2]
        else:
            # Fallback: take text after last period in target
            inf_t = target.split(".")[-1].strip()
        # Ensure it starts with "Therefore,"
        if not inf_t.startswith("Therefore"):
            inf_t = "Therefore, " + inf_t
        # Ensure it ends with a period
        if not inf_t.endswith("."):
            inf_t += "."

        inf_examples.append({
            "prompt": inf_prompt,
            "target": inf_t
        })

    return inf_examples

def save_inference_data(inf_examples, path="inference_train.json"):
    with open(path, "w") as f:
        json.dump(inf_examples, f, indent=2)

def main():
    sel_examples = load_selection_data("selection_train.json")
    inf_examples = convert_to_inference(sel_examples)
    save_inference_data(inf_examples, "inference_train.json")
    print(f"Converted {len(sel_examples)} selection examples to inference examples.")

if __name__ == "__main__":
    main()