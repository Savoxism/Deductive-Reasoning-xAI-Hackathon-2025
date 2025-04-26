import os
import json
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2. Hằng số
MODEL_NAME     = "gemini-2.5-flash-preview-04-17"
OUTPUT_FILE    = "train_examples.json"
NUM_EXAMPLES   = 1000
ONE_SHOT       = {
    "prompt": (
        "sent 0: If something is rough then it visits the lion.\n"
        "sent 1: the cow visits the lion.\n"
        "sent 2: the lion visits the cow.\n"
        "sent 3: the cow is rough.\n"
        "sent 4: If something visits the cow and the cow visits the lion then the lion is nice.\n"
        "sent 5: If something is nice it visits the lion.\n"
        "Does it imply that the statement \"The lion visits the lion\" is True?\n"
        "Selection:"
    ),
    "target": "sent 4. We know that sent 2 and sent 1. Therfore, the lion is nice"
}

def load_examples():
    if os.path.exists(OUTPUT_FILE):
        return json.load(open(OUTPUT_FILE))
    return []

def save_examples(exs):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(exs, f, indent=2)

def generate_example():
    system = (
        "You are a training data generator for the Selection-Inference Selection module.\n"
        "Produce exactly one JSON object with two keys:\n"
        "  \"prompt\": full prompt with numbered facts + question + 'Selection:'\n"
        "  \"target\": only the sentence labels like 'sent 2. We know that sent 0 and sent 1. Therefore, {inference}'\n"
        "Think of the best inference. Follow this one-shot example:\n\n"
        + json.dumps(ONE_SHOT, indent=2)
    )
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    resp  = model.generate_content(contents=system)
    text  = resp.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group())

def main():
    examples = load_examples()
    for i in range(NUM_EXAMPLES):
        try:
            new_ex = generate_example()
            examples.append(new_ex)
            save_examples(examples)
            print(f"[{i+1}/{NUM_EXAMPLES}] Appended and saved: {new_ex}")
        except Exception as e:
            print(f"Error on example {i+1}: {e}")
        # time.sleep(1)  # small delay to avoid rate limits

if __name__ == "__main__":
    main()

