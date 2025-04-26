from together import Together
from dotenv import load_dotenv
import os
import re

load_dotenv()
client = Together()

# TOGETHER_API_KEY = "a97c5b69c97abb798cdc7e22e1b3fb7c270945a9ee016fcefba412c7127714b1"

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

LOOP = 2
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-Turbo"

# ================================= SYSTEM PROMPT ================================= #
selection_instructions = """
You are the SELECTION module in a Selection-Inference pipeline.
First, you will see 5 examples in this format:

Context:
1. <premise 1>
2. <premise 2>
...
Question: <question text>
Selection:
- <premise number>. <premise text>
- ...

Then you will be given the real Context and Question.
Your task:
    1. Choose the minimal subset of the numbered premises, at most 3-4 premises, that is strictly necessary to perform the very next inference step toward answering the final question.
    2. Only select from the provided premises; do not invent new facts.
    3. Respond with one line per selected premise, in the form:
        - <premise number>. <premise text>
""" 

inference_instructions = """
You are the INFERENCE module in a Selection-Inference pipeline.
First, you will see 5 examples:

Selection:
- <premise text>
- ...
Inference:
Therefore, <new fact>.

Then you will be given the real Selection.
Your task is to produce exactly one new inference fact, phrased as a single sentence beginning with "Therefore,".
"""

verification_instructions = (
"""You are the VERIFICATION module. Based only on the premises and the question, determine which of the three question types it is, and reply with exactly one answer as follows:

1. Yes/No/Uncertain: If the question asks for a binary or three-state factual judgment, reply exactly with Yes, No, or Uncertain.  

2. Multiple Choice: If the question offers options labeled A, B, C, or D, reply exactly with one of A, B, C or D.  

3. Enumeration: If the question asks you to list or enumerate items (including filling in blanks or providing numbers), reply with a comma-separated list of the expected entries (e.g. `item1, item2, item3`).  

Do NOT include any explanations, punctuation beyond commas in enumeration, or extra text—only the precise answer as specified above.
"""
)

# ================================= PROMPT TEMPLATE ================================= #
SELECTION_TEMPLATE = [
    {"role": "system", "content": selection_instructions},
    
    # Shot 1
    {"role": "user", "content": "Context: Here are some facts and rules:\n1.Nice people are quiet.\n2. If Dave is smart then Dave is nive.\n 3. All white people are smart.\n4. Dave is smart.\n5. Harry is cold.\nQuestion: Does it imply that the statement \"Dave is not quiet\" is true?\nSelection:"},
    {"role": "assistant", "content": "If Dave is smart then Dave is nice (premise 2). We know that Dave is smart (premise 4)"},
    
    # Shot 2
    {"role": "user", "content": "Context: Here are some facts and rules:\n1. Blue things are green.\n2. All blue things are white.\n3. If Anne is not big then Anne is blue.\n4. Big things are white.\n5. All kind things are round.\n6. If something is white and big then it is not kind.\n7. If something is big and not rough then it is green.\n8. If something is white and blue then it is not green.\n9. Erin is not white.\n10. Anne is big.\n11. Bob is rough.\n12. Anne is white.\nQuestion: Does it imply that the statement \"Anne is kind\" is true?\nSelection:"},
    {"role": "assistant", "content": "If something is white and big then it is not kind (premise 6). We know that Anne is white and Anne is big (premise 10 & 12)"},
    
    # Shot 3
    {"role": "user", "content": "Context: Here are some facts and rules:\n1. If a person is over 18 and has a valid driver's license, then they can rent a car.\n2. If a person can rent a car and has travel insurance, they can drive abroad.\n3. John is 20 years old.\n4. John has a valid driver's license.\nQuestion: Does it imply that the statement \"John can rent a car\" is true?\nSelection:"},
    {"role": "assistant", "content": "If a person is over 18 and has a valid driver's license, then they can rent a car (premise 1). We know that John is 20 years old (premise 3) and John has a valid driver's license (premise 4)"},
    
    # Shot 4
    {"role": "user", "content": "Context: Here are some facts and rules:\n1. If a plant receives sufficient sunlight and water, it will grow.\n2. If a plant grows, it can produce flowers.\n3. The rose plant receives sufficient sunlight.\n4. The rose plant receives sufficient water.\nQuestion: Does it imply that the statement \"The rose plant will grow\" is true?\nSelection:"},
    {"role": "assistant", "content": "If a plant receives sufficient sunlight and water, it will grow (premise 1). We know that the rose plant receives sufficient sunlight (premise 3) and the rose plant receives sufficient water (premise 4)"},
    
    # Shot 5
    {"role": "user", "content": "Context: Here are some facts and rules:\n1. If a user enters correct credentials and completes the captcha, they get account access.\n2. If a user gets account access and has two-factor enabled, they can perform transactions.\n3. Alice entered correct credentials.\n4. Alice completed the captcha.\n5. Alice has two-factor enabled.\nQuestion: Does it imply that the statement \"Alice gets account access\" is true?\nSelection:"},
    {"role": "assistant", "content": "If a user enters correct credentials and completes the captcha, they get account access (premise 1). We know that Alice entered correct credentials (premise 3) and Alice completed the captcha (premise 4)"},
]

INFERENCE_PROMPT = [
    {"role": "system", "content": inference_instructions},
    
    # Shot 1
    {"role": "user", "content": "Selection: Nice people are quiet. We know that Dave is nice.\nInference:"},
    {"role": "assistant", "content": "Therefore, Dave is quiet."},
    
    # Shot 2
    {"role": "user", "content": "Selection: If someone is younf then they are energetic. We know that Alice is young.\nInference:"},
    {"role": "assistant", "content": "Therefore, Alice is energetic."},
    
    # Shot 3
    {"role": "user", "content": "Selection: If a number is divisible by 3 then it is a factor of 6. We know that 6 is divisible by 3.\nInference:"},
    {"role": "assistant", "content": "Therefore, 6 is a factor of 6."},
    
    # Shot 4
    {"role": "user", "content": "Selection: If a plant receives sufficient sunlight and water, it will grow. We know that the rose plant receives sufficient sunlight and the rose plant receives sufficient water.\nInference:"},
    {"role": "assistant", "content": "Therefore, the rose plant will grow."},
    
    # Shot 5
    {"role": "user", "content": "Selection: If a user enters correct credentials and completes the captcha, they get account access. We know that Alice entered correct credentials and Alice completed the captcha.\nInference:"},
    {"role": "assistant", "content": "Therefore, Alice gets account access."},
]

# ================================= TESTING  ================================= #
def score_premise(premise: str, ctx_text: str, question: str, current_sel: str):
    messages = SELECTION_TEMPLATE + [
        {"role": "user", "content": f"\nContext:\n{ctx_text}\nQuestion: {question}\nSelection:\n{current_sel}{premise}"}
    ]
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        logprobs=1,
        echo=True,
        max_tokens=0,
    )
    
    token_logprobs = resp.choices[0].logprobs.token_logprobs
    return sum(t for t in token_logprobs if t is not None)

premises = [
"If a cloud service is not scalable, then it is not reliable.",
"All cloud services are scalable.",
"If a cloud service does not meet performance standards, then it does not adhere to security protocols.",
"All cloud services are thoroughly tested.",
"If a cloud service meets performance standards, then it is cost-effective.",
"If a cloud service is not thoroughly tested, then it does not meet performance standards.",
"If not being thoroughly tested implies that a cloud service does not meet performance standards, then there exists at least one cloud service that is reliable.",
"If all cloud services are scalable, then all cloud services are thoroughly tested.",
"If a cloud service is thoroughly tested, then it is reliable.",
"All cloud services meet performance standards.",
"If a cloud service meets performance standards, then it is thoroughly tested.",
"If a cloud service does not meet performance standards, then it is not cost-effective."
]

question = "Are all cloud services cost-effective?"

NUM_SELECTIONS = 2
selected_premises_history = set()

for iteration in range(LOOP):
    ctx_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    
    # SELECTION MODULE 
    selections: list[str] = []
    remaining = premises.copy()
    current_selection_str = ""
    
    for _ in range(NUM_SELECTIONS):
        best_score = -float("inf")
        best_premise = None
        
        for p in remaining:
            score = score_premise(p, ctx_text, question, current_selection_str)
            if score > best_score:
                best_score, best_premise = score, p
        
        current_selection_str += f"{best_premise}. "
        clean = best_premise.strip().rstrip(".") + "."
        selections.append(clean)
        remaining.remove(best_premise)

    selected_premises_history.add(tuple(selections))
    
    # INFERENCE MODULE
    inference_user = (
        "Selection:\n" +
        "\n".join(f"- {s}" for s in selections) +
        "\nInference:\n"
    )
    messages = INFERENCE_PROMPT + [
        {"role": "user", "content": inference_user}
    ]
    inf_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0
    )
    raw = inf_resp.choices[0].message.content.strip()
    new_fact = re.match(r"[^.?!;\n]+", raw).group(0).strip()
    premises.append(new_fact)
    

# VERIFICATION MODULE
ctx_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
verification_input = (
    f"Context:\n{ctx_text}\nQuestion: {question}\nAnswer:\n"
)
VERIFICATION_TEMPLATE = [
    {"role": "system", "content": verification_instructions},
    {"role": "user", "content": verification_input}
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=VERIFICATION_TEMPLATE,
)

final_answer = response.choices[0].message.content.strip()

# FINAL ANSWER
for idx, sels in enumerate(selected_premises_history , start=1):
    for p in sels:
        print(p)
print(final_answer)

