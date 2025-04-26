from together import Together
from dotenv import load_dotenv
import os

load_dotenv()
client = Together()

# TOGETHER_API_KEY = "a97c5b69c97abb798cdc7e22e1b3fb7c270945a9ee016fcefba412c7127714b1"

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")

LOOP = 3
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
    "You are the VERIFICATION module. Given some premises and a yes/no question,\n"
    "Select ONLY ONE premise that is most relevant to answering the question.\n"
    "Then, based on that premise alone, answer strictly with 'Yes' or 'No'.\n"
    "Do NOT explain. Do NOT output anything else. Only reply with 'Yes' or 'No'.\n"
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

premises = [
  "Learning through play enhances engagement and retention.",
            "Gamification elements such as rewards and challenges increase motivation.",
            "If learners actively participate in simulations, they develop problem-solving skills.",
            "Digital learning platforms provide personalized feedback, which improves learning outcomes.",
            "Collaborative online environments foster knowledge sharing and deeper understanding.",
            "If a student interacts with diverse cultural content, they develop a broader worldview.",
            "Augmented reality (AR) and virtual reality (VR) make abstract concepts more tangible.",
            "If learners set their own goals, they are more likely to persist in their studies.",
            "Immediate feedback helps correct misunderstandings before they become ingrained.",
            "If digital learning is well-structured, it can be as effective as traditional classroom learning.",
            "If a learning environment is interactive, students are more likely to engage with the material.",
            "Social learning enhances knowledge retention by encouraging discussion and collaboration.",
            "If educational content is presented in multiple formats, comprehension improves.",
            "If a game-based learning platform adapts to a student's progress, they will remain motivated.",
            "If learners experience real-world applications of knowledge, they are more likely to retain it.",
            "Repetitive practice in a digital setting reinforces learning through spaced repetition.",
            "Peer feedback can help learners identify and correct their own mistakes.",
            "If a student reflects on their learning process, they develop metacognitive skills.",
            "Students who receive constructive feedback are more likely to improve their performance.",
            "Engaging digital experiences reduce cognitive overload, making learning more efficient."
]

question =  "If a student engages in play-based learning, participates in simulations, receives immediate and constructive feedback, and reflects on their learning, they will retain knowledge more effectively and develop problem-solving skills. Is this statement true?"

for iteration in range(LOOP):
    print(f"Iteration {iteration+1}:")

    ctx_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    
    # SELECTION MODULE
    selection_user = f"\nContext:\n{ctx_text}\nQuestion: {question}\nSelection:\n"
    
    selection_message = SELECTION_TEMPLATE + [{"role": "user", "content": selection_user}]
    
    selection_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=selection_message,
    )
    
    selection_text = selection_response.choices[0].message.content.strip()
        
    print("Selected premises:", selection_text)
    
    # INFERENCE MODULE
    inference_user = f"\nSelection:\n{selection_text}\nInference:\n"
    
    inference_message = INFERENCE_PROMPT + [{"role": "user", "content": inference_user}]
    
    inference_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=inference_message,
    )
    
    new_fact = inference_response.choices[0].message.content.strip()
    
    premises.append(new_fact)

    print("Inferred new fact:", new_fact)
    print("END OF ITERATION\n")

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
print("\nFinal answer:", final_answer)

