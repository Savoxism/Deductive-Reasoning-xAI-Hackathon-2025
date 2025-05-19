import re
import os

from vllm import SamplingParams
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login

API_KEY = "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"
login(token=API_KEY)

max_seq_length = 2048
lora_rank = 64
SEED = 3407
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=False,# Turn off quantization to increase accuracy for reasoning
    fast_inference=True, # optimize throughput
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=SEED
)


# ====================================================================== #
DATASET_PATH = "5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated"
dataset = load_dataset(DATASET_PATH, split='train')
# ====================================================================== #

answer_pattern_en = re.compile(
    r"(?:the answer is:|answer:)\s*(.*)",
    re.IGNORECASE
)

formatted_dataset = []
for item in dataset:
    response = item['response_en'].strip().lower()
    match = answer_pattern_en.search(response)
    if match:
        answer = match.group(1).strip()
        formatted_dataset.append({
            "question": item['query_en'],
            "answer": answer,
        })

reasoning_start = "<thinking>"
reasoning_end   = "</thinking>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
    f"""You are given a problem.
Think about the problem and provide your thought process.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your final answer between {solution_start}{solution_end}"""

train_dataset = Dataset.from_list(formatted_dataset[:8000])
train_dataset = train_dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x['question']},
    ],
    "answer": x["answer"],
})

# Reward for correct formatting
match_format = re.compile(rf"""
    ^\s*                              # bất kỳ khoảng trắng đầu dòng
    {re.escape(reasoning_start)}     # <thinking>
    .*?                               # chain-of-thought (non-greedy)
    {re.escape(reasoning_end)}        # </thinking>
    .*?                               # có thể có text khác giữa
    {re.escape(solution_start)}       # <SOLUTION>
    (.+?)                             # nhóm 1: nội dung solution
    {re.escape(solution_end)}         # </SOLUTION>
    \s*                               # optional trailing whitespace
    $                                 # kết thúc chuỗi
""", flags=re.DOTALL | re.MULTILINE | re.VERBOSE)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]['content']
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]['content']
        # mỗi tag đúng một lần thì +0.5, thiếu hoặc lặp lại thì -1.0
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores

# Reward for correct answer
match_solution = re.compile(
    rf"{re.escape(solution_start)}\s*(.+?)\s*{re.escape(solution_end)}",
    flags=re.DOTALL
)

def check_answer(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [
        m.group(1).strip() if (m := match_solution.search(r)) else None
        for r in responses
    ]
    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0); continue
        scores.append(
            3.0 if guess == true
            else 1.5 if guess.strip() == true.strip()
            else -1.5
        )
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [
        m.group(1).strip() if (m := match_solution.search(r)) else None
        for r in responses
    ]
    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0); continue
        try:
            t = float(true.replace(",", ""))
            g = float(guess.replace(",", ""))
            scores.append(1.5 if g == t else -0.5)
        except:
            scores.append(0)
    return scores

# ====================================================================== #
max_len = max(train_dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(
        x['prompt'], add_generation_prompt=True, tokenize=True)},
    batched=True,
).map(lambda x: {"length": len(x['tokens'])})['length'])

max_prompt_length = max_len + 1

training_args = GRPOConfig(
    # Diagnostics
    report_to= None,
    output_dir="output_bz2",
    logging_steps=1,
    logging_dir="output_bz2/logs",  # thư mục chứa TensorBoard logs
    run_name  = "grpo-run1",

    # Optimization
    learning_rate=5e-6,
    weight_decay=5e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim='adamw_torch_fused',
    max_grad_norm=0.1,

    # Batch
    per_device_train_batch_size=8,
    gradient_accumulation_steps=32,

    # Specific settings
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=1,
    max_steps=-1,
    save_steps=50,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

print(trainer.state.global_step, "/", trainer.state.max_steps)
print("Train time:", trainer.state.log_history[-1]["train_runtime"])

model.save_lora("grpo_saved_lora")
