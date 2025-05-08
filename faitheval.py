import os
import re
import string
from datasets import load_dataset
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# normalize answer function

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "‘’´`")
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    replace_underscore(s)
                )
            )
        )
    ).strip()

# Config
model_name = "gpt-3.5-turbo"
strict_match = False

# Choose dataset
dataset_name = "Salesforce/FaithEval-inconsistent-v1.0"
dataset = load_dataset(dataset_name, split="test").select(range(10))

# Define valid phrases
if not strict_match:
    valid_phrases = [
        'conflict','multiple answers','disagreement',
        'inconsistent','contradictory','contradiction',
        'inconsistency','two answers','2 answers','conflicting'
    ]
else:
    valid_phrases = ['conflict']

# Task prompt
task_specific_prompt = (
    "If there is conflict information or multiple answers from the context, "
    "the answer should be 'conflict'."
)

# Evaluation loop
correct = 0
for example in tqdm(dataset, desc="Processing examples"):
    prompt = (
        "You are an expert in retrieval question answering.\n"
        "Please respond with the exact answer only. Do not be verbose or provide extra information.\n"
        f"{task_specific_prompt}\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )

    # Call ChatGPT
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=256,
        temperature=0
    )
    pred_answer = resp.choices[0].message.content.strip()
    print(pred_answer, "\n")

    # Check correctness
    if any(phrase in normalize_answer(pred_answer) for phrase in valid_phrases):
        correct += 1

# Print accuracy
print(f"Accuracy: {correct / len(dataset):.2f}")
