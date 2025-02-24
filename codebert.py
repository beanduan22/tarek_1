from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)  # Set maximum token length
model = AutoModelForSequenceClassification.from_pretrained(model_name)

problems = read_problems()

for problem in problems:
    file_name = "codebert/" + problems[problem]['task_id'] + ".txt"
    # Truncate or split the text to handle length appropriately
    inputs = tokenizer(problems[problem]['prompt'], truncation=True, max_length=512, return_tensors="pt")
    try:
        outputs = model(**inputs)
        logits_str = str(outputs.logits.detach().cpu().numpy().tolist())
        with open(file_name, "w", encoding='utf-8') as f:
            f.write(logits_str)
    except Exception as e:
        print(f"An error occurred with problem ID {problems[problem]['task_id']}: {str(e)}")