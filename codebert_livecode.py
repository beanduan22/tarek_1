import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 确保输出目录存在
output_dir = "codebert_livecode_outputs"
os.makedirs(output_dir, exist_ok=True)

# 加载 LiveCodeBench 数据集（code_generation_lite）
dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", trust_remote_code=True)
problems = dataset["test"]  # 选择 test 数据集

# 加载 CodeBERT 模型
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)  # 限制最大长度
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 遍历数据集并调用 CodeBERT
for problem in problems:
    problem_id = problem.get("question_id", "Unknown")
    content = problem.get("question_content", "No content")

    # 确保文件名合法（防止文件系统错误）
    safe_problem_id = str(problem_id).replace("/", "_")
    file_name = os.path.join(output_dir, f"{safe_problem_id}.txt")

    # Tokenize 输入
    inputs = tokenizer(content, truncation=True, max_length=512, return_tensors="pt")

    try:
        # 进行 CodeBERT 推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取 logits 作为输出
        logits_str = str(outputs.logits.detach().cpu().numpy().tolist())

        # 将 CodeBERT 的输出保存到文件
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"Problem ID: {problem_id}\n")
            f.write(f"Problem Content:\n{content}\n\n")
            f.write(f"CodeBERT Logits:\n{logits_str}\n")

    except Exception as e:
        print(f"An error occurred with problem ID {problem_id}: {str(e)}")

print("CodeBERT processing completed. Results saved in 'codebert_livecode_outputs/' directory.")
