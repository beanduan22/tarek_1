import ollama
import os
from datasets import load_dataset

# 确保输出目录存在
output_dir = "qwen2_livecode_outputs"
os.makedirs(output_dir, exist_ok=True)

# 加载 LiveCodeBench 数据集（code_generation_lite）
dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", trust_remote_code=True)
problems = dataset["test"]  # 选择 test 数据集

# 定义 Ollama 调用函数
def call_qwen2(prompt):
    response = ollama.chat(
        model="qwen2",  # 使用 Qwen2 模型
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# 遍历数据集并调用 Qwen2
for problem in problems:
    problem_id = problem.get("question_id", "Unknown")
    content = problem.get("question_content", "No content")

    # 确保文件名合法（防止文件系统错误）
    safe_problem_id = str(problem_id).replace("/", "_")
    file_name = os.path.join(output_dir, f"{safe_problem_id}.txt")

    # 生成 Qwen2 适用的 Prompt
    prompt = f"""
    Here is a programming problem:
    {content}

    Analyze the problem and predict its category based on difficulty, required programming skills, and complexity.
    """

    try:
        # 调用 Qwen2 API
        output = call_qwen2(prompt)

        # 将 Qwen2 的输出保存到文件
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"Problem ID: {problem_id}\n")
            f.write(f"Problem Content:\n{content}\n\n")
            f.write(f"Qwen2 Analysis:\n{output}\n")

    except Exception as e:
        print(f"An error occurred with problem ID {problem_id}: {str(e)}")

print("Qwen2 processing completed. Results saved in 'qwen2_outputs/' directory.")
