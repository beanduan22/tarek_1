import ollama
import os
import json
from human_eval.data import write_jsonl, read_problems

# 目录确保存在
output_dir = "codestral_outputs"
os.makedirs(output_dir, exist_ok=True)


# 定义 Ollama 调用函数（改为 codestral）
def call_codestral(prompt):
    response = ollama.chat(
        model="codestral",  # 使用 Codestral 模型
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# 读取 HumanEval 数据
problems = read_problems()

# 处理每个问题并调用 Codestral 进行推理
for problem_id, problem_data in problems.items():
    task_id = problem_data.get("task_id", f"unknown_{problem_id}")

    # **确保子目录存在**
    safe_task_id = task_id.replace("/", "_")  # 避免 Windows/Linux 路径问题
    file_name = os.path.join(output_dir, f"{safe_task_id}.txt")

    # 生成 Codestral 适用的 Prompt
    prompt = f"""
    Here is a programming problem:
    {problem_data["prompt"]}

    Analyze the problem and predict its category based on difficulty, required programming skills, and complexity.
    """

    try:
        # 调用 Codestral API
        output = call_codestral(prompt)

        # 将 Codestral 的输出保存到文件
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    except Exception as e:
        print(f"An error occurred with problem ID {task_id}: {str(e)}")

print("Codestral processing completed. Results saved in 'codestral_outputs/' directory.")
