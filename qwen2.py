import ollama
import os
import json
from human_eval.data import write_jsonl, read_problems

# 目录确保存在
os.makedirs("qwen2_outputs", exist_ok=True)


# 定义 Ollama 调用函数
def call_qwen2(prompt):
    response = ollama.chat(
        model="qwen2",  # 使用 Qwen2 模型
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# 读取 HumanEval 数据
problems = read_problems()

# 处理每个问题并调用 Qwen2 进行推理
for problem_id, problem_data in problems.items():
    task_id = problem_data.get("task_id", f"unknown_{problem_id}")

    # **确保子目录存在**
    task_dir = os.path.join("qwen2_outputs", os.path.dirname(task_id))
    os.makedirs(task_dir, exist_ok=True)  # **新增：确保子目录存在**

    file_name = os.path.join("qwen2_outputs", f"{task_id}.txt")

    # 生成 Qwen2 适用的 Prompt
    prompt = f"""
    Here is a programming problem:
    {problem_data["prompt"]}

    Analyze the problem and predict its category based on difficulty, required programming skills, and complexity.
    """

    try:
        # 调用 Qwen2 API
        output = call_qwen2(prompt)

        # 将 Qwen2 的输出保存到文件
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    except Exception as e:
        print(f"An error occurred with problem ID {task_id}: {str(e)}")

print("Qwen2 processing completed. Results saved in 'qwen2_outputs/' directory.")
