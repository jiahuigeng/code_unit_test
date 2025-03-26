import os
import argparse
import pandas as pd
from utils_llm import *
from concurrent.futures import ThreadPoolExecutor, as_completed

file_ext_map = {
    "Python": "py",
    "C++": "cpp",
    "C": "c",
    "Java": "java",
    "JavaScript": "js",
    "Ruby": "rb",
    "Rust": "rs",
    "Go": "go",
}


def get_llm_prompt_python_generation(description, code):
    prompt = f"""
    Generate comprehensive unit tests for the given function using the provided problem description.

    Requirements:
    - Create exactly 10 unit tests.
    - Each test must be written as a separate test function.

    Inputs:
    - PROBLEM DESCRIPTION: 
    {description}
    - FUNCTION TO TEST: 
    {code}

    Guidelines for the tests:
    - Use either 'unittest'.
    - Mock the 'input()' function to supply test data.
    - Provide the complete runnable Python code for the test module.
    - Use multiline strings (triple-quoted) instead of newline characters when defining multiline string variables.

    Output instructions:
    - Do not include the full implementation of the function being tested in the output.
    - Instead, import it from an external file using the syntax: 'from function_file import function_under_test'
    - Do not add any explanation or commentary before or after the output.
    """
    return prompt


def func_generation_prompt(lang):
    if lang == "Python":
        return get_llm_prompt_python_generation


def save_file(save_path, content):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def process_row(row, args, client, save_folder):
    pid, neg_submission_id = row["pid"], row["neg_submission_id"]
    save_path = f"{save_folder}/{pid}_{neg_submission_id}_{args.model_name}.txt"
    if os.path.exists(save_path):
        return

    desc_path = row["desc_path"]
    if not os.path.exists(desc_path):
        print(f"{desc_path} not exists")
        return

    desc_content = open(desc_path).read()
    wrapped_code_path = f"wrapped_code_dataset/{pid}_{neg_submission_id}.py"
    if not os.path.exists(wrapped_code_path):
        print(f"{wrapped_code_path} not exists")
        return

    neg_code_content = open(wrapped_code_path).read()
    prompt = func_generation_prompt(args.lang)(desc_content, neg_code_content)
    resp = prompt_commercial_model(client, args.model_name, prompt, image_id="")
    save_file(save_path, resp)


def main(args):
    client = get_commercial_model(args.model_name)
    multi_thread = False
    MAX_THREADS = 8
    if args.model_name in ["gpt4o", "gemini"]:
        multi_thread = True
    ext = file_ext_map[args.lang]

    lang_map_xlsx = pd.read_excel(f"datasets/full_{args.lang}.xlsx")
    save_folder = f"generated/{args.lang}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if multi_thread:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            for idx, row in lang_map_xlsx.iterrows():
                print(idx)
                # 改这里可以控制处理的范围
                # if idx > 0:
                #     break
                futures.append(executor.submit(process_row, row, args, client, save_folder))

            # 可选：等待所有任务完成，并打印异常
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread: {e}")
    else:
        for idx, row in lang_map_xlsx.iterrows():
            pid, neg_submission_id = row["pid"], row["neg_submission_id"]
            save_path = f"{save_folder}/{pid}_{neg_submission_id}_{args.model_name}.txt"
            if os.path.exists(save_path):
                continue
            desc_path = row["desc_path"]
            desc_content = open(desc_path).read()
            wrapped_code_path = f"wrapped_code_dataset/{pid}_{neg_submission_id}.{ext}"
            if not os.path.exists(wrapped_code_path):
                print(f"{wrapped_code_path} not exists")
                continue
            else:
                neg_code_content = open(wrapped_code_path).read()

                prompt = get_llm_prompt_python_generation(desc_content, neg_code_content)

                resp = prompt_commercial_model(client, args.model_name, prompt, image_id="")

                save_file(save_path, resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Python")
    parser.add_argument("--model_name", type=str, default="gpt4o")
    # parser.add_argument("--debug", type=str, default="gpt4o")
    args = parser.parse_args()

    main(args)

