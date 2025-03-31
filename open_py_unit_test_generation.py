import os
import argparse
import pandas as pd
import time
from utils_openllm import *
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



def main(args):
    # client = get_commercial_model(args.model_name)
    model, tokenizer = get_open_model(args.model_name, quantization='4bit')
    multi_thread = False
    MAX_THREADS = 8
    if args.model_name in ["gpt4o", "gemini"]:
        multi_thread = True
    ext = file_ext_map[args.lang]

    lang_map_xlsx = pd.read_excel(f"datasets/full_{args.lang}.xlsx")
    save_folder = f"generated/{args.lang}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    
    for idx, row in lang_map_xlsx.iterrows():
        print(idx)
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
            
            try:
                start_time = time.time()
                prompt = get_llm_prompt_python_generation(desc_content, neg_code_content)
                print("run time:", time.time() - start_time)
            except:
                prompt = "Fail to generate code"

            resp = prompt_open_model(args.model_name, model, tokenizer, prompt)
            print(resp)
            save_file(save_path, resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Python")
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()

    main(args)

