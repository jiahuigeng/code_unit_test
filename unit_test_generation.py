import os
import argparse
import pandas as pd
from utils_llm import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *

file_ext_map = {
    "Python": "py",
    "C++": "cpp",
    "C": "c",
    "Java": "java",
    "JavaScript": "js",
    "Ruby": "rb",
    "Rust":"rs",
    "Go":"go",
}
def get_python_prompt_unit_test_generation(description, code):
    prompt = f"""
Generate comprehensive unit tests for the given code using the provided problem description.

Requirements:
- Create exactly 10 unit tests.
- Each test must be written as a separate test function.

Inputs:
- PROBLEM DESCRIPTION: 
{description}
- CODE TO TEST: 
{code}

Guidelines for the tests:
- Use 'unittest' module.
- Mock the sys.stdin and sys.stdout appropriately for the given code.
- Use runpy.run_path() to run the submission code instead of importing it. The path of the given code is 'submission_code.py', therefore this path must be run. Do not run any other path or try to mock this file.
- Provide the complete runnable Python code for the test module.
- Do not add any explanation or commentary before or after the output.

Here is an example of mocking sys.stdin and sys.stdout:
def test_sample_input_1(self):
        user_input = "example_input"
        expected_output = "expected_output"
        with patch('sys.stdin', io.StringIO(user_input)), patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            runpy.run_path('submission_code.py')
            self.assertEqual(mock_stdout.getvalue(), expected_output)
- If there are multiple answers or there is an allowed precision described in the description for the output, do not use the exact format above, use appropriate assertion method logic that checks if the answer is within the allowed range, or within the possible answers.
- Make sure to follow the input pattern in the problem description. E.g. if the end of the input is denoted by 0 or # or any similar character, make sure to add it to the input string in the test case.
"""
    return prompt


def get_java_prompt_unit_test_generation(description, code):
    prompt = f"""
Generate comprehensive unit tests for the code given using the provided problem description.

Requirements:
- Create exactly 10 unit tests.
- Each test must be written as a separate test function.

Inputs:
- PROBLEM DESCRIPTION: 
{description}
- CODE TO TEST: 
{code}

Guidelines for the tests:
- Use JUnit 5 library.
- Use the class name: MainTest.
- You are given a function called getTestOutput which gets the code output and it will be implemented later. For now, add this function to test code exactly like this inside the MainTest class:
private String getTestOutput(String input) throws Exception {{
    return 'This function will be implemented later.';
}}
- Remember, do not implement the getTestOutput function. Just add it to the test code as it is given above.
- Create tests like this if there is only a single answer for the input:
@Test
public void testSampleInput1() throws Exception {{
    String input = "example input 1";
    String expectedOutput = "example output 1";
    String result = getTestOutput(input);

    assertEquals(expectedOutput.trim(), result.trim());
}}
- If there are multiple answers or there is an allowed precision described in the description for the output, do not use the exact format above, use appropriate assertion method logic that checks if the answer is within the allowed range, or within the possible answers.
- Make sure to follow the input pattern in the problem description. E.g. if the end of the input is denoted by 0 or # or any similar character, make sure to add it to the input string in the test case.

Output instructions:
- Do not add any explanation or commentary before or after the test code.
"""
    return prompt

def func_generation_prompt(lang):
    if lang == "Python":
        return get_python_prompt_unit_test_generation
    elif lang == "Java":
        return get_java_prompt_unit_test_generation
def save_txt(save_path, content):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

def process_row(idx, row, args, client, save_folder, col_name):
    # if not pd.isna(row[col_name]):
    #     return idx, None  # 已处理，跳过

    pid, submission_id = row["problem_id"], row["submission_id"]
    save_path = f"{save_folder}/{pid}_{submission_id}_{args.model}.txt"

    if os.path.exists(save_path) and "rate_limit_error" not in open(save_path).read():
        return idx, save_path  # 文件已存在，跳过生成

    desc_path = f"thesis_dataset/problem_descriptions/{pid}.html"
    try:
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc_content = f.read()
    except FileNotFoundError:
        print(f"{desc_path} not found")
        return idx, None

    ext = file_ext_map[args.lang]
    wrapped_code_path = f"thesis_dataset/data/{pid}/{args.lang}/{submission_id}.{ext}"
    if not os.path.exists(wrapped_code_path):
        print(f"{wrapped_code_path} not exists")
        return idx, None

    with open(wrapped_code_path, 'r', encoding='utf-8') as f:
        code_content = f.read()

    prompt = func_generation_prompt(args.lang)(desc_content, code_content)
    resp = prompt_commercial_model(client, args.model, prompt, image_id="")
    save_txt(save_path, resp)

    return idx, save_path

def main(args):
    client = get_commercial_model(args.model)
    multi_thread = False
    MAX_THREADS = 8
    if args.model in ["gpt-4o", "gemini-2.0-flash", "gpt-4o-mini", "claude-3-haiku"]:
        multi_thread = True
    ext = file_ext_map[args.lang]

    # lang_map_xlsx = pd.read_excel(f"datasets/full_{args.lang}.xlsx")

    save_folder = f"thesis_dataset/generated/{args.lang}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    for status in ["Accepted", "Wrong_Answer"]:
        save_file = f"thesis_dataset/generated/{args.lang}/{args.lang}_{status}_res.xlsx"
        if not os.path.exists(save_file):
            src_file = f"thesis_dataset/splits/{args.lang}_{status}.csv"
            df = pd.read_csv(src_file)
            df.to_excel(save_file, index=False)

        df_input = pd.read_excel(save_file)
        col_name = f"{args.model}_path"
        if args.model not in df_input.columns:
            init_cols(df_input, [col_name])
        args.col_name = col_name

        if multi_thread:

            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = {
                    executor.submit(process_row, idx, row, args, client, save_folder, col_name): idx
                    for idx, row in df_input.iterrows()
                }

                for future in as_completed(futures):
                    idx, save_path = future.result()
                    if save_path:
                        df_input.at[idx, col_name] = save_path

            df_input.to_excel(save_file)

        else:

            for idx, row in df_input.iterrows():
                # if not pd.isna(row[col_name]):
                #     continue
                # if idx > 1:
                #     break
                pid, submission_id = row["problem_id"], row["submission_id"]
                save_path = f"{save_folder}/{pid}_{submission_id}_{args.model}.txt"
                if os.path.exists(save_path)  and "rate_limit_error" not in open(save_path).read():
                    df_input.at[idx, col_name] =save_path
                    continue
                desc_path = f"thesis_dataset/problem_descriptions/{pid}.html"
                desc_content = open(desc_path).read()
                wrapped_code_path = f"thesis_dataset/data/{pid}/{args.lang}/{submission_id}.{ext}"
                if not os.path.exists(wrapped_code_path):
                    print(f"{wrapped_code_path} not exists")
                    continue
                else:
                    code_content = open(wrapped_code_path).read()

                    prompt = func_generation_prompt(args.lang)(desc_content, code_content)

                    resp = prompt_commercial_model(client, args.model, prompt, image_id="")

                    save_txt(save_path, resp)
                    df_input.at[idx, col_name] = save_path

            # df_input.to_excel(save_file)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Python")
    # gemini - 2.0 - flash
    parser.add_argument("--model", type=str, default="claude-3-haiku")
    # parser.add_argument("--debug", type=str, default="gpt4o")
    args = parser.parse_args()

    main(args)

