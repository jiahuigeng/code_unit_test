import sys
import argparse
import subprocess
import stat
import os
from pathlib import Path
import re
import json
from xml.etree import ElementTree as ET
from utils_llm import *
from multiprocessing import Pool, Process
import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CXX_FLAGS = ["-Wno-write-strings",
             "-Wno-return-type",
             "-pthread",
             "-std=c++20",
             "-fprofile-arcs", "-ftest-coverage", "-g",
             "-I", "/usr/local/include",
             "-L", "/usr/local/lib",
             ]
GCC = "g++"


def get_llm_prompt_unit_test_generation(description, code):
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
- Use Google Test (GTest) framework.
- You are given a function called getTestOutput which gets the code output and it will be implemented later. For now, add this function to test code exactly like this inside the source code:
char* getTestOutput(const char* input) {{
    return 'This function will be implemented later.';
}}
- Remember, do not implement the getTestOutput function. Just add it to the test code as it is given above.
- Create tests like this if there is only a single answer for the input:
TEST(Test, Test1) {{
    char* input = "example input 1";
    char* expect_string = "example output 1";
    char* result = getTestOutput(input);
    EXPECT_STREQ(result, expected_output);
}}
- If there are multiple answers or there is an allowed precision described in the description for the output, do not use the exact format above, use appropriate assertion method logic that checks if the answer is within the allowed range, or within the possible answers.
- Make sure to follow the input pattern in the problem description. E.g. if the end of the input is denoted by 0 or # or any similar character, make sure to add it to the input string in the test case.

Output instructions:
- Do not add any explanation or commentary before or after the test code.
- Wrap the entire code inside triple backticks like this:
  ```
  // your code here
"""
    return prompt

class Code:
    def __init__(self, pid, desc, src_path, lang):
        self.pid = pid
        self.desc = desc
        self.src_path = src_path
        self.lang = lang
        with open(src_path, 'r') as fd:
            self.src = fd.read()
        self.src_filename = os.path.basename(src_path).split('.')[0]

    def __str__(self):
        return f'Problem ID: {self.pid}, Language: {self.lang}, Path: {self.src_path}'

    def gen_filename(self, model):
        return f'{self.pid}_{self.src_filename}_{model}.txt'

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

def get_codes():
    problem_description_dir = Path("thesis_dataset/problem_descriptions")
    base_code_data_dir = Path("thesis_dataset/data")

    problem_ids = [x for x in problem_description_dir.glob("*.html")]
    codes = []
    for pid in problem_ids:
        with open(pid, 'r') as fd:
            description = fd.read()

        langs = ['C++', 'Python', 'Java', 'JavaScript']
        for lang in langs:
            code_data_dir = base_code_data_dir / pid.stem / lang
            if not code_data_dir.exists():
                print(f"No code for problem id: {pid.stem}")
                continue

            for src_path in code_data_dir.glob(f"*.{file_ext_map[lang]}"):
                code = Code(pid.stem, description, src_path, lang)
                codes.append(code)

    return codes

def filter_by_lang(codes, lang):
    return [code for code in codes if code.lang == lang]


def generate_cases(prompt, model):
    client = get_commercial_model(model)
    resp = prompt_commercial_model(client, model, prompt, image_id="")
    return resp


################################################################################
# post processing
def extract_code_in_backticks(unprocessed_code):
    pattern = r"```.*?\n(.*?)```"
    codes = re.findall(pattern, unprocessed_code, re.DOTALL)
    if len(codes) == 0:
        return None
    return codes[0]

def extract_tests(unprocessed_code):
    # test_pattern = r"TEST\([^\)]+\)\s*{[^}]+}" # gpt-4o's pattern
    test_pattern = r"TEST\s*\(\s*\w+\s*,\s*\w+\s*\)\s*\{(?:[^{}]|\{[^{}]*\})*\}" # claude-3-haiku pattern
    # Find all matches in the code
    tests = re.findall(test_pattern, unprocessed_code)

    # Output the tests
    # for idx, test in enumerate(tests, 1):
    #     print(f"Test {idx}:\n{test}\n")
    return tests
def extract_headers(unprocessed_code):
    pattern = r"#include\s+<[^>]+>"  # Matches #include directives
    headers = re.findall(pattern, unprocessed_code)
    return headers

################################################################################
def assemble(src_content, tests):
    getTestOutput = r"""#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;
int test_main();

char* getTestOutput(const char* input) {
   // Create unique temp file names
    char input_filename[] = "/tmp/tmp_input_XXXXXX";
    char output_filename[] = "/tmp/tmp_output_XXXXXX";
    int input_fd = mkstemp(input_filename);
    int output_fd = mkstemp(output_filename);

    if (input_fd == -1 || output_fd == -1) {
        perror("mkstemp");
        exit(1);
    }

    // Write input to input temp file
    FILE* input_file = fdopen(input_fd, "w");
    fputs(input, input_file);
    fclose(input_file);

    // Reopen input file for reading and redirect stdin
    freopen(input_filename, "r", stdin);

    // Redirect stdout to output file
    freopen(output_filename, "w", stdout);

    test_main();

  // Flush and close redirected stdout
    fflush(stdout);
    freopen("/dev/tty", "w", stdout); // restore stdout for terminal

    // Read back output
    FILE* output_file = fopen(output_filename, "r");
    fseek(output_file, 0, SEEK_END);
    long size = ftell(output_file);
    rewind(output_file);

    char* buffer = (char*)malloc(size + 1);
    if (!buffer) return NULL;
    fread(buffer, 1, size, output_file);
    buffer[size] = '\0';
    fclose(output_file);

    // Clean up temp files
    remove(input_filename);
    remove(output_filename);

    return buffer;
}
"""

    # Remove multi-line comments (/* ... */)
    src_content = re.sub(r'/\*.*?\*/', '', src_content, flags=re.DOTALL)
    # Remove whole lines that begin with optional whitespace followed by //
    src_content = re.sub(r'^\s*//.*$', '', src_content, flags=re.MULTILINE)
    main_start_pos = src_content.index("main")
    main_end_pos = main_start_pos + len("main")
    while True:
        c = src_content[main_end_pos]
        if c == ')':
            main_end_pos += 1
            break
        main_end_pos += 1
    main_function_id = src_content[main_start_pos:main_end_pos]
    test_main_function_id = f'test_{main_function_id}'
    # src_content = src_content.replace(main_function_id, test_main_function_id)
    src_content = src_content.replace(main_function_id, 'test_main()')

    full_code = f'{getTestOutput}\n\n'
    full_code += f'{src_content}\n\n'
    for test in tests:
        full_code += f'{test}\n\n'

    full_code = full_code.replace('constexpr', 'const')
    # full_code = full_code.replace('int test_main();', f'int {test_main_function_id};');
    return full_code

    '''g++ -std=c++17 p02277_s572622168_gpt-4o.cpp  -I  /opt/homebrew/opt/googletest/include -L /opt/homebrew/opt/googletest/lib  -lgtest -lgtest_main -pthread -o test.out'''
    '''brew install googletest'''

def parse_gtest_xml(xml_file):
    """Parse a GTest XML report and return JSON-compatible summary."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        summary = {
            "file": os.path.basename(xml_file),
            "name": root.attrib.get("name", ""),
            "tests": int(root.attrib.get("tests", 0)),
            "failures": int(root.attrib.get("failures", 0)),
            "errors": int(root.attrib.get("errors", 0)),
            "disabled": int(root.attrib.get("disabled", 0)),
            "timestamp": root.attrib.get("timestamp", "")
        }

        summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["disabled"]
        return summary
    except ET.ParseError:
        print(f"Failed to parse XML: {xml_file}")
        return None
    except FileNotFoundError:
        print(f"File not found: {xml_file}")
        return None

def write_json_report(results, json_path="gtest_summary.json"):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ JSON summary written to: {json_path}")

def compile_cpp_and_run(cpp_file: Path | tuple):
    if isinstance(cpp_file, tuple):
        cpp_file = cpp_file[0]

    output_file = cpp_file.with_suffix(".out")
    if not output_file.exists():
        # cmd = [GCC, *CXX_FLAGS, str(cpp_file), "-o", str(output_file)]
        cmd = [GCC, *CXX_FLAGS, str(cpp_file), "-lgtest", "-lgtest_main", "-o", str(output_file)]
        print(f"Compiling {cpp_file} -> {output_file}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed to compile {cpp_file}, cmd: {' '.join(cmd)}")
            # cpp_file.unlink()                                  # avoid bugs in coverage
            # if 'p02403_s976586168_gpt-4o' in str(output_file): #gpt-4o, generated tests is invalid due to mismatched type
            #     pass
            # elif 'p02405_s591959194_gpt-4o' in str(output_file): #gpt-4o, generated tests is invalid due to mismatched type
            #     pass
            # elif 'p02403_s555993861_gpt-4o' in str(output_file): #gpt-4o, generated tests is invalid due to mismatched type
            #     pass
            # elif 'p02388_s821487200' in str(output_file): # main contain parameter, main(int argc, char* argv[]), we don't process it
            #     pass
            # elif 'p02262_s322394753' in str(output_file): # original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p00047_s514230605' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02414_s411876042' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02270_s850877678' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02262_s511397216' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02270_s502947922' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p00356_s530969750' in str(output_file):# original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02419_s792789589' in str(output_file): # original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p00042_s296753194' in str(output_file): # original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p00001_s355998091' in str(output_file): # original code error (maybe due to system, fail on macos
            #     pass
            # elif 'p02282_s667514689' in str(output_file): # original code error (maybe due to system, fail on macos
            #     pass
            # else:
            #     sys.exit(1)

    if not output_file.exists() or output_file.stat().st_size == 0:
        # NOTE: we fail to compile this cpp file
        return

    output_file.chmod(output_file.stat().st_mode | stat.S_IXUSR)

    if cpp_file.with_suffix(".json").exists():
        print(f"Already tested {cpp_file}")
        return

    xml_file = str(cpp_file.with_suffix(".xml"))
    try:
        cmd = [str(output_file), f"--gtest_output=xml:{xml_file}"]
        print(f"> run command: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        summary = parse_gtest_xml(xml_file)
        with open(str(cpp_file.with_suffix(".json")), "w") as fd:
            json.dump(summary, fd)
    except subprocess.TimeoutExpired:
        print(f"The subprocess took too long and was terminated: {output_file}")
        summary = {"timeout": "The subprocess took too long and was terminated"}
        with open(str(cpp_file.with_suffix(".json")), "w") as fd:
            json.dump(summary, fd)
    except subprocess.CalledProcessError as ex:
        print(f"Error running {output_file}: {ex}")
        summary = {"called_error": "The subprocess took too long and was terminated"}
        with open(str(cpp_file.with_suffix(".json")), "w") as fd:
            json.dump(summary, fd)


def compile_cpp_and_run_wrapper(cpp_dir: Path):
    print("🔨 Compiling C++ sources...")
    # for cpp_file in cpp_dir.glob("*.cpp"):
    #     compile_cpp_and_run(cpp_file)
    # try:
    with Pool(32) as pool:
        pool.map(compile_cpp_and_run, [(cpp_file,) for cpp_file in cpp_dir.glob("*.cpp")])

    pool.join()
    # except Exception as ex:
    #     pool.close()

    #     sys.exit(0)


def run_coverage(path: Path, model: str):
    path.mkdir(exist_ok=True)

    html_command = [
        "gcovr",
    ]

    # for summary in path.parent.rglob("*.json"):
    #     html_command.append(str(summary.with_suffix('.cpp')))

    html_command.extend([
        "--html", "--html-details", "--output", f"{str(path)}/report.html",
    ])
    print("\n📊 Generating individual coverage reports for each executable...")
    subprocess.run(html_command, check=True)
    print(f"HTML coverage report generated")


# def run_coverage(path: Path, model: str):
#     print("\n📊 Generating individual coverage reports for each executable...")
#     path.mkdir(exist_ok=True)

#     args_list = []
#     for cpp_file in path.parent.rglob(f"*{model}*.cpp"):
#         cpp_file = cpp_file.resolve()
#         json_output = path / f'{cpp_file.stem}.json'
#         args_list.append((cpp_file, json_output))

#     # with Pool(32) as pool:
#     #     pool.map(coverage_wrapper, args_list)
#     for args in args_list:
#         coverage_wrapper(args)


# def coverage_wrapper(args):
#     coverage(*args)


# def coverage(cpp_file: Path, json_output: Path):
#     if json_output.exists():
#         return
#     escaped_path = re.sub(r'\+', r'\\+', str(cpp_file))  # Escape only + and .
#     cmd = [
#         "gcovr",
#         "-r", ".",
#         "--filter", f'"{escaped_path}"',
#         "--json",
#         "--output", str(json_output)
#     ]
#     print(f"Generating JSON coverage report for: {cpp_file.name}, \ncmd: {' '.join(cmd)}")
#     subprocess.run(cmd)

# def merge_coverage_results(path: Path):
#     import csv
#     # Collect all gcovr JSON files
#     json_files = list(path.rglob("*.json"))
#     summary_rows = []

#     for json_path in json_files:
#         with open(json_path) as f:
#             data = json.load(f)

#         files = data.get('files', [])
#         for file in files:
#             # Extract summary info from gcovr's JSON structure
#             file_coverage = {
#                 "filename": file['file'],
#                 "lines_total": data.get("line_coverage", {}).get("count", 0),
#                 "lines_covered": data.get("line_coverage", {}).get("covered", 0),
#                 "lines_percent": data.get("line_coverage", {}).get("percent", 0.0),
#                 "branches_total": data.get("branch_coverage", {}).get("count", 0),
#                 "branches_covered": data.get("branch_coverage", {}).get("covered", 0),
#                 "branches_percent": data.get("branch_coverage", {}).get("percent", 0.0),
#             }
#             summary_rows.append(file_coverage)

#     # Write to a CSV summary
#     summary_path = path / "00_coverage_summary.csv"
#     with open(summary_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
#         writer.writeheader()
#         writer.writerows(summary_rows)

#     print(f"✅ Summary saved to {str(summary_path)}")

def run_generate_cases(code: Code, model: str):
    gen_code_dir = Path(f'thesis_dataset/generated/{code.lang}')
    prompt = get_llm_prompt_unit_test_generation(code.desc, code.src)
    # print(prompt)
    gen_filename = code.gen_filename(model)
    gen_filepath = gen_code_dir / gen_filename

    if not gen_filepath.exists():
        gen_src_content = generate_cases(prompt, model)
        with open(gen_filepath, 'w') as fd:
            fd.write(gen_src_content)
        print(f"Generate raw file {code.src_path} -> {str(gen_filepath)}")
    else:
        print(f"Existed raw file {code.src_path} -> {str(gen_filepath)}")

def run_generate_cases_wrapper(args):
    run_generate_cases(*args)

def main(args):
    codes = get_codes()
    codes = filter_by_lang(codes, args.lang)

    gen_code_dir = Path(f'thesis_dataset/generated/{args.lang}')
    # Here we only select one code, use for loop to generate more tests and get coverage and tests stats
    # model requests
    # for idx, code in enumerate(codes):
    #     run(code, args.model_name)

    codes = codes[:50]
    with Pool(8) as pool:
        pool.map(run_generate_cases_wrapper, [(code, args.model_name) for code in codes])
    pool.join()

    for idx, code in enumerate(codes):
        gen_filename = code.gen_filename(args.model_name)
        gen_filepath = gen_code_dir / gen_filename
        if gen_filepath.exists() is False:
            print(f"Fail to get response for {gen_filepath}, please re-generate it ...")
            continue

        with open(gen_filepath, 'r') as fd:
            unprocessed_code = fd.read()

        # Post processing
        unprocessed_code = extract_code_in_backticks(unprocessed_code)
        if unprocessed_code is None:
            continue

        tests = extract_tests(unprocessed_code)
        gen_full_code = assemble(code.src, tests)
        with open(f'{gen_filepath}.cpp', 'w') as fd:
            fd.write(gen_full_code)

    compile_cpp_and_run_wrapper(gen_code_dir)

    if args.cov:
        # coverage_dir = gen_code_dir / f'coverage-{args.model_name}'
        coverage_dir = gen_code_dir / 'coverage'
        run_coverage(coverage_dir, args.model_name)
        # merge_coverage_results(coverage_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="C++")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--cov", action="store_true")
    args = parser.parse_args()

    main(args)
