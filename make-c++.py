import subprocess
import os
from pathlib import Path
import difflib
import gcovr
import shutil
import re
import json
from xml.etree import ElementTree as ET

C_FLAGS = ["-fprofile-arcs", "-ftest-coverage", "-g", "-I", "/opt/homebrew/opt/googletest/include", "-L", "/opt/homebrew/opt/googletest/lib", "-lgtest", "-lgtest_main", "-pthread"]
CXX_FLAGS = ["-fprofile-arcs", "-ftest-coverage", "-g", "-I", "/opt/homebrew/opt/googletest/include", "-L", "/opt/homebrew/opt/googletest/lib", "-lgtest", "-lgtest_main", "-pthread"]

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

def get_codes():
    problem_description_dir = "thesis_dataset/problem_descriptions"
    code_data_dir = "thesis_dataset/data"

    problem_ids = [pid for pid in os.listdir(problem_description_dir) if pid.endswith(".html")]

    pid_to_desc_and_lang_to_src = {}
    codes = []
    for pid in problem_ids:
        description_path = os.path.join(problem_description_dir, pid)
        with open(description_path, 'r') as fd:
            description = fd.read()

        langs = ['C++', 'Python', 'Java', 'JavaScript']
        for lang in langs:
            code_data_dir = os.path.join(code_data_dir, pid.split(".")[0], lang)
            if not os.path.isdir(code_data_dir):
                continue

            src_files = [src for src in os.listdir(code_data_dir)]
            for src_file in src_files:
                src_path = os.path.join(code_data_dir, src_file)
                code = Code(pid.split('.')[0], description, src_path, lang)
                codes.append(code)

    return codes

def filter_by_lang(codes, lang):
    return [code for code in codes if code.lang == lang]


def generate_cases(prompt, model):
    return ''


################################################################################
# post processing
def extract_code_in_backticks(unprocessed_code):
    pattern = r"```.*?\n(.*?)```"
    codes = re.findall(pattern, unprocessed_code, re.DOTALL)
    if len(codes) == 0:
        return None
    return codes[0]

def extract_tests(unprocessed_code):
    test_pattern = r"TEST\([^\)]+\)\s*{[^}]+}"
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
int test_main();
char* getTestOutput(const char* input) {
   // Create unique temp file names
    char input_filename[] = "tmp_input_XXXXXX";
    char output_filename[] = "tmp_output_XXXXXX";
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

    src_content = src_content.replace("main", "test_main")
    full_code = f'{getTestOutput}\n\n'
    full_code += f'{src_content}\n\n'
    for test in tests:
        full_code += f'{test}\n\n'
    return full_code


# def compile():
#     print("🔨 Compiling C sources...")
#     for c_file in C_DIR.glob("*.c"):
#         output_file = c_file.with_suffix(".out")
#         cmd = ["gcc", *C_FLAGS, str(c_file), "-o", str(output_file)]
#         print(f"Compiling {c_file} -> {output_file}")
#         result = subprocess.run(cmd)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {c_file}")


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

def write_json_report(results, json_path="gtest_summary.json"):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ JSON summary written to: {json_path}")

def compile_cpp_and_run(cpp_dir: Path):
    print("🔨 Compiling C++ sources...")
    for cpp_file in cpp_dir.glob("*.cpp"):
        output_file = cpp_file.with_suffix(".out")
        cmd = ["g++", *CXX_FLAGS, str(cpp_file), "-o", str(output_file)]
        print(f"Compiling {cpp_file} -> {output_file}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed to compile {cpp_file}")
        xml_file = str(cpp_file.with_suffix(".xml"))
        try:
            subprocess.run(
                [str(output_file), f"--gtest_output=xml:{xml_file}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            summary = parse_gtest_xml(xml_file)
            with open(str(cpp_file.with_suffix(".json")), "w") as fd:
                json.dump(summary, fd)

        except subprocess.CalledProcessError as ex:
            print(f"Error running {output_file}: {ex}")

def coverage(path):
    os.makedirs(path, exist_ok=True)
    print("\n📊 Generating individual coverage reports for each executable...")
    html_command = [
        "gcovr",
        "--html", "--html-details", "--output", f"{path}/report.html",
        ]
    subprocess.run(html_command, check=True)
    print(f"HTML coverage report generated")

def main():
    codes = get_codes()
    codes = filter_by_lang(codes, 'C++')
    gen_code_dir = 'thesis_dataset/generated/C++'
    # Here we only select one code, use for loop to generate more tests and get coverage and tests stats
    code = codes[0]
    model = 'gpt-4o'
    prompt = get_llm_prompt_unit_test_generation(code.desc, code.src)
    print(prompt)
    gen_filename = code.gen_filename(model)
    gen_filepath = os.path.join(gen_code_dir, gen_filename)
    # TODO
    if not os.path.isfile(gen_filepath):
        gen_src_content = generate_cases(prompt, model)
        with open(gen_filepath, 'w') as fd:
            fd.write(gen_src_content)
        print(f"Generate {code.src_path} -> {gen_filepath}")

    with open(gen_filepath, 'r') as fd:
        unprocessed_code = fd.read()

    # Post processing
    unprocessed_code = extract_code_in_backticks(unprocessed_code)
    tests = extract_tests(unprocessed_code)
    gen_full_code = assemble(code.src, tests)

    with open(f'{gen_filepath}.cpp', 'w') as fd:
        fd.write(gen_full_code)

    compile_cpp_and_run(Path(gen_code_dir))

    coverage(os.path.join(gen_code_dir, 'coverage'))

if __name__ == "__main__":
    main()
