import subprocess
import os
from pathlib import Path
import difflib
import gcovr
import shutil
import re

# Directories
C_DIR = Path("C")
CPP_DIR = Path("C++")
GEN_C_DIR = Path("gen/C")
GEN_CPP_DIR = Path("gen/C++")

# JAVA_DIR = Path("Java")
# GO_DIR = Path("Go")
# RUST_DIR = Path("Rust")

TEST_DIR = Path("tests")
COVERAGE_DIR = Path("coverage")
# Java Coverage
JACOCO_AGENT = Path("jacocoagent.jar")
JACOCO_CLI = Path("jacococli.jar")
EXEC_FILE = Path("jacoco.exec")

C_FLAGS = ["-fprofile-arcs", "-ftest-coverage", "-g", "-I", "/opt/homebrew/opt/googletest/include", "-L", "/opt/homebrew/opt/googletest/lib", "-lgtest", "-lgtest_main", "-pthread"]
CXX_FLAGS = ["-fprofile-arcs", "-ftest-coverage", "-g", "-I", "/opt/homebrew/opt/googletest/include", "-L", "/opt/homebrew/opt/googletest/lib", "-lgtest", "-lgtest_main", "-pthread"]

# def get_llm_prompt_unit_test_generation(description, code):
#     prompt = f"""
# Generate comprehensive unit tests for the given code using the provided problem description.

# Requirements:
# - Create exactly 10 unit tests.
# - Each test must be written as a separate test function.

# Inputs:
# - PROBLEM DESCRIPTION:
# {description}
# - CODE TO TEST:
# {code}

# Guidelines for the tests:
# - Use 'unittest' module.
# - Mock the sys.stdin and sys.stdout appropriately for the given code.
# - Use runpy.run_path() to run the submission code instead of importing it. The path of the given code is 'submission_code.py', therefore this path must be run. Do not run any other path or try to mock this file.
# - Provide the complete runnable Python code for the test module.
# - Do not add any explanation or commentary before or after the output.

# Here is an example of mocking sys.stdin and sys.stdout:
# def test_sample_input_1(self):
#         user_input = "example_input"
#         expected_output = "expected_output"
#         with patch('sys.stdin', io.StringIO(user_input)), patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
#             runpy.run_path('submission_code.py', run_name='__main__')
#             self.assertEqual(mock_stdout.getvalue(), expected_output)
# - If there are multiple answers or there is an allowed precision described in the description for the output, do not use the exact format above, use appropriate assertion method logic that checks if the answer is within the allowed range, or within the possible answers.
# - Make sure to follow the input pattern in the problem description. E.g. if the end of the input is denoted by 0 or # or any similar character, make sure to add it to the input string in the test case.
# """
#     return prompt

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


# def process_generated_unit_tests(run_correct_submission_code):
#     if not os.path.exists(processed_unit_tests_dir):
#         os.makedirs(processed_unit_tests_dir)

#     for filename in os.listdir(generated_unit_tests_dir):
#         if filename.endswith('.txt'):
#             with open(os.path.join(generated_unit_tests_dir, filename), 'r', encoding='utf-8') as file:
#                 content = file.read()

#             content = extract_string_between_triple_quotes(content)

#             problem_id = filename.split('_')[0]
#             submission_code_path_for_evaluation = None
#             data = pd.read_excel(python_metadata_path)
#             for i in range(len(data)):
#                 if data.iloc[i]['pid'] == problem_id:
#                     submission_id_for_evaluation = data.iloc[i]['pos_submission_id'] if run_correct_submission_code else data.iloc[i]['neg_submission_id']
#                     submission_code_path_for_evaluation = f'{submission_dataset_dir}/{problem_id}_{submission_id_for_evaluation}.py'
#                     break

#             content = content.replace('submission_code.py', submission_code_path_for_evaluation)

#             with open(os.path.join(processed_unit_tests_dir, filename).replace('.txt','.py'), 'w', encoding='utf-8') as file:
#                 file.write(content)

# def compile_c():
#     print("🔨 Compiling C sources...")
#     for c_file in C_DIR.glob("*.c"):
#         output_file = c_file.with_suffix(".out")
#         cmd = ["gcc", *C_FLAGS, str(c_file), "-o", str(output_file)]
#         print(f"Compiling {c_file} -> {output_file}")
#         result = subprocess.run(cmd)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {c_file}")

# def compile_cpp():
#     print("🔨 Compiling C++ sources...")
#     for cpp_file in CPP_DIR.glob("*.cpp"):
#         output_file = cpp_file.with_suffix(".out")
#         cmd = ["g++", *CXX_FLAGS, str(cpp_file), "-o", str(output_file)]
#         print(f"Compiling {cpp_file} -> {output_file}")
#         result = subprocess.run(cmd)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {cpp_file}")

# def compile_java():
#     print("☕ Compiling Java sources...")
#     for java_file in JAVA_DIR.glob("*.java"):
#         cmd = ["javac", str(java_file)]
#         print(f"Compiling {java_file}")
#         result = subprocess.run(cmd)
#         class_name = java_file.stem
#         main_class = JAVA_DIR / "Main.class"
#         new_class = JAVA_DIR / f"{class_name}.class"
#         if main_class.exists():
#             shutil.move(main_class, new_class)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {java_file}")

# def compile_go():
#     print("🐹 Compiling Go sources...")
#     for go_file in GO_DIR.glob("*.go"):
#         output_file = go_file.with_suffix(".out")
#         cmd = ["go", "build", "-o", str(output_file), str(go_file)]
#         print(f"Compiling {go_file} -> {output_file}")
#         result = subprocess.run(cmd)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {go_file}")

# def compile_rust():
#     print("🦀 Compiling Rust sources...")
#     for rs_file in RUST_DIR.glob("*.rs"):
#         output_file = rs_file.with_suffix(".out")
#         cmd = ["rustc", str(rs_file), "-o", str(output_file)]
#         print(f"Compiling {rs_file} -> {output_file}")
#         result = subprocess.run(cmd)
#         if result.returncode != 0:
#             print(f"❌ Failed to compile {rs_file}")

# def compile_sources():
#     compile_c()
#     compile_cpp()
#     # compile_java()
#     # compile_go()
#     # compile_rust()

# def get_c_cmd(exe_path: Path):
#     cmd = [str(exe_path)]
#     return cmd

# def get_cpp_cmd(exe_path: Path):
#     cmd = [str(exe_path)]
#     return cmd

# # def get_java_cmd(exe_path: Path):
# #     class_name = "Main"
# #     # cmd = ["java", "-cp", str(exe_path.parent), class_name]
# #     # JACOCO_CLI = "org.jacoco.cli-0.8.6.jar"
# #     # cmd = [
# #     #     "java", "-jar", str(JACOCO_CLI), "report", str(EXEC_FILE),
# #     #     "--classfiles", str(JAVA_DIR),
# #     #     "--sourcefiles", str(JAVA_DIR),
# #     #     "--html", str(COVERAGE_DIR),
# #     #     "--xml", str(COVERAGE_DIR / "report.xml"),
# #     #     "--csv", str(COVERAGE_DIR / "report.csv"),
# #     # ]
# #     JACOCO_AGENT = "org.jacoco.agent-0.8.5.jar"
# #     EXEC_FILE = Path("jacoco.exec")
# #     cmd = [
# #         "java",
# #         f"-javaagent:{JACOCO_AGENT}=destfile={EXEC_FILE}",
# #         "-cp", str(exe_path.parent),
# #         class_name
# #     ]

# #     return cmd

# # def get_go_cmd(exe_path: Path):
# #     cmd = []
# #     return cmd

# # def get_rust_cmd(exe_path: Path):
# #     cmd = []
# #     return cmd


# def run_test(exe_path: Path, test_input: str, expected_output: str):
#     print(f"\n[TEST] {exe_path}")
#     # if exe_path.suffix == ".class" and "Java" in str(exe_path):
#     #     cmd = get_java_cmd(exe_path)
#     # else:
#     cmd = [str(exe_path)]

#     try:
#         result = subprocess.run(
#             cmd,
#             input=test_input,
#             text=True,
#             capture_output=True,
#             timeout=5  # Optional timeout
#         )
#     except Exception as e:
#         print(f"❌ ERROR running {exe_path.name}: {e}")
#         return False

#     # Compare actual output with expected
#     actual = result.stdout.strip().splitlines()
#     expected = expected_output.strip().splitlines()

#     if actual == expected:
#         print("✅ PASS")
#         return True
#     else:
#         print("❌ FAIL")
#         print("Expected:")
#         print("\n".join(expected))
#         print("Got:")
#         print("\n".join(actual))

#         diff = difflib.unified_diff(expected, actual, fromfile='expected', tofile='actual', lineterm='')
#         print("\n--- Diff ---")
#         for line in diff:
#             print(line)
#         return False

# # Function to analyze code coverage and generate HTML and JSON reports for each executable
# def coverage():
#     print("\n📊 Generating individual coverage reports for each executable...")
#     html_command = [
#         "gcovr",
#         "--html", "--html-details", "--output", "coverage/report.html",
#         ]
#     subprocess.run(html_command, check=True)
#     print(f"HTML coverage report generated")
#     # Iterate through each executable and generate reports
#     # for exe in list(C_DIR.glob("*.out")) + list(CPP_DIR.glob("*.out")):
#     #     base = exe.stem
#     #     # Run the executable to generate .gcda files
#     #     # try:
#     #     #     subprocess.run([str(exe)], check=True)
#     #     # except subprocess.CalledProcessError as e:
#     #     #     print(f"❌ Error running {exe}: {e}")
#     #     #     continue

#     #     # Running gcovr to generate HTML report for each executable
#     #     gcovr_html_report = f"{base}_coverage_report.html"
#     #     html_command = [
#     #         "gcovr",
#     #         "--html", "--html-details", "--output", gcovr_html_report,
#     #         "--object-directory", str(exe.parent)  # Ensure gcovr finds .gcda/.gcno files
#     #     ]
#     #     subprocess.run(html_command, check=True)
#     #     print(f"HTML coverage report generated for {exe}: {gcovr_html_report}")

#     #     # Running gcovr to generate JSON report for each executable
#     #     gcovr_json_report = f"{base}_coverage_report.json"
#     #     json_command = [
#     #         "gcovr",
#     #         "--json", "--output", gcovr_json_report,
#     #         "--object-directory", str(exe.parent)  # Ensure gcovr finds .gcda/.gcno files
#     #     ]
#     #     subprocess.run(json_command, check=True)
#     #     print(f"JSON coverage report generated for {exe}: {gcovr_json_report}")

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
        # if self.lang == 'C++':
        #     return f'{self.pid}_{self.src_filename}_{model}.cpp'

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

def compile_cpp_and_run(cpp_dir: Path):
    print("🔨 Compiling C++ sources...")
    for cpp_file in cpp_dir.glob("*.cpp"):
        output_file = cpp_file.with_suffix(".out")
        cmd = ["g++", *CXX_FLAGS, str(cpp_file), "-o", str(output_file)]
        print(f"Compiling {cpp_file} -> {output_file}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed to compile {cpp_file}")

        subprocess.run([output_file])


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
    code = codes[0]
    model = 'gpt-4o'
    prompt = get_llm_prompt_unit_test_generation(code.desc, code.src)
    print(prompt)
    gen_filename = code.gen_filename(model)
    gen_filepath = os.path.join(gen_code_dir, gen_filename)
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
    # total = 0
    # passed = 0

    # files = list(set([f.split('.')[0] for f in os.listdir(TEST_DIR)]))

    # for exe in (
    #     list(C_DIR.glob("*.out")) +
    #     list(CPP_DIR.glob("*.out"))
    #     # list(JAVA_DIR.glob("*.class"))
    #     ):
    #     for file in files:
    #         in_file = TEST_DIR / f"{file}.in"
    #         out_file = TEST_DIR / f"{file}.out"
    #         if not in_file.exists() or not out_file.exists():
    #             print(f"⚠️ Skipping {exe}: missing test files")
    #             continue

    #         total += 1
    #         in_str = in_file.read_text()
    #         out_str = out_file.read_text()

    #         if run_test(exe, in_str, out_str):
    #             passed += 1

    # print(f"\n📊 Summary: {passed}/{total} passed")
    # coverage()


if __name__ == "__main__":
    main()
