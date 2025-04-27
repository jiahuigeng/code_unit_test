import os
import argparse
import pandas as pd
from utils_llm import *
from shutil import copyfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import copyfile

# meta_folder = "datasets/origin_datasets/metadata"
# solu_folder = "datasets/origin_datasets/solution"
# desc_folder = "datasets/origin_datasets/problem_description"


desc_folder = "thesis_dataset/problem_descriptions"

langs = ["JavaScript"]

def init_cols(df_input, new_cols):
    for col in new_cols:
        if col not in df_input.columns:
            df_input[col] = None
    return df_input


# def dataset_build(args):
#     # save csv files for languages
#     # pid: desc_path：pos_submission_id, neg_submission_id, pos_submission_path, neg_submission_path
#     #
#     full_dataset = {lang: [] for lang in langs}
#
#     files = os.listdir(meta_folder)
#     files = [f for f in files if "gt" not in f]
#     for fn in sorted(files):
#         pid = fn.split(".")[0]
#         # desc_path = os.path.join(desc_folder, f'{pid}.html')
#         desc_path = f"{desc_folder}/{pid}.html"
#         if not os.path.exists(desc_path):
#             print(f"{desc_path} not exists")
#         data = {}
#         neg_csv = pd.read_csv(os.path.join(meta_folder, fn))
#         for idx, row in neg_csv.iterrows():
#             row_lang = row["language"]
#             if row_lang not in data:
#                 data[row_lang] = {}
#             data[row_lang]["neg_submission_id"] = row["submission_id"]
#             # neg_path = os.path.join(solu_folder, pid, row_lang, f"{row['submission_id']}.py")
#             neg_path = f"{solu_folder}/{pid}/{row_lang}/{row['submission_id']}.{row['filename_ext']}"
#             if not os.path.exists(neg_path):
#                 print(f"{neg_path} not exists")
#             data[row_lang]["neg_submission_path"] = neg_path
#
#         pos_csv = pd.read_csv(os.path.join(meta_folder, f"{pid}_gt.csv"))
#         for idx, row in pos_csv.iterrows():
#             row_lang = row["language"]
#             data[row_lang]["pos_submission_id"] = row["submission_id"]
#             # pos_path = os.path.join(solu_folder, pid, row_lang, f"{row['submission_id']}.py")
#             pos_path = f"{solu_folder}/{pid}/{row_lang}/{row['submission_id']}.{row['filename_ext']}"
#             if not os.path.exists(pos_path):
#                 print(f"{pos_path} not exists")
#             data[row_lang]["pos_submission_path"] = pos_path
#
#         for lang in langs:
#             if lang not in full_dataset:
#                 continue
#             try:
#                 full_dataset[lang].append({"pid": pid, "desc_path": desc_path,
#                                            "neg_submission_id": data[lang]["neg_submission_id"],
#                                            "neg_submission_path": data[lang]["neg_submission_path"],
#                                            "pos_submission_id": data[lang]["pos_submission_id"],
#                                            "pos_submission_path": data[lang]["pos_submission_path"]})
#             except:
#                 continue
#
#
#     for lang in langs:
#         save_xlsx_file = os.path.join("datasets", f"full_{lang}.xlsx")
#         df = pd.DataFrame(full_dataset[lang])
#         df.to_excel(save_xlsx_file, index=False)

import re

def contains_japanese(text):
    # Hiragana: \u3040–\u309F
    # Katakana: \u30A0–\u30FF
    # Kanji (CJK Unified Ideographs): \u4E00–\u9FAF
    pattern = re.compile(r'[\u3040-\u30FF\u4E00-\u9FFF]')
    return bool(pattern.search(text))


def translate_description(args):
    # desc_files = [os.path.join(desc_folder, "p00094.html")]
    client = get_commercial_model("gpt4o")
    PMP_translate = """Translate the Japanese content in the html page into English, Output only the converted html content, do not generate other content
    Input: {input}\n\n
    Output:
    """
    desc_files = os.listdir(desc_folder)
    print(desc_files[:3])

    def process_file(fn):
        file_path = os.path.join(desc_folder, fn)
        print(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if contains_japanese(content):
                print(f"[Processing] {file_path}")
                res = prompt_commercial_model(client, "gpt4o", PMP_translate.format(input=content), image_id=None)

                os.makedirs("updated_description", exist_ok=True)
                copyfile(file_path, f"updated_description/{fn}")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(res)
                print(f"[Translated] {file_path}")
            else:
                print(f"[Skipped] {file_path} (No Japanese detected)")
        except Exception as e:
            print(f"[Error] {file_path}: {e}")

    # 控制线程数量，例如同时运行5个任务
    max_workers = min(5, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, fn) for fn in desc_files]
        for future in as_completed(futures):
            pass  # 你可以选择在这里打印结果或处理异常

    # for fn in desc_files:
    #     file = os.path.join(desc_folder, fn)
    #     content = open(file).read()
    #     if contains_japanese(content):
    #         print(file)
    #         res = (prompt_commercial_model(client, "gpt4o", PMP_translate.format(input=content), image_id=None))
    #         copyfile(file, f"tmp_description/{fn}")
    #         with open(file, "w", encoding="utf-8") as f:
    #             f.write(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="translate")

    args = parser.parse_args()

    # if args.task == "dataset_build":
    #     dataset_build(args)
    if args.task == "translate":
        translate_description(args)
