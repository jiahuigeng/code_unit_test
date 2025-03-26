import os
import argparse
import pandas as pd

meta_folder = "datasets/origin_datasets/metadata"
solu_folder = "datasets/origin_datasets/solution"
desc_folder = "datasets/origin_datasets/problem_description"

langs = ["Python", "Java", "C++", "JavaScript", "Go", "Ruby", "Rust", "C"]



def dataset_build(args):
    # save csv files for languages
    # pid: desc_path：pos_submission_id, neg_submission_id, pos_submission_path, neg_submission_path
    #
    full_dataset = {lang: [] for lang in langs}

    files = os.listdir(meta_folder)
    files = [f for f in files if "gt" not in f]
    for fn in sorted(files):
        pid = fn.split(".")[0]
        # desc_path = os.path.join(desc_folder, f'{pid}.html')
        desc_path = f"{desc_folder}/{pid}.html"
        if not os.path.exists(desc_path):
            print(f"{desc_path} not exists")
        data = {}
        neg_csv = pd.read_csv(os.path.join(meta_folder, fn))
        for idx, row in neg_csv.iterrows():
            row_lang = row["language"]
            if row_lang not in data:
                data[row_lang] = {}
            data[row_lang]["neg_submission_id"] = row["submission_id"]
            # neg_path = os.path.join(solu_folder, pid, row_lang, f"{row['submission_id']}.py")
            neg_path = f"{solu_folder}/{pid}/{row_lang}/{row['submission_id']}.{row['filename_ext']}"
            if not os.path.exists(neg_path):
                print(f"{neg_path} not exists")
            data[row_lang]["neg_submission_path"] = neg_path

        pos_csv = pd.read_csv(os.path.join(meta_folder, f"{pid}_gt.csv"))
        for idx, row in pos_csv.iterrows():
            row_lang = row["language"]
            data[row_lang]["pos_submission_id"] = row["submission_id"]
            # pos_path = os.path.join(solu_folder, pid, row_lang, f"{row['submission_id']}.py")
            pos_path = f"{solu_folder}/{pid}/{row_lang}/{row['submission_id']}.{row['filename_ext']}"
            if not os.path.exists(pos_path):
                print(f"{pos_path} not exists")
            data[row_lang]["pos_submission_path"] = pos_path

        for lang in data:
            full_dataset[lang].append({"pid": pid, "desc_path": desc_path,
                                       "neg_submission_id": data[lang]["neg_submission_id"],
                                       "neg_submission_path": data[lang]["neg_submission_path"],
                                       "pos_submission_id": data[lang]["pos_submission_id"],
                                       "pos_submission_path": data[lang]["pos_submission_path"]})


    for lang in full_dataset:
        save_xlsx_file = os.path.join("datasets", f"full_{lang}.xlsx")
        df = pd.DataFrame(full_dataset[lang])
        df.to_excel(save_xlsx_file, index=False)











if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="dataset_build")

    args = parser.parse_args()

    dataset_build(args)