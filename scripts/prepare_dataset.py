import os
import json
import shutil
import click
from sklearn.model_selection import train_test_split
import random


BASE_DIR = "/mnt/data/bwalkow/voxceleb/vox2"
SAVE_DIR = "/mnt/data/bwalkow/voxceleb/datasets"

def save_files(files, dest, i):
    for file in files:
        file_name = file.split("/")[-1]
        shutil.copy(file, f"{dest}/{i}/{file_name}")


def prepare_datasets(
    test_new_users=10,
    ds_name="test_aac",
    train_ratio=0.6,
    dev_ratio=0.2,
    max_user_files=15
):
    dataset_dir=f"{BASE_DIR}/{ds_name}"
    dest_train = f"{SAVE_DIR}/train"
    dest_test = f"{SAVE_DIR}/test_known"
    dest_dev = f"{SAVE_DIR}/dev_known" # for modyfication
    dest_dev_unknown = f"{SAVE_DIR}/dev_unknown"
    dest_test_unknown = f"{SAVE_DIR}/test_unknown"
    os.makedirs(dest_train, exist_ok=True)
    os.makedirs(dest_test, exist_ok=True)
    os.makedirs(dest_dev, exist_ok=True)


    cls = os.listdir(dataset_dir)
    to_test_new = cls[:test_new_users*2]
    to_train = cls[test_new_users:]
    to_test = cls[test_new_users:]
    train_cls_mapping = {}
    sizes = {"train": 0, "test": 0, "test_unknown": 0, "dev": 0, "dev_unknown": 0}
    print(f"Train: {len(to_train)}, test: {len(to_test)}, test_new: {len(to_test_new)}")
    i = 0
    for user in os.listdir(dataset_dir):
        files = []
        for dir in os.listdir(f"{dataset_dir}/{user}"):
            for file in os.listdir(f"{dataset_dir}/{user}/{dir}"):
                files.append(f"{dataset_dir}/{user}/{dir}/{file}")

        if user in to_test_new:
            if user in to_test_new[:test_new_users]:
                os.makedirs(f"{dest_test_unknown}/{i}", exist_ok=True)
                for j, file in enumerate(files):
                    if j >= max_user_files:
                        break
                    file_name = file.split("/")[-1]
                    shutil.copy(file, f"{dest_test}/{i}/{file_name}")
                    sizes["test_unknown"] += 1
                train_cls_mapping[user] = i
                i += 1
            else: # dev unknown
                os.makedirs(f"{dest_dev_unknown}/{i}", exist_ok=True)
                for j, file in enumerate(files):
                    if j >= max_user_files:
                        break
                    file_name = file.split("/")[-1]
                    shutil.copy(file, f"{dest_dev_unknown}/{i}/{file_name}")
                    sizes["dev_unknown"] += 1
                i += 1
        else:
            # split
            train_files, test_files = train_test_split(files, test_size=1-train_ratio)
            dev_files, test_files = train_test_split(test_files, test_size=dev_ratio/(1-train_ratio))
            train_files = random.sample(train_files, min(len(train_files), max_user_files))
            dev_files = random.sample(dev_files, min(len(dev_files), max_user_files))
            test_files = random.sample(test_files, min(len(test_files), max_user_files))
            sizes["train"] += len(train_files)
            sizes["test"] += len(test_files)
            sizes["dev"] += len(dev_files)

            # train
            os.makedirs(f"{dest_train}/{i}", exist_ok=True)
            save_files(train_files, dest_train, i)

            # dev
            os.makedirs(f"{dest_dev}/{i}", exist_ok=True)
            save_files(dev_files, dest_dev, i)

            # test
            os.makedirs(f"{dest_test}/{i}", exist_ok=True)
            save_files(test_files, dest_test, i)

            train_cls_mapping[user] = i
            i += 1


    report = {
        "mapping": train_cls_mapping,
        "sizes": sizes
    }
    print("Sizes:", sizes)
    with open(f"{SAVE_DIR}/cls_mapping.json", "w") as f:
        f.write(json.dumps(report, ensure_ascii=False))


@click.command()
@click.option("--test_new_users", default=10, help="Number of new users to test")
@click.option("--ds_name", default="test_aac", help="Dataset name")
@click.option("--train_ratio", default=0.6, help="Train ratio")
@click.option("--dev_ratio", default=0.2, help="Dev ratio")
@click.option("--max_user_files", default=10, help="Max files per user")
def main(
    test_new_users=10,
    ds_name="test_aac",
    train_ratio=0.6,
    dev_ratio=0.2,
    max_user_files=15,
):
    # test_aac
    prepare_datasets(
        test_new_users,
        ds_name,
        train_ratio,
        dev_ratio,
        max_user_files
    )


main()
