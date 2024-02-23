import argparse
import subprocess
import huggingface_hub as hfh
import os

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("token", type=str)
parser.add_argument("dataset_name", type=str)

args = parser.parse_args()

print("Logging in to HF Hub, first 5 characters of token: ", args.token[:5])

hfh.login(args.token)

folders = []
for split_name in ["train", "valid", "test"]:
    folders.append(f"{split_name}_grappa_reconstruction")
    folders.append(f"{split_name}_sum_reconstruction")
    folders.append(f"{split_name}_mask")
    folders.append(f"{split_name}_masked_grappa_reconstruction")
    folders.append(f"{split_name}_masked_sum_reconstruction")


# filter out non-existing folders

folders = [folder for folder in folders if os.path.exists(folder)]

print("Existing folders:", folders)

# zip the folders

for folder in folders:
    print(f"Zipping {folder}...")
    subprocess.run(["zip", "-r", f"{folder}.zip", folder])

    

# upload the zipped folders
hfh.create_repo(
    "fastmri-prostate-reconstruction/"+args.dataset_name,
    repo_type="dataset",
    exist_ok=True,
    private=False,
)

api = hfh.HfApi()
for folder in folders:
    print(f"Uploading {folder}.zip...")
    api.upload_file(
        repo_id="fastmri-prostate-reconstruction/"+args.dataset_name,
        repo_type="dataset",
        path_or_fileobj=f"{folder}.zip",
        path_in_repo=f"{folder}.zip",
    )
