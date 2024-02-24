import argparse
import subprocess
import huggingface_hub as hfh
import os

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("--token", type=str)
parser.add_argument("--dataset_name", type=str)

args = parser.parse_args()

print("Uploading results to HF Hub")
print("Logging in to HF Hub, first 5 characters of token: ", args.token[:5])
print("Dataset name: ", args.dataset_name)

hfh.login(args.token)

folders = []
for split_name in ["train", "valid", "test"]:
    folders.append(f"{split_name}_grappa_reconstruction_numpy")
    folders.append(f"{split_name}_sum_reconstruction_numpy")
    folders.append(f"{split_name}_mask_numpy")
    folders.append(f"{split_name}_masked_grappa_reconstruction_numpy")
    folders.append(f"{split_name}_masked_sum_reconstruction_numpy")

    folders.append(f"{split_name}_grappa_reconstruction_png")
    folders.append(f"{split_name}_sum_reconstruction_png")
    folders.append(f"{split_name}_mask_png")
    folders.append(f"{split_name}_masked_grappa_reconstruction_png")
    folders.append(f"{split_name}_masked_sum_reconstruction_png")


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
