import subprocess
import huggingface_hub as hfh
import os
from pathlib import Path


def create_folders(root_path):
    folders = []

    for split_name in ["train", "valid", "test"]:
        folders.extend([
            f"{root_path}/{split_name}_grappa_reconstruction_numpy",
            f"{root_path}/{split_name}_sum_reconstruction_numpy",
            f"{root_path}/{split_name}_mask_numpy",
            f"{root_path}/{split_name}_masked_grappa_reconstruction_numpy",
            f"{root_path}/{split_name}_masked_sum_reconstruction_numpy",

            f"{root_path}/{split_name}_kspace_numpy",
            f"{root_path}/{split_name}_kspace_masked_numpy",
            
            f"{root_path}/{split_name}_grappa_reconstruction_png",
            f"{root_path}/{split_name}_sum_reconstruction_png",
            f"{root_path}/{split_name}_mask_png",
            f"{root_path}/{split_name}_masked_grappa_reconstruction_png",
            f"{root_path}/{split_name}_masked_sum_reconstruction_png",
        ])
    for folder in folders:
        print(f"Creating folder {folder}...")
        os.makedirs(folder, exist_ok=True)
    return folders
    
def zip_folders(root_path, folders):

    os.chdir(root_path)
    for folder in folders:
        print(f"Zipping {folder}...")
        # change directory to the folder
        os.system(f"zip -r {folder.split('/')[-1]}.zip {folder.split('/')[-1]}")

def upload_folders(dataset_name, folders, root_path):
    # upload the zipped folders
    hfh.create_repo(
        "fastmri-prostate-reconstruction/"+dataset_name,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    api = hfh.HfApi()
    os.chdir(root_path)
    for folder in folders:
        print(f"Uploading {folder}.zip...")
        api.upload_file(
            repo_id="fastmri-prostate-reconstruction/"+dataset_name,
            repo_type="dataset",
            path_or_fileobj=f"{folder}.zip",
            path_in_repo=f"data/{folder.split('/')[-1]}.zip",
        )
