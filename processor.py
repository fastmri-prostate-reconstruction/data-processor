import os
import subprocess
import gc
import shutil
import argparse
import sys
import json

import pandas as pd
import numpy as np
import huggingface_hub as hfh
from tqdm.notebook import tqdm
import h5py
from PIL import Image
import fastmri
import matplotlib.pyplot as plt
import matplotlib.image
from fastmri.data import transforms as T
import torch

import upload_results
import process_file

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("--token", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--cartesian_mask_acc", type=int, default=4,)
parser.add_argument("--cartesian_mask_sample_n", type=int, default=10,)
parser.add_argument("-c", "--cartesian_mask_centered", action="store_true")
parser.add_argument("-d", "--debug", action="store_true")

args = parser.parse_args()

print("Logging in to HF Hub, first 5 characters of token: ", args.token[:5])
print("Dataset name: ", args.dataset_name)

hfh.login(args.token)

print("Cartesian mask acc: ", args.cartesian_mask_acc)
print("Cartesian mask sample_n: ", args.cartesian_mask_sample_n)
print("Cartesian mask centered: ", args.cartesian_mask_centered)

if args.debug:
    print("Debug mode is on")

all_files = list(hfh.list_files_info(
    repo_id="osbm/fastmri-prostate",
    repo_type="dataset",
))
all_files = [file.path for file in all_files]

train_files = [file for file in all_files if "training_T2" in file]
valid_files = [file for file in all_files if "validation_T2" in file]
test_files = [file for file in all_files if "test_T2" in file]

root_path = "/app"

def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

def center_crop_im(im_3d: np.ndarray, crop_to_size: tuple) -> np.ndarray:
    """
    Center crop an image to a given size.
    
    Parameters:
    -----------
    im_3d : numpy.ndarray
        Input image of shape (slices, x, y).
    crop_to_size : list
        List containing the target size for x and y dimensions.
    
    Returns:
    --------
    numpy.ndarray
        Center cropped image of size {slices, x_cropped, y_cropped}. 
    """
    x_crop = im_3d.shape[-1]/2 - crop_to_size[0]/2
    y_crop = im_3d.shape[-2]/2 - crop_to_size[1]/2

    return im_3d[:, int(y_crop):int(crop_to_size[1] + y_crop), int(x_crop):int(crop_to_size[0] + x_crop)]  


from numpy.lib.stride_tricks import as_strided

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def cartesian_mask(shape, acc, sample_n=10, centered=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centered:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask

folders = upload_results.create_folders(root_path)

for split_name, split in zip(["train", "valid", "test"], [train_files, valid_files, test_files]):
    print(f"Started {split_name} split")    

    for filename in split:
        print(f"Started image {filename}")
        
        file_path = hfh.hf_hub_download(
            repo_id="osbm/fastmri-prostate",
            repo_type="dataset",
            filename=filename,
            cache_dir="hf_cache",
        )

        print("Processing", file_path)        
        process_file.process_file(
            file_path,
            split_name,
            root_path=root_path,
            cartesian_mask_acc=args.cartesian_mask_acc,
            cartesian_mask_sample_n=args.cartesian_mask_sample_n,
            cartesian_mask_centered=args.cartesian_mask_centered
        )

        shutil.rmtree('hf_cache')
        gc.collect()
        if args.debug:
            break

upload_results.zip_folders(root_path, folders)
upload_results.upload_folders(args.dataset_name, folders, root_path)


images = [
    (f"{root_path}/train_grappa_reconstruction_png/0001.15.png", "grappa_reconstruction"),
    # f"{root_path}/train_grappa_reconstruction_png/0001.15.png",
    (f"{root_path}/train_sum_reconstruction_png/0001.15.png", "sum_reconstruction"),
    (f"{root_path}/train_mask_png/0001.15.png", "mask"),
    (f"{root_path}/train_masked_grappa_reconstruction_png/0001.15.png", "masked_grappa_reconstruction"),
    (f"{root_path}/train_masked_sum_reconstruction_png/0001.15.png", "masked_sum_reconstruction"),
]


for image, image_type in images:
    if not os.path.exists(image):
        print(f"Example Image {image} does not exist, skipping")
        continue

    api = hfh.HfApi()
    api.upload_file(
        repo_id="fastmri-prostate-reconstruction/"+args.dataset_name,
        repo_type="dataset",
        path_or_fileobj=image,
        path_in_repo="example_image/" + image_type+"_" + image.split("/")[-1],
    )

print("Finished uploading example images")

config = {
    "cartesian_mask_acc": args.cartesian_mask_acc,
    "cartesian_mask_sample_n": args.cartesian_mask_sample_n,
    "cartesian_mask_centered": args.cartesian_mask_centered,
    "debug": args.debug,
}

with open(f"{root_path}/config.json", "w") as f:
    json.dump(config, f)

api.upload_file(
    repo_id="fastmri-prostate-reconstruction/"+args.dataset_name,
    repo_type="dataset",
    path_or_fileobj=f"{root_path}/config.json",
    path_in_repo="config.json",
)

print("Finished uploading config file")