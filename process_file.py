
import argparse
import os

import pandas as pd
import numpy as np
import huggingface_hub as hfh
from tqdm import tqdm
import h5py
from PIL import Image
import shutil
import fastmri
import matplotlib.pyplot as plt
import matplotlib.image
from fastmri.data import transforms as T
# from torchvision.transforms import CenterCrop
# from fastmri_prostate.reconstruction.utils import center_crop_im
import torch
 
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("kspace_path")
parser.add_argument("split_name")
parser.add_argument("output_numpy", default=False, type=bool)
parser.add_argument("output_png", default=False, type=bool)



args = parser.parse_args()

if not (args.output_numpy or args.output_png):
    raise ValueError("At least one of output_numpy or output_png must be True")

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

def process_file(file_path, split_name=None):
    file_id = file_path.split("_")[-1].replace(".h5", "")
    file = h5py.File(file_path)
    
    kspace = file["kspace"][:]
    
    grappa_reconstruction = file["reconstruction_rss"][:]
    
    num_slices = grappa_reconstruction.shape[0]
    num_coils = kspace.shape[2] #  (averages, slices, coils, readout, phase)
    
    # save our reconstruction # compute
    kspace_sum = kspace[0, :, :, :] + kspace[1, :, :, :]
    kspace_sum = T.to_tensor(kspace_sum)
    kspace_sum = fastmri.ifft2c(kspace_sum)
    kspace_sum = fastmri.complex_abs(kspace_sum)
    kspace_sum = fastmri.rss(kspace_sum, dim=1)
    kspace_sum = torch.flip(kspace_sum, dims=[1])
    kspace_sum = center_crop_im(kspace_sum, crop_to_size=(320, 320))
    
    kspace_mask = cartesian_mask([num_slices, kspace.shape[-2], kspace.shape[-1]], 4, centered=True)
    
    kspace_mask = kspace_mask.reshape(1, num_slices, 1, kspace.shape[-2], kspace.shape[-1])
    kspace_mask = np.repeat(np.repeat(kspace_mask, 3, axis=0), num_coils, axis=2)    

    kspace_masked = kspace * kspace_mask

    # save our reconstruction # compute
    kspace_sum_masked = kspace_masked[0, :, :, :] + kspace_masked[1, :, :, :]
    kspace_sum_masked = T.to_tensor(kspace_sum_masked)
    kspace_sum_masked = fastmri.ifft2c(kspace_sum_masked)
    kspace_sum_masked = fastmri.complex_abs(kspace_sum_masked)
    kspace_sum_masked = fastmri.rss(kspace_sum_masked, dim=1)
    kspace_sum_masked = torch.flip(kspace_sum_masked, dims=[1])
    kspace_sum_masked = center_crop_im(kspace_sum_masked, crop_to_size=(320, 320))
    
    print("started saving calculations")
    for slice_idx in range(num_slices):

        reconstruction_slice = grappa_reconstruction[slice_idx]
        kspace_sum_reconstruction_slice = kspace_sum[slice_idx]
        mask_slice = kspace_mask[0, slice_idx, 0]
        kspace_sum_masked_slice = kspace_sum_masked[slice_idx]

        if args.output_numpy:
            np.save(f"{split_name}_grappa_reconstruction/{file_id}.{slice_idx}.npy", reconstruction_slice)
            np.save(f"{split_name}_sum_reconstruction/{file_id}.{slice_idx}.npy", kspace_sum_reconstruction_slice)
            np.save(f"{split_name}_mask/{file_id}.{slice_idx}.npy", mask_slice)
            np.save(f"{split_name}_masked_sum_reconstruction/{file_id}.{slice_idx}.npy", kspace_sum_masked_slice)

        if args.output_png:
            matplotlib.image.imsave(f"{split_name}_grappa_reconstruction/{file_id}.{slice_idx}.png", reconstruction_slice, cmap="gray")
            matplotlib.image.imsave(f"{split_name}_sum_reconstruction/{file_id}.{slice_idx}.png", kspace_sum_reconstruction_slice, cmap="gray")
            matplotlib.image.imsave(f"{split_name}_mask/{file_id}.{slice_idx}.png", mask_slice, cmap="gray")
            matplotlib.image.imsave(f"{split_name}_masked_sum_reconstruction/{file_id}.{slice_idx}.png", kspace_sum_masked_slice, cmap="gray")
        
    print("finished saving calculations")


if __name__ == "__main__":
    process_file(args.kspace_path, args.split_name)