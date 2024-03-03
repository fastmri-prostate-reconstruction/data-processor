
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
from numpy.lib.stride_tricks import as_strided
# from torchvision.transforms import CenterCrop
# from fastmri_prostate.reconstruction.utils import center_crop_im
import torch
 
from pathlib import Path

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


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc=4, sample_n=10, centered=False):
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

def process_file(file_path, split_name=None, root_path="/app", cartesian_mask_centered=True, cartesian_mask_acc=4, cartesian_mask_sample_n=10):
    print("started processing file")
    file_id = file_path.split("_")[-1].replace(".h5", "")
    file = h5py.File(file_path)
    
    kspace = file["kspace"][:]
    
    grappa_reconstruction = file["reconstruction_rss"][:]
    
    num_slices = grappa_reconstruction.shape[0]

    for slice_idx in range(num_slices):
        reconstruction_slice = grappa_reconstruction[slice_idx]
        print(f"Saving slice {slice_idx} to {root_path}/{split_name}_grappa_reconstruction_numpy/{file_id}.{slice_idx}.npy")
        np.save(f"{root_path}/{split_name}_grappa_reconstruction_numpy/{file_id}.{slice_idx}.npy", reconstruction_slice)
        matplotlib.image.imsave(f"{root_path}/{split_name}_grappa_reconstruction_png/{file_id}.{slice_idx}.png", reconstruction_slice, cmap="gray")

    del grappa_reconstruction
    del reconstruction_slice

    num_coils = kspace.shape[2] #  (averages, slices, coils, readout, phase)
    
    # save our reconstruction # compute
    kspace_sum = kspace[0, :, :, :] + kspace[1, :, :, :]
    kspace_sum = T.to_tensor(kspace_sum)
    kspace_sum = fastmri.ifft2c(kspace_sum)
    kspace_sum = fastmri.complex_abs(kspace_sum)
    kspace_sum = fastmri.rss(kspace_sum, dim=1)
    kspace_sum = torch.flip(kspace_sum, dims=[1])
    kspace_sum = center_crop_im(kspace_sum, crop_to_size=(320, 320))

    for slice_idx in range(num_slices):
        kspace_sum_slice = kspace_sum[slice_idx]
        np.save(f"{root_path}/{split_name}_sum_reconstruction_numpy/{file_id}.{slice_idx}.npy", kspace_sum_slice)
        matplotlib.image.imsave(f"{root_path}/{split_name}_sum_reconstruction_png/{file_id}.{slice_idx}.png", kspace_sum_slice, cmap="gray")

    del kspace_sum
    del kspace_sum_slice
    
    kspace_mask = cartesian_mask(
        [num_slices, kspace.shape[-2], kspace.shape[-1]],
        acc=cartesian_mask_acc,
        sample_n=cartesian_mask_sample_n,
        centered=cartesian_mask_centered
    )
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

        mask_slice = kspace_mask[0, slice_idx, 0]
        kspace_sum_masked_slice = kspace_sum_masked[slice_idx]
        np.save(f"{root_path}/{split_name}_mask_numpy/{file_id}.{slice_idx}.npy", mask_slice)
        np.save(f"{root_path}/{split_name}_masked_sum_reconstruction_numpy/{file_id}.{slice_idx}.npy", kspace_sum_masked_slice)

        matplotlib.image.imsave(f"{root_path}/{split_name}_mask_png/{file_id}.{slice_idx}.png", mask_slice, cmap="gray")
        matplotlib.image.imsave(f"{root_path}/{split_name}_masked_sum_reconstruction_png/{file_id}.{slice_idx}.png", kspace_sum_masked_slice, cmap="gray")

    file.close()
    del kspace
    del kspace_mask
    del kspace_masked
    del kspace_sum_masked
    del mask_slice
    del kspace_sum_masked_slice
    
    print("finished saving calculations")
