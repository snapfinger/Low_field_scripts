#!/usr/bin/env python
# coding: utf-8

import numpy as np

import pydicom as dicom
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, show, plot_roi
from nilearn.input_data import NiftiSpheresMasker
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn import input_data

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix

import nibabel as nib

import scipy.io


# t_r = 3.66
# n_scans = 80

t_r = 2
# n_scans = 360
n_scans = 450


rest_fmri_path = '/hd2/research/EEG-fMRI/code/low_field_exp/output/resting.nii'
# rest_fmri_path = '/hd2/research/EEG-fMRI/data/Low_field_fMRI/sub_2021_12_03/sub2_recon2_rsfMRI1_TR2_len300.nii'
# rest_fmri_path = '/hd2/research/EEG-fMRI/data/Low_field_fMRI/sub_2021_12_03/sub2_recon2_rsfMRI1_TR2_len360.nii'

# rest_fmri_path = '/hd2/research/EEG-fMRI/data/Low_field_fMRI/sub_2021_12_03/20211203_110025ep2dbold15minrestings007a001.nii.gz'

rest_fmri_path = '/hd2/research/EEG-fMRI/data/Low_field_fMRI/sub_2021_12_03/sub2_recon2_rsfMRI1_TR%s_len%s.nii' % (t_r, n_scans)
rest = nib.load(rest_fmri_path)

slice_time_ref = 0.

# pcc_coords = (32, 23, 14)
# justin
# pcc_coords = (8.5, 2, 2)
# siemens
pcc_coords = (8, 2, -3)
fwhm=4
seed_radius=3
if_detrend=False



# masker = NiftiMasker(smoothing_fwhm=8, memory='nilearn_cache', memory_level=1,
#                      mask_strategy='epi', standardize=True)
# rest_masked = masker.fit_transform(rest)
# print(rest_masked.shape)



# # compute brain mask
# mask_rest = compute_epi_mask(rest_fmri_path, connected=True, opening=1)
# plot_roi(mask_rest, mean_img(rest_fmri_path));
# plt.show()


# # plot mean image
# plot_epi(mean_img(rest_fmri_path), cmap='gray', cut_coords=(8.5, 3, 3));
# plt.show()


display = plot_epi(mean_img(rest_fmri_path), cmap='gray', cut_coords=pcc_coords);
display.add_markers(marker_coords=[pcc_coords], marker_color='y', marker_size=400)
plt.show()


# extract default mode network
seed_masker = NiftiSpheresMasker([pcc_coords], smoothing_fwhm=fwhm, radius=seed_radius, detrend=if_detrend, standardize=True, low_pass=0.5, high_pass=0.00001, t_r=t_r, \
                                memory='nilearn_cache', memory_level=1, verbose=0)
seed_time_series = seed_masker.fit_transform(rest_fmri_path)

print(seed_time_series.shape)


# scipy.io.savemat('pcc_time_series_ori.mat', {'series': seed_time_series})
# plt.plot(np.arange(n_scans), seed_time_series)
# plt.show()


brain_masker = NiftiMasker(smoothing_fwhm=fwhm, detrend=if_detrend, standardize=True, low_pass=0.1, high_pass=0.01, t_r=t_r,
                            mask_strategy='epi', memory='nilearn_cache', memory_level=1, verbose=0)
brain_time_series = brain_masker.fit_transform(rest)
print("rest mask shape: ", brain_time_series.shape)


# # print(seed_time_series.shape)
# # print(brain_time_series.shape)


seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) /
                              seed_time_series.shape[0])

# print("seed_to_voxel_correlations shape", seed_to_voxel_correlations.shape)


seed_to_voxel_correlations_img = brain_masker.inverse_transform(
    seed_to_voxel_correlations.T)
display = plotting.plot_stat_map(seed_to_voxel_correlations_img,
                                 bg_img=mean_img(rest_fmri_path),
                                 threshold=0.25, vmax=1,
                                 cut_coords=pcc_coords,
                                 title="Seed-to-voxel correlation (PCC seed)")
plt.show()

# # display.add_markers(marker_coords=pcc_coords, marker_color='g',
# #                     marker_size=300)
# # # # At last, we save the plot as pdf.
# # # display.savefig('pcc_seed_correlation.pdf')
