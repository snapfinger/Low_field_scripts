# script to find activated area with a coarse preprocessing + GLM
# by Yijun

import numpy as np
from numpy import array
import pandas as pd

import pydicom as dicom
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, show, plot_roi, plot_design_matrix
from nilearn.plotting import plot_stat_map, plot_anat, plot_img

from nilearn.input_data import NiftiSpheresMasker
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn import input_data

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix

import nibabel as nib
from scipy.stats import pearsonr


task_fmri_path = '/hd2/research/EEG-fMRI/code/low_field_exp/output/task.nii'
task_fmri = nib.load('/hd2/research/EEG-fMRI/code/low_field_exp/output/task.nii')
print("task fMRI shape: ", task_fmri.shape)

n_scans = 80 # number of time points in fMRI
t_r = 3.66
frame_times = np.arange(n_scans) * t_r
print("frame_times: ", len(frame_times))
slice_time_ref = 0.

if_detrend=True
pcc_coords = (20, 24, 20)
fwhm=4
seed_radius=3
lpf_freq=0.01


# average fMRI across time, this is just to provide a reference image for visualization
task_mean_img = mean_img(task_fmri_path)
display = plot_epi(task_mean_img, cmap='gray', cut_coords=pcc_coords);
display.add_markers(marker_coords=[pcc_coords], marker_color='y', marker_size=30)
# plt.show()


# ROI to check activation with preprocessing
seed_masker = NiftiSpheresMasker([pcc_coords], smoothing_fwhm=fwhm, radius=seed_radius, detrend=if_detrend, standardize=True, high_pass=lpf_freq, t_r=t_r, memory='nilearn_cache', memory_level=1, verbose=0)
seed_time_series = seed_masker.fit_transform(task_fmri_path)
print(seed_time_series.shape)
plt.show()


# construct design_matrix
hrf_model = 'spm'
onsets = np.array([0, 10, 20, 30, 40, 50, 60, 70]) * t_r
duration = 10 * t_r * np.ones(len(onsets))
conditions = ['rest', 'active', 'rest', 'active', 'rest', 'active', 'rest', 'active']
conditions2 = {
    'active': array([1., 0., 0., 0, 0, 0, 0 ,0]),
    'rest':   array([0., 1., 0., 0, 0, 0, 0 ,0])}
events = pd.DataFrame({'onset': onsets,
                       'duration': duration,
                       'trial_type': conditions})
print(events)
# events.to_csv('sub_111921_finger_tapping_events.csv')

print("start fitting")
print("task fmri shape:", task_fmri.shape)
drift_model='cosine'
fmri_glm = FirstLevelModel(t_r=t_r,
                           noise_model='ar1',
                           smoothing_fwhm=fwhm,
                           standardize=True,
                           hrf_model=hrf_model,
                           drift_model=drift_model,
                           high_pass=lpf_freq,
                           signal_scaling=False,
                           minimize_memory=False)
fmri_glm = fmri_glm.fit(task_fmri, events)


design_matrix = fmri_glm.design_matrices_[0]
print("design_matrix shape", design_matrix.shape)
print(design_matrix['active'].values.shape)
print("activate length:", design_matrix['active'].shape)
print("correlation", pearsonr(design_matrix['active'].values.flatten(), 2*seed_time_series.flatten()))
plt.plot(np.arange(n_scans), 2*design_matrix['active'])
plt.plot(np.arange(n_scans), seed_time_series)
plt.xlabel('scan')
plt.title('Real v.s. Expected Motor Response')
plt.show()