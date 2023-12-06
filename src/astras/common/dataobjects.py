# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:49:32 2020

@author: bittmans
"""
import pickle
import re
import os
import json

import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq
import scipy.io as sio
from scipy.signal import savgol_filter, cwt, ricker, morlet2
from scipy.ndimage import gaussian_filter


import numpy as np
import traceback
import lmfit
from collections import OrderedDict
from .mpl import (plot, pcolor, fill_between, text_in_plot, PlotColorManager2D,
                  set_line_cycle_properties, get_fill_list)
from .fitmethods import LineFit, GlobalFit


def import_scan_file(fileName):
    # function for importing from json files
    with open(fileName) as binfile:
        compiledScan = json.load(binfile)
        binfile.close()
    wl = np.asarray(compiledScan[0], dtype=np.float64)  # wavelengths
    tdlay = np.asarray(compiledScan[1], dtype=np.int32)  # time delays
    dODScan = np.asarray(compiledScan[2], dtype=np.float64)  # delta OD
    try:
        bkgd = np.asarray(compiledScan[3], dtype=np.float64)  # bkgd
    except Exception:
        bkgd = np.zeros(len(wl))
    try:
        power = np.asarray(compiledScan[4], dtype=np.float64)
    except Exception:
        power = np.zeros(len(tdlay))
    return dODScan, wl, tdlay, bkgd, power


def savgol_filter_wrap(dat, order=5, framelength=3):
    return savgol_filter(dat, order, framelength)


# %%
""" TA data object """


class TAData:
    def __init__(self, time_delay_precision=10, time_unit='ps',
                 input_time_conversion_factor=1e3):
        self.time_delay_precision = time_delay_precision
        self.input_time_conversion_factor = input_time_conversion_factor

        self.ta_map = []
        self.timesteps = []
        self.raw_scans = {}
        self.cmap = plt.get_cmap('RdBu_r')

        self.load_dict = {'.mat': self.load_scan_mat,
                          '.scan': self.load_scan_json,
                          '.txt': self.load_scan_txt,
                          '.dat': self.load_scan_txt}
        self._is_loaded = False
        self.pixel_binned = False
        self.chirp_corrected = False
        self.init_common_attributes(time_unit=time_unit)
        self._init_children()
        self.set_ylabel()

    def _init_children(self):
        self.integral = TATrace()
        self.centroid = TATrace()
        self.plot_data = DataToPlot()
        self.color = PlotColorManager2D()

    def init_common_attributes(self, obj=None, time_unit='ps'):
        if not obj:
            obj = self
        obj.time_unit = time_unit
        obj.time_unit_factors = {'ps': [1, 'fs'],
                                 'fs': [1e-3, 'as'],
                                 'ns': [1e3, 'ps'],
                                 'us': [1e6, 'ns'],
                                 'as': [1e-6, 'zs'],
                                 'ms': [1e9, 'us']}
        obj.x_unit_dict = {'wavelength': {'nm': 1,
                                          'pm': 1e-3,
                                          'um': 1e3,
                                          'mm': 1e6,
                                          'A': 1e-1},
                           'wavenumber': {'cm^(-1)': 1,
                                          'm^(-1)': 1e-2},
                           'energy': {'eV': 1,
                                      'keV': 1e-3,
                                      'MeV': 1e-6,
                                      'meV': 1e3,
                                      'mJ': 6.242e15,
                                      'J': 6.242e18}}

        obj.delA_unit_dict = {'mOD': 1, 'OD': 1e3}
        obj.spec_axis = None
        obj._init_spec_conversion_dict()
        obj.wavenumbers = None
        obj.time_delay_factor = 1
        obj.wavelength_shift = 0
        obj.spectral_modifier = 1
        obj.delA_modifier = 1
        obj.delA_unit = 'mOD'
        obj.zlabel = obj.delA_unit.join(["$\Delta$ Abs. (", ")"])
        obj._xmode = 'wavelength'
        obj.spec_unit = 'nm'
        obj.set_xlabel()

    """ File Loading """
    # Master load functions
    # Loading data from single file
    def load_single_scan(self, path, filetype='.mat'):
        try:
            self.delA, self.wavelengths, self.time_delays, self.power = (
                self.load_dict[filetype](path, load_all=True))
        except Exception:
            raise
        else:
            self.files = 0
            self._write_obj_properties()

    # Averaging multiple scans
    # Master function: Loading and averaging
    def load_and_average_scans(self, path, filetypes=['.mat'],
                               interpolate_nan=True, power_weight=False,
                               remove_outliers=False, outlier_threshold=2,
                               outlier_step=1,
                               outlier_mode='Stat_ROI',
                               wavelength_of_interest=440,
                               wavelength_of_interest_window=10,
                               progressbar=None, files_to_average='all',
                               nan_wl_window=None):
        if progressbar is not None:
            if not progressbar.get_timer_started():
                progressbar.start_timer()
        all_timesteps_equal, all_timesteps_same = self.read_scan_files(
            path, filetypes, progressbar=progressbar)
        outliers = self.average_scans(
            all_timesteps_equal, all_timesteps_same,
            interpolate_nan=interpolate_nan,
            power_weight=power_weight,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold,
            outlier_step=outlier_step,
            outlier_mode=outlier_mode,
            nan_wl_window=nan_wl_window,
            wavelength_of_interest=wavelength_of_interest,
            wavelength_of_interest_window=wavelength_of_interest_window,
            progressbar=progressbar, files_to_average=files_to_average)
        return outliers

    # Loading files in directory
    def read_scan_files(self, path, filetypes, progressbar=None,
                        import_subfolders=True):
        self.files = [f for f in os.listdir(
            path) if re.search('(\\.mat|\\.scan|\\.txt)', f)]
        if import_subfolders:
            subfolders = [f for f in os.listdir(
                path) if not re.search('\.', f)]
            for subdir in subfolders:
                files = self._get_scans_from_folders_recursion(path, subdir)
                self.files.extend(files)
        self.raw_scans = {}
        self.timesteps = []
        all_timesteps_equal = True
        all_timesteps_same = True
        files = []
        if progressbar is not None:
            progressbar.update_label('Loading Scans')
            progressbar.reset_max(len(self.files))
            progressbar.update_timer()
        for f in self.files:
            correct_type = False
            for t in filetypes:
                if re.search(t, f):
                    filetype = t
                    correct_type = True
                    break
            if not correct_type:
                continue
            files.append(f)
            delA, wavelengths, time_delays, power = self.load_dict[filetype](
                path + '\\' + f, load_all=True)
            current_t_step = np.round(np.abs(time_delays[1]-time_delays[0]), 6)
            if current_t_step not in self.timesteps:
                self.timesteps.append(current_t_step)
            self.raw_scans[f] = {'delA': delA,
                                 'wavelengths': wavelengths,
                                 'time_delays': time_delays,
                                 'power': power,
                                 'first_time_step': current_t_step,
                                 'filetype': filetype,
                                 'power_weighted': False}
            if all_timesteps_equal:
                for i in range(1, len(time_delays)):
                    if (np.round(np.abs(time_delays[i]-time_delays[i - 1]), 6)
                            != current_t_step):
                        all_timesteps_equal = False
                        break
            if progressbar is not None:
                progressbar.increase_value()
                progressbar.update_timer()
        self.files = files
        td_check = self.raw_scans[self.files[0]]['time_delays']
        for i, f in enumerate(self.files[1:]):
            if (np.shape(self.raw_scans[f]['time_delays'])
                    != np.shape(self.raw_scans[self.files[i]]['time_delays'])):
                all_timesteps_same = False
                break
            td_check = (self.raw_scans[f]['time_delays']
                        - self.raw_scans[self.files[i]]['time_delays'])
            if np.sum(td_check) != 0:
                all_timesteps_same = False
                break
        self.timesteps.sort()
        self.wavelengths = wavelengths
        return all_timesteps_equal, all_timesteps_same

    def _get_scans_from_folders_recursion(self, path, subdir):
        files = [subdir + '/' + f for f in os.listdir(
            path + '/' + subdir) if re.search('(\\.mat|\\.scan|\\.txt)', f)]
        subfolders = [
            subdir + '/' + f for f in os.listdir(path + '/' + subdir)
            if not re.search('\.', f)]
        for subfolder in subfolders:
            new_files = self._get_scans_from_folders_recursion(path, subfolder)
            files.extend(new_files)
        return files

    # Averaging
    def average_scans(self, all_timesteps_equal, all_timesteps_same,
                      power_weight=False, **averaging_options):
        self.toggle_power_weight(weight=power_weight)
        if all_timesteps_same:
            outliers = self._average_scan_matrices(**averaging_options)
        else:
            outliers = self._average_point_by_point(**averaging_options)
        if outliers is not False:
            self._write_obj_properties()
        return outliers

    # If weighting by pump power is possible this can be done or reversed here
    def toggle_power_weight(self, weight=True):
        for scan in self.raw_scans.values():
            if scan['power_weighted'] != weight:
                av_pwr = np.mean(scan['power'])
                if weight:
                    for i, pwr in enumerate(scan['power']):
                        scan['delA'][i, :] = scan['delA'][i, :] / pwr * av_pwr
                else:
                    for i, pwr in enumerate(scan['power']):
                        scan['delA'][i, :] = scan['delA'][i, :] * pwr / av_pwr
                scan['power_weighted'] = weight

    # Adding delta Abs. Matrices, assuming same time delay array for all scans
    def _average_scan_matrices(self, interpolate_nan=True, nan_wl_window=None,
                               remove_outliers=False, outlier_threshold=2,
                               outlier_step=1,
                               outlier_mode='Stat_ROI',
                               wavelength_of_interest=440,
                               wavelength_of_interest_window=10,
                               progressbar=None, files_to_average='all',
                               manual_outliers={}):
        def differential_outlier_removal_pixel(i, j):
            values = []
            for f in files_local:
                try:
                    vals = [self.raw_scans[f]['delA'][i-outlier_step, j],
                            self.raw_scans[f]['delA'][i, j],
                            self.raw_scans[f]['delA'][i+outlier_step, j]]
                    val_diff = np.diff(vals)
                except Exception:
                    pass
                else:
                    if not ((np.abs(val_diff[0]) > outlier_threshold
                             and np.abs(val_diff[-1]) > outlier_threshold)
                            or np.isnan(vals[1])):
                        values.append(vals[1])
            self.delA[i, j] = np.nanmean(values)
            return len(files_to_average) - len(values)

        def differential_outlier_removal_roi(i):
            check = []
            k = 0
            for f in files_to_average:
                k += 1
                try:
                    roi = []
                    for j in range(3):
                        val = self.raw_scans[f]['delA'][
                            i+(j-1)*outlier_step,
                            wavelength_of_interest_index
                            - wavelength_of_interest_window:
                                wavelength_of_interest_index
                                + wavelength_of_interest_window]
                        roi.append(np.mean(val))
                    vals = np.diff(roi)
                except Exception as e:
                    print(e)
                    check.append(False)
                else:
                    if (np.abs(vals[0]) > outlier_threshold
                            and np.abs(vals[-1]) > outlier_threshold):
                        check.append(False)
                    elif np.isnan(roi[1]):
                        check.append(False)
                    else:
                        check.append(True)
            count = 0
            outliers = []
            for j, f in enumerate(files_to_average):
                if check[j]:
                    self.delA[i, :] += self.raw_scans[f]['delA'][i, :]
                    count += 1
                else:
                    outliers.append(files_to_average[j])
            if count > 0:
                self.delA[i, :] = self.delA[i, :] / count
            else:
                self.delA[i, :] = np.nan
            return outliers

        def statistical_outlier_removal_pixel(i, j):
            vals = [self.raw_scans[f]['delA'][i, j] for f in files_local]
            pixel_values = []
            for k in range(len(vals)):
                mean_array = [v for ind, v in enumerate(vals) if ind != k]
                if not ((np.abs(vals[k] - np.mean(mean_array))
                        > outlier_threshold * np.std(mean_array))
                        or np.isnan(vals[k])):
                    pixel_values.append(vals[k])
            self.delA[i, j] = np.nanmean(pixel_values)
            return len(files_to_average) - len(pixel_values)

        def statistical_outlier_removal_roi(i):
            vals = []
            for f in files_to_average:
                val = self.raw_scans[f]['delA'][
                    i, wavelength_of_interest_index
                    - wavelength_of_interest_window:
                        wavelength_of_interest_index
                        + wavelength_of_interest_window]
                vals.append(np.mean(val))
            count = 0
            outliers = []
            for j in range(len(vals)):
                if i in manual_outliers[files_to_average[j]]:
                    outliers.append(files_to_average[j])
                    continue
                mean_array = [v for k, v in enumerate(vals) if k != j]
                if not ((np.abs(vals[j] - np.nanmean(mean_array))
                        > (outlier_threshold * np.std(mean_array)))
                        or np.isnan(vals[j])):
                    self.delA[i, :] += np.ma.array(
                        self.raw_scans[files_to_average[j]]['delA'][i, :],
                        mask=np.isnan(
                            self.raw_scans[files_to_average[j]]['delA'][i, :]))
                    count += 1
                else:
                    outliers.append(files_to_average[j])
            if count > 0:
                self.delA[i, :] = self.delA[i, :] / count
            else:
                self.delA[i, :] = np.nan
            return outliers

        def average_first_and_last_point():
            for f in files_to_average:
                self.delA[0, :] += (self.raw_scans[f]['delA'][0, :]
                                    / (len(files_to_average)))
                self.delA[-1, :] += (self.raw_scans[f]['delA'][-1, :]
                                     / (len(files_to_average)))

        if progressbar is not None:
            if not progressbar.get_timer_started():
                progressbar.start_timer()
        if files_to_average == 'all':
            files_to_average = self.files
        try:
            if len(files_to_average) == 0:
                return False
        except Exception:
            return False
        for f in files_to_average:
            if f not in manual_outliers.keys():
                manual_outliers[f] = []

        self.time_delays = self.raw_scans[files_to_average[0]]['time_delays']
        self.delA = np.zeros(
            np.shape(self.raw_scans[files_to_average[0]]['delA']))

        if remove_outliers:
            if progressbar is not None:
                progressbar.update_label('Removing outliers')
                progressbar.reset_max(np.shape(self.delA)[0])
            outliers = {}
            if re.search('pixel', outlier_mode, re.I):
                if re.search('deriv', outlier_mode, re.I):
                    num_outliers = 0
                    for i in range(1, np.shape(self.delA)[0]-1):
                        files_local = []
                        for f in files_to_average:
                            if i not in manual_outliers[f]:
                                files_local.append(f)
                        for j in range(np.shape(self.delA)[1]):
                            num_outliers += differential_outlier_removal_pixel(
                                i, j)
                        if progressbar is not None:
                            if progressbar.get_cancelled():
                                return False
                            progressbar.increase_value()
                            progressbar.update_timer()
                    outliers['total'] = num_outliers
                    average_first_and_last_point()
                    if progressbar is not None:
                        progressbar.update_value(np.shape(self.delA)[0])
                else:
                    num_outliers = 0
                    for i in range(np.shape(self.delA)[0]):
                        files_local = []
                        for f in files_to_average:
                            if i not in manual_outliers[f]:
                                files_local.append(f)
                        for j in range(np.shape(self.delA)[1]):
                            num_outliers += (
                                statistical_outlier_removal_pixel(i, j))
                        if progressbar is not None:
                            if progressbar.get_cancelled():
                                return False
                            progressbar.increase_value()
                            progressbar.update_timer()
                    outliers['total'] = num_outliers
            else:
                try:
                    wavelength_of_interest_index = np.where(
                        self.wavelengths >= wavelength_of_interest)[0][0]
                except Exception:
                    return False
                if re.search('deriv', outlier_mode, re.I):
                    for f in files_to_average:
                        outliers[f] = []
                    for i in range(1, np.shape(self.delA)[0]-1):
                        outlier = differential_outlier_removal_roi(i)
                        for out in outlier:
                            outliers[out].append(i)
                        if progressbar is not None:
                            if progressbar.get_cancelled():
                                return False
                            progressbar.increase_value()
                            progressbar.update_timer()
                    average_first_and_last_point()
                    if progressbar is not None:
                        progressbar.update_value(np.shape(self.delA)[0])
                else:
                    for f in files_to_average:
                        outliers[f] = []
                    for i in range(np.shape(self.delA)[0]):
                        outlier = statistical_outlier_removal_roi(i)
                        for out in outlier:
                            outliers[out].append(i)
                        if progressbar is not None:
                            if progressbar.get_cancelled():
                                return False
                            progressbar.increase_value()
                            progressbar.update_timer()
        else:
            outliers = None
            if progressbar is not None:
                progressbar.reset_max(len(self.time_delays))
                progressbar.update_label('Averaging')
            for i in range(len(self.time_delays)):
                array_to_average = []
                for f in files_to_average:
                    if i in manual_outliers[f]:
                        continue
                    array_to_average.append(self.raw_scans[f]['delA'][i, :])
                self.delA[i, :] = np.nanmean(
                    np.asarray(array_to_average), axis=0)
                if progressbar is not None:
                    if progressbar.get_cancelled():
                        return False
                    progressbar.increase_value()
                    progressbar.update_timer()
        if interpolate_nan:
            try:
                xwindow = [
                    np.where(self.wavelengths >= nan_wl_window[0])[0][0],
                    np.where(self.wavelengths <= nan_wl_window[1])[0][-1]]
            except Exception:
                xwindow = None
            self.nan_filter(xwindow=xwindow, progressbar=progressbar)
        return outliers

    def nan_filter(self, xwindow=None, ywindow=None, maxiter=30,
                   progressbar=None):
        if not np.any(np.isnan(self.delA)):
            return None
        if progressbar is not None:
            progressbar.reset_max(np.shape(self.delA)[0])
            progressbar.update_label('Removing NaN')
        if not xwindow:
            xiter = range(1, np.shape(self.delA)[1] - 1)
        else:
            xiter = range(*xwindow)
        if not ywindow:
            yiter = range(1, np.shape(self.delA)[0] - 1)
        else:
            yiter = range(*ywindow)
        for i in yiter:
            if progressbar is not None:
                if progressbar.get_cancelled():
                    return False
                progressbar.increase_value()
                progressbar.update_timer()
            if not np.any(np.isnan(self.delA[i, :])):
                continue
            for j in xiter:
                k = 1
                while np.isnan(self.delA[i, j]):
                    try:
                        self.delA[i, j] = np.nanmean(
                            [self.delA[i + n, j + m]
                             for n in range(-k, k + 1)
                             for m in range(-k, k + 1)])
                    except Exception:
                        break
                    k += 1
                    if k > 30:
                        break

    # Averaging and/or concatenating scans with different time points
    def _average_point_by_point(self, interpolate_nan=True,
                                nan_wl_window=None, remove_outliers=False,
                                outlier_threshold=2, outlier_step=1,
                                outlier_mode='Stat_ROI',
                                wavelength_of_interest=440,
                                wavelength_of_interest_window=10,
                                progressbar=None, files_to_average='all',
                                manual_outliers={}):
        def statistical_outlier_removal_pixel():
            outliers = {}
            if progressbar is not None:
                progressbar.reset_max(len(self.time_delays))
            for f in files_to_average:
                outliers[f] = []
            for i in range(len(self.time_delays)):
                if progressbar is not None:
                    if progressbar.get_cancelled():
                        return False
                    progressbar.update_timer()
                    progressbar.increase_value()
                scans = delA_dict[self.time_delays[i]]
                for j in range(pixel_no):
                    vals = []
                    keys = []
                    for f, scan in scans.items():
                        vals.append(scan[j])
                        keys.append(f)
                    counter = 0
                    for k in range(len(vals)):
                        mean_array = [v for k, v in enumerate(vals) if k != j]
                        if not ((np.abs(vals[k] - np.mean(mean_array))
                                > (outlier_threshold * np.std(mean_array)))
                                or np.isnan(vals[k])):
                            self.delA[i, j] += vals[k]
                            counter += 1
                        else:
                            outliers[keys[k]].append([i, j])
                    if counter > 1:
                        self.delA[i, j] = self.delA[i, j]/counter
                    elif counter == 0:
                        self.delA[i, j] = np.nan
            return outliers

        def differential_outlier_removal_pixel():
            outliers = {}
            counter = np.zeros(np.shape(self.delA))
            if progressbar is not None:
                progressbar.reset_max(len(files_to_average))
            for f in files_to_average:
                outliers[f] = []
                time_delays = self.raw_scans[f]['time_delays']
                for i in range(1, len(time_delays) - 1):
                    if i in manual_outliers[f]:
                        continue
                    if progressbar is not None:
                        if progressbar.get_cancelled():
                            return False
                        progressbar.update_timer()
                    scan = delA_dict[time_delays[i]][f]
                    ind = np.where(self.time_delays == time_delays[i])[0][0]
                    for j in range(pixel_no):
                        vals = []
                        for k in range(3):
                            vals.append(
                                delA_dict[time_delays[i + k - 1]][f][j])
                        vals = np.diff(vals)
                        if not (np.abs(vals[0]) > outlier_threshold
                                and np.abs(vals[-1]) > outlier_threshold
                                or np.isnan(scan[j])):
                            self.delA[ind, j] = self.delA[ind, j] + scan[j]
                            counter[ind, j] += 1
                if progressbar is not None:
                    progressbar.update_timer()
                for j in range(pixel_no):
                    self.delA[0, j] += delA_dict[time_delays[0]][f][j]
                    counter[0, j] += 1
                    self.delA[-1, j] += delA_dict[time_delays[-1]][f][j]
                    counter[-1, j] += 1
                if progressbar is not None:
                    progressbar.increase_value()
            if progressbar is not None:
                progressbar.reset_max(len(self.time_delays))
                progressbar.update_label('Averaging')
            for i in range(len(self.time_delays)):
                for j in range(pixel_no):
                    if counter[i, j] > 1:
                        self.delA[i, j] = self.delA[i, j]/counter[i, j]
                    elif counter[i, j] == 0:
                        self.delA[i, j] = np.nan
                if progressbar is not None:
                    if progressbar.get_cancelled():
                        return False
                    progressbar.update_timer()
                    progressbar.increase_value()
            return {}

        def statistical_outlier_removal_roi():
            outliers = {}
            for f in files_to_average:
                outliers[f] = []
            if progressbar is not None:
                progressbar.reset_max(len(self.time_delays))
            for i in range(len(self.time_delays)):
                if progressbar is not None:
                    if progressbar.get_cancelled():
                        return False
                    progressbar.update_timer()
                    progressbar.increase_value()
                vals = []
                keys = []
                scans = delA_dict[self.time_delays[i]]
                for key, val in scans.items():
                    keys.append(key)
                    vals.append(
                        np.mean(val[
                            wavelength_of_interest_index
                            - wavelength_of_interest_window:
                                wavelength_of_interest_index
                                + wavelength_of_interest_window]))
                if len(vals) > 2:
                    for j in range(len(vals)):
                        mean_array = [v for k, v in enumerate(vals) if k != j]
                        if ((np.abs(vals[j] - np.mean(mean_array))
                             > (outlier_threshold * np.std(mean_array)))
                                or np.isnan(vals[j])):
                            del scans[keys[j]]
                            outliers[keys[j]].append(i)
            return outliers

        def differential_outlier_removal_roi():
            outliers = {}
            if progressbar is not None:
                progressbar.reset_max(len(files_to_average))
            for f in files_to_average:
                if progressbar is not None:
                    progressbar.increase_value()
                outlier = []
                scan = self.raw_scans[f]
                for i in range(1, np.shape(scan['delA'])[0] - 1):
                    if i in manual_outliers[f]:
                        continue
                    if progressbar is not None:
                        if progressbar.get_cancelled():
                            return False
                        progressbar.update_timer()
                    roi = []
                    for j in range(3):
                        val = scan['delA'][
                            i+(j-1)*outlier_step,
                            wavelength_of_interest_index
                            - wavelength_of_interest_window:
                                wavelength_of_interest_index
                                + wavelength_of_interest_window]
                        roi.append(np.mean(val))
                    vals = np.diff(roi)
                    if (np.abs(vals[0]) > outlier_threshold
                            and np.abs(vals[-1]) > outlier_threshold
                            or np.isnan(roi[1])):
                        del delA_dict[scan['time_delays'][i]][f]
                        outlier.append(i)
                outliers[f] = outlier
            return outliers

        if files_to_average == 'all':
            files_to_average = self.files
        try:
            if len(files_to_average) == 0:
                return False
        except Exception:
            return False
        if progressbar is not None:
            if not progressbar.get_timer_started():
                progressbar.start_timer()
            progressbar.update_label('Averaging: Sorting time delays.')

        for f in files_to_average:
            if f not in manual_outliers.keys():
                manual_outliers[f] = []

        self.time_delays = list(
            self.raw_scans[files_to_average[0]]['time_delays'])
        delA_dict = {}
        for i, td in enumerate(
                self.raw_scans[files_to_average[0]]['time_delays']):
            delA_dict[td] = {
                files_to_average[0]:
                    self.raw_scans[files_to_average[0]]['delA'][i, :]}
        for i in range(1, len(files_to_average)):
            scan = self.raw_scans[files_to_average[i]]
            for j, td in enumerate(scan['time_delays']):
                if td in self.time_delays:
                    delA_dict[td][files_to_average[i]] = scan['delA'][j, :]
                else:
                    self.time_delays.append(td)
                    delA_dict[td] = {files_to_average[i]: scan['delA'][j, :]}

        self.time_delays.sort()
        pixel_no = np.shape(self.raw_scans[files_to_average[0]]['delA'])[1]
        self.delA = np.zeros((len(self.time_delays), pixel_no))

        for f, manual_out in manual_outliers.items():
            for ind in manual_out:
                td = self.raw_scans[f]['time_delays'][ind]
                del delA_dict[td][f]
        if progressbar is not None:
            progressbar.update_timer()
            if remove_outliers:
                progressbar.update_label('Averaging: Removing Outliers.')

        if remove_outliers and re.search('pixel', outlier_mode, re.I):
            if re.search('Stat', outlier_mode, re.I):
                outliers = statistical_outlier_removal_pixel()
            else:
                outliers = differential_outlier_removal_pixel()
        else:
            if remove_outliers and re.search('ROI', outlier_mode, re.I):
                try:
                    wavelength_of_interest_index = np.where(
                        self.raw_scans[files_to_average[0]]['wavelengths']
                        >= wavelength_of_interest)[0][0]
                except Exception:
                    raise
                if re.search('Stat', outlier_mode, re.I):
                    outliers = statistical_outlier_removal_roi()
                else:
                    outliers = differential_outlier_removal_roi()
            else:
                outliers = None
            if progressbar is not None:
                progressbar.update_timer()
                progressbar.update_label('Averaging.')
                progressbar.reset_max(len(self.time_delays))
            for i, td in enumerate(self.time_delays):
                self.delA[i, :] = np.nanmean(np.asarray(
                    [val for val in delA_dict[td].values()]), axis=0)
                if progressbar is not None:
                    progressbar.update_timer()
                    progressbar.increase_value()
        self.time_delays = np.array(self.time_delays)
        if interpolate_nan:
            try:
                xwindow = [
                    np.where(self.wavelengths >= nan_wl_window[0])[0][0],
                    np.where(self.wavelengths <= nan_wl_window[1])[0][-1]]
            except Exception:
                xwindow = None
            self.nan_filter(progressbar=progressbar, xwindow=xwindow)
        return outliers

    # Secondary loading functions, file type specific
    # Matlab files (.mat)
    def load_scan_mat(self, path, load_all=True):
        loaded_data = sio.loadmat(path, mdict=None, appendmat=True)
        try:
            # case: separate variables for delta Abs,
            # wavelength, time delay
            delA = None
            wavelengths = None
            time_delays = None
            for key in loaded_data.keys():
                if key.lower() in ('da', 'dela', 'dod', 'filta'):
                    delA = np.array(loaded_data[key])
                    break
            for key in loaded_data.keys():
                if re.search('wavel|lambd', key, re.I):
                    wavelengths = np.array(loaded_data[key])
                    break
            for key in loaded_data.keys():
                if re.search('time', key, re.I):
                    time_delays = np.array(loaded_data[key])
            if delA is None:
                raise
            if wavelengths is None:
                wavelengths = np.array(range(np.shape(delA)[1]))
            if time_delays is None:
                time_delays = np.array(range(np.shape(delA)[0]))
            try:
                self.bkg = loaded_data["bkg"]
            except Exception:
                pass
            try:
                power = loaded_data["power"]
            except Exception:
                power = []
            wavelengths = wavelengths.flatten()
            time_delays = time_delays.flatten()
        except Exception:
            # case all in one matrix, same format as for txt files
            try:
                # trying standard variable names (data, vipMat)
                data_mat = loaded_data["data"]
            except Exception:
                try:
                    data_mat = loaded_data["vipMat"]
                except Exception:
                    # trying first variable in file
                    try:
                        for k in loaded_data.keys():
                            if k[0] != '_':
                                data_mat = loaded_data[k]
                                break
                    except Exception:
                        pass
            delA, wavelengths, time_delays, power = self.load_from_matrix(
                data_mat)
        if load_all:
            return delA, wavelengths, time_delays, power
        else:
            return delA

    # JSON object files (see imported method import_scan_file)
    def load_scan_json(self, path, load_all=True):
        delA, wavelengths, time_delays, bkg, power = import_scan_file(path)
        if load_all:
            return (delA*1000, wavelengths[0],
                    time_delays[0].astype(np.float64)
                    / self.input_time_conversion_factor,
                    power[0])
        else:
            return delA*1000

    # .txt and .dat ASCII files
    def load_scan_txt(self, path, load_all=True):
        data_mat, header = self.read_txt_matrix(path)
        delA, wavelengths, time_delays, power = self.load_from_matrix(data_mat)
        if load_all:
            return delA, wavelengths, time_delays, power
        else:
            return delA

    # (potentially) shared loading sub-methods
    def read_txt_matrix(self, fname):
        with open(fname) as f:
            header = []
            line = f.readline()
            while line[0] == "#":
                header.append(line)
                line = f.readline()
            mat = [line]
            mat.extend(f.readlines())
            f.close()
        try:
            mat = np.array([x.strip().split(",")
                           for x in mat]).astype(np.float64)
        except Exception:
            try:
                mat = np.array([x.strip().split("\t") for x in mat])
                mat = mat.astype(np.float64)
            except Exception:
                # mat = None
                raise
        return mat, header

    def load_from_matrix(self, data_mat):
        if data_mat[-1, 0] < data_mat[-2, 0]:
            wavelengths = np.array(data_mat[1:-1, 0])
            delA = np.transpose(data_mat[1:-1, 1:])
        else:
            wavelengths = np.array(data_mat[1:, 0])
            delA = np.transpose(data_mat[1:, 1:])
        time_delays = (np.array(data_mat[0, 1:])
                       / self.input_time_conversion_factor)
        return delA, wavelengths, time_delays, []

    # loading misc. data files
    def load_wavelength_file(self, path):
        with open(path) as f:
            m = np.array(f.readlines())
            f.close()
        self.wavelengths = m.astype(np.float64)
        self.set_wavenumbers()

    def load_power_file(self, path):
        with open(path) as f:
            m = np.array(f.readlines())
            f.close()
        self.power = m.astype(np.float64)

    def load_time_zero_file(self, file):
        with open(file.name) as f:
            m = f.readlines()
            f.close()
        m = np.array([x.strip().split(",") for x in m])
        self.time_zeros = m.astype(np.float64)

    """ post-loading methods """

    def _write_obj_properties(self, delete_nan=True):
        if delete_nan:
            self.delA = self.delA[:, np.isfinite(self.wavelengths)]
            self.delA = self.delA[np.isfinite(self.time_delays), :]
            self.wavelengths = self.wavelengths[np.isfinite(self.wavelengths)]
            self.time_delays = self.time_delays[np.isfinite(self.time_delays)]
        self.raw_delA = self.delA
        self.non_filt_delA = self.delA
        self.time_zero_abs = 0
        self.time_zero_shift = 0
        self.wavelength_shift = 0
        self.time_delays_raw = self.time_delays
        self.time_zeros = []
        self._region_cutout = []
        self.set_wavenumbers()
        self._init_spec_axis_dict()
        self._wavelengths = self.wavelengths
        self.set_xlimits(np.array([
            self.wavelengths[0], self.wavelengths[-1]]))
        self._get_time_steps()
        self._time_range_indices = [0, len(self.time_delays)]
        self._is_loaded = True

    def _init_spec_axis_dict(self, wavelengths=None, wavenumbers=None):
        if wavelengths is not None:
            self.wavelengths = wavelengths
        if wavenumbers is not None:
            self.wavenumbers = wavenumbers
        self.spec_axis = {}
        try:
            self.spec_axis['wavenumber'] = self.wavenumbers
        except Exception:
            self.spec_axis['wavenumber'] = None
            self.spec_axis['energy'] = None
        else:
            self.spec_axis['energy'] = self.wavenumbers*0.000124
        try:
            self.spec_axis['wavelength'] = self.wavelengths
        except Exception:
            self.spec_axis['wavelength'] = None

    def _init_spec_conversion_dict(self):
        # functions for conversion from key to wavenumber ('in')
        # and from wavenumber to key ('out')
        self._spec_conversion = {}
        self._spec_conversion['wavenumber'] = {'in': lambda x: x,
                                               'out': lambda x: x}
        self._spec_conversion['energy'] = {'in': lambda x: x/0.000124,
                                           'out': lambda x: x*0.000124}
        self._spec_conversion['wavelength'] = {
            'in': lambda x: np.power(10, 7)/x,
            'out': lambda x: np.power(10, 7)/x}

    def convert_spec_values(self, x_values, x_in='wavelength',
                            x_out='wavenumber'):
        try:
            x_wavenum = self._spec_conversion[x_in]['in'](x_values)
        except Exception:
            try:
                self._init_spec_conversion_dict()
                x_wavenum = self._spec_conversion[x_in]['in'](x_values)
            except Exception:
                return None
        else:
            try:
                return self._spec_conversion[x_out]['out'](x_wavenum)
            except Exception:
                return None

    def set_xlimits(self, xlimits):
        self.lambda_lim = np.sort(self.convert_spec_values(
            xlimits, x_in=self._xmode, x_out='wavelength'))
        if self.lambda_lim[0] < np.min(self.wavelengths):
            self.lambda_lim[0] = np.min(self.wavelengths)
        if self.lambda_lim[1] > np.max(self.wavelengths):
            self.lambda_lim[1] = np.max(self.wavelengths)
        ind = np.array([0, 1])
        for i in range(2):
            ind[i] = self._get_lambda_index(self.lambda_lim[i])
            if ind[i] is None:
                ind[i] = self.wavelengths[0-i]
        self._lambda_lim_index = ind

    def _get_lambda_index(self, lam):
        try:
            return np.where(self.wavelengths >= lam)[0][0]
        except IndexError:
            return len(self.wavelengths) - 1
        except Exception:
            raise

    def get_xlim_indices(self, limits=None):
        """ returns spectral indices for given limits or current default """
        if limits is None:
            limits = self.lambda_lim
        ind = []
        for lim in limits:
            ind.append(self._get_lambda_index(lim))
        return ind

    def get_lambda_lim(self):
        return [lim for lim in self.lambda_lim]

    def get_xlim(self):
        """ returns current spectral range """
        xlim_wavenum = self._spec_conversion['wavelength']['in'](
            self.lambda_lim)
        xlim = self._spec_conversion[self._xmode]['out'](xlim_wavenum)
        return np.sort(xlim)

    def get_xlim_slice(self):
        xlim = self.get_xlim_indices()
        return slice(xlim[0], xlim[1] + 1)

    def get_x_mode(self):
        mode = self._xmode
        return mode

    def get_is_loaded(self):
        return bool(self._is_loaded)

    def set_time_win_indices(self, limits):
        ind = []
        try:
            for lim in limits:
                a = np.where(self.time_delays >= lim)[0]
                try:
                    ind.append(a[0])
                except Exception:
                    a = np.where(self.time_delays <= lim)[0]
                    ind.append(a[-1])
        except Exception:
            ind = [0]
            ind.append(len(self.time_delays) - 1)
        self._time_range_indices = ind
        self.td_slice = slice(self._time_range_indices[0],
                              self._time_range_indices[1] + 1)
        return ind

    def _get_time_steps(self, td=None, td_round_prec=None):
        if td is None:
            td = self.time_delays
        if td_round_prec is None:
            td_round_prec = self.time_delay_precision
        self.timesteps = []
        for i in range(len(td)-1):
            if np.round(td[i+1] - td[i],
                        td_round_prec) not in self.timesteps:
                self.timesteps.append(np.round(
                    td[i+1] - td[i],
                    td_round_prec))
        self._unequal_time_steps = len(self.timesteps) > 1

    """ file saving """
    def save_plot_data(self, file, plot_data, parent=None):
        with open(file.name, 'wb') as f:
            np.savetxt(f, plot_data, delimiter=',')
            f.close()

    def save_time_zero_file(self, file):
        self.save_plot_data(file, self.time_zeros)

    def save_data_matrix(self, file, matrix=None, x=None, y=None, header='',
                         convert_to_fs=False, dtype='map', labels=None):
        def save_to_mat():
            if dtype == 'map':
                vardict = {'delA': matrix,
                           'time_delays': y,
                           self.get_x_mode(): x,
                           'bkg': []}
            elif dtype == 'trace':
                vardict = {'traces': matrix,
                           'xdata': x,
                           'labels': labels if labels else []}
            sio.savemat(file.name, vardict)

        def save_to_txt():
            if dtype == 'map':
                save_matrix = np.transpose(
                    np.concatenate(([np.insert(x, 0, 100)],
                                    np.transpose(np.concatenate(
                                        ([y], np.transpose(matrix)))))))
            else:
                save_matrix = np.transpose(np.concatenate(([x], matrix)))
            with open(file.name, 'wb') as f:
                np.savetxt(f, save_matrix, delimiter=',', header=header)
                f.close()
        if matrix is None:
            matrix = self.delA
        if y is None:
            y = self.time_delays
        if x is None:
            x = self.spec_axis[self.get_x_mode()]
        if file.name.endswith('.mat'):
            save_to_mat()
        elif file.name.endswith('.txt'):
            if dtype == 'map':
                try:
                    y = (y * self.time_unit_factors[self.time_unit][0]
                         * self.input_time_conversion_factor)
                except Exception:
                    pass
            save_to_txt()

    def save_data_obj(self, fname):
        obj = self
        file = open(fname, 'w')
        pickle.dump(obj, file)

    """  Plot functions """

    def plot_ta_map(self, *args, fig=None, ax=None, dat=None, yvalues=None,
                    xvalues=None, vmin=-10, vmax=10, ymode='values',
                    xmode=None, write_map=False, transpose=False,
                    color_obj=None, **kwargs):
        if not fig:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if not ax:
            ax = fig.add_subplot(111)
        else:
            ax.cla()
        if dat is None:
            dat = self.delA
        if yvalues is None:
            yvalues = self.time_delays
        if not re.search('val', ymode, re.I):
            yvalues = np.arange(len(yvalues))
        if re.search('log', ymode, re.I):
            fnc = re.search('log\d*', ymode, re.I).group(0)
            f = getattr(np, fnc)
            yvalues = np.array([f(y) for y in yvalues])
            dat = dat[np.isfinite(yvalues), :]
            yvalues = yvalues[np.isfinite(yvalues)]
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = self.cmap
        if xvalues is None:
            xvalues = self.spec_axis[self._xmode]
        ta_map = pcolor(ax, xvalues, yvalues, dat, transpose=transpose,
                        vmin=vmin, vmax=vmax, **kwargs)
        if write_map:
            self.ta_map = ta_map
        self.cmap = kwargs['cmap']
        self._mask_cutout_region(ax, transpose=transpose, vmin=vmin,
                                 vmax=vmax, **kwargs)
        return ta_map

    def _mask_cutout_region(self, ax, **kwargs):
        for cut_region in self._region_cutout:
            if cut_region[0].lower() == 'x':
                x = cut_region[1][1]
                y = self.time_delays
                z = np.zeros((len(y), len(x)))
            elif cut_region[0].lower() == 'y':
                y = cut_region[1][1]
                x = self.spec_axis[self._xmode]
                z = np.zeros((len(y), len(x)))
            pcolor(ax, x, y, z, **kwargs)

    def plot_raw_scan(self, filename, *args, yvalues=None, **kwargs):
        try:
            delA = self.raw_scans[filename]['delA']
        except Exception:
            return None
        if yvalues is None:
            yvalues = self.raw_scans[filename]['time_delays']

        return self.plot_ta_map(
            *args, dat=delA, yvalues=yvalues,
            xvalues=self.raw_scans[filename]['wavelengths'],
            write_map=False, **kwargs)

    def get_clim(self, dat, opt='symmetric', contrast=1, offset=0.0):
        if opt == 'symmetric':
            return np.round(np.array([-contrast, contrast])
                            * np.max(np.max(np.abs(
                                np.ma.array(dat, mask=np.isnan(dat)))))
                            * (1+offset))
        elif opt == 'asymmetric':
            return np.round([
                np.min(np.min(
                    np.ma.array(dat, mask=np.isnan(dat))))
                * (1+offset) * contrast,
                np.max(np.max(
                    np.ma.array(dat, mask=np.isnan(dat))))
                * (1+offset) * contrast])
        else:
            print('Invalid option for function get_clim.')
            return [-10, 10]

    # Data modification
    def modify_x(self, unit=None, factor=None):
        x_changed = True
        if unit:
            try:
                unit_factor = (self.x_unit_dict[self._xmode][self.spec_unit]
                               / self.x_unit_dict[self._xmode][unit])
                self.wavelengths = np.array(
                    [wl * unit_factor for wl in self.wavelengths])
            except KeyError:
                print("Invalid Unit " + str(unit))
                unit_factor = 1
            except Exception:
                raise
            else:
                self.spec_unit = unit
                self.set_xlabel()
                x_changed = True
        else:
            unit_factor = 1
        if factor is not None:
            if not self._xmode == 'wavelength':
                fac = 1 / factor
            else:
                fac = factor
            self.wavelengths = np.array([wl * fac for wl in self.wavelengths])
            self.spectral_modifier = self.spectral_modifier * fac
            x_changed = True
        else:
            factor = 1
        if x_changed:
            self.set_wavenumbers()
            self._init_spec_axis_dict()
        return unit_factor * factor

    def shift_wavelengths(self, shift):
        self.wavelengths = self.wavelengths + shift - self.wavelength_shift
        self._init_spec_axis_dict(wavenumbers=self.set_wavenumbers())
        self.wavelength_shift = shift

    def shift_timezero(self, shift):
        self.time_zero_abs = self.time_zero_abs - shift + self.time_zero_shift
        self.time_zero_shift = shift
        self.time_delays = self.time_delays_raw - self.time_zero_abs

    def cutout_region(self, x=None, y=None):
        if x is not None:
            x_ind = np.array([val >= x[0] and val <= x[1]
                             for i, val in enumerate(
                                     self.spec_axis[self._xmode])])
            self._region_cutout.append([
                'x', [np.where(x_ind), self.wavelengths[x_ind],
                               self.delA[:, x_ind]]])
            self.delA = self.delA[:, ~x_ind]
            self.wavelengths = self.wavelengths[~x_ind]
            self._init_spec_axis_dict(wavenumbers=self.set_wavenumbers())
        if y is not None:
            y_ind = np.array([val >= y[0] and val <= y[1]
                             for i, val in enumerate(self.time_delays)])
            self._region_cutout.append([
                'y', [np.where(y_ind), self.time_delays[y_ind],
                               self.delA[y_ind, :]]])
            self.delA = self.delA[~y_ind, :]
            self.time_delays = self.time_delays[~y_ind]

    def reset_cutout(self):
        for i, cut in enumerate(self._region_cutout[::-1]):
            inds = cut[1][0][0]
            delA_cut = cut[1][2]
            ax_cut = cut[1][1]
            if cut[0].lower() == 'x':
                delA_new = np.zeros(
                    (np.shape(self.delA)[0],
                     np.shape(self.delA)[1] + len(inds)))
                wl_new = np.zeros(len(self.wavelengths) + len(inds))
                k = 0
                for j, v in enumerate(wl_new):
                    if not np.any(inds == j):
                        wl_new[j] = self.wavelengths[k]
                        delA_new[:, j] = self.delA[:, k]
                        k += 1
                    else:
                        wl_new[j] = ax_cut[inds == j]
                        delA_new[:, j] = delA_cut[:, inds == j].flatten()
                self.delA = delA_new
                self.wavelengths = wl_new
            else:
                delA_new = np.zeros(
                    (np.shape(self.delA)[0] + len(inds),
                     np.shape(self.delA)[1]))
                td_new = np.zeros(len(self.time_delays) + len(inds))
                k = 0
                for j, v in enumerate(td_new):
                    if not np.any(inds == j):
                        td_new[j] = self.time_delays[k]
                        delA_new[j, :] = self.delA[k, :]
                        k += 1
                    else:
                        td_new[j] = ax_cut[inds == j]
                        delA_new[j, :] = delA_cut[inds == j, :].flatten()
                self.delA = delA_new
                self.time_delays = td_new
        self._region_cutout = []
        self._init_spec_axis_dict(wavenumbers=self.set_wavenumbers())

    def modify_time(self, unit=None, factor=None):
        if unit is not None:
            try:
                unit_factor = (self.time_unit_factors[self.time_unit][0]
                               / self.time_unit_factors[unit][0])
                self.time_delays = np.array(
                    [td * unit_factor for td in self.time_delays])
            except KeyError:
                print("Invalid Unit " + str(unit))
                unit_factor = 1
            except Exception:
                raise
            else:
                self.time_unit = unit
        else:
            unit_factor = 1
        if factor is not None:
            self.time_delays = np.array([td*factor for td in self.time_delays])
            self.time_delay_factor = factor*self.time_delay_factor
        else:
            factor = 1
        return factor * unit_factor

    def modify_delA(self, unit=None, factor=None):
        if unit is not None:
            try:
                unit_factor = (self.delA_unit_dict[self.delA_unit]
                               / self.delA_unit_dict[unit])
                self.delA = self.delA * unit_factor
            except KeyError:
                print("Invalid Unit " + str(unit))
                unit_factor = 1
                # raise
            except Exception:
                raise
            else:
                self.delA_unit = unit
                self.zlabel = '$\Delta$ Abs. (' + self.delA_unit + ')'
        else:
            unit_factor = 1
        if factor is not None:
            self.delA = self.delA * factor
            self.delA_modifier = self.delA_modifier * factor
        else:
            factor = 1
        return unit_factor * factor

    def set_wavenumbers(self, wl=None):
        if wl is None:
            wl = self.wavelengths
        self.wavenumbers = np.power(10, 7)/wl
        return self.wavenumbers

    def set_x_mode(self, mode, unit=True):
        try:
            keys = self._spec_conversion.keys()
        except Exception:
            self._init_spec_conversion_dict()
            keys = self._spec_conversion.keys()
        if mode in keys:
            std_units = {'wavelength': 'nm',
                         'wavenumber': 'cm^(-1)',
                         'energy': 'eV'}
            if unit:
                unit = std_units[self._xmode]
            self.modify_x(unit=unit)
            self._xmode = mode
            self.spec_unit = std_units[mode]
        elif re.search('time', mode, re.I):
            self._xmode = mode
        else:
            return
        self.set_xlabel()

    def get_wavenum_index(self, wavenum):
        ind = []
        for v in wavenum:
            try:
                ind.append(np.where(self.wavenumbers >= v)[0][-1])
            except Exception:
                try:
                    ind.append(np.where(self.wavenumbers <= v)[0][0])
                except Exception:
                    raise
        return ind

    def bin_pixels(self, n, matrix=None):
        if not matrix:
            matrix = self.non_filt_delA
        n_in = np.shape(matrix)[1]
        n_out = int(n_in/n)
        mod = np.mod(n_in, n)
        output = np.zeros((np.shape(matrix)[0], n_out))
        out_wl = np.zeros(n_out)
        for j in range(np.shape(matrix)[0]):
            k = 0
            for i in range(mod, n_in, n):
                output[j, k] = sum(matrix[j, i:i+n])/n
                out_wl[k] = sum(self._wavelengths[i:i+n])/n
                k += 1
        self.delA = output
        self.wavelengths = out_wl
        self.set_wavenumbers()
        self._init_spec_axis_dict()
        self.pixel_binned = True

    def reset_bin(self):
        self.delA = self.non_filt_delA
        self.wavelengths = self._wavelengths
        self.set_wavenumbers()
        self._init_spec_axis_dict()
        self.pixel_binned = False

    def time_to_str(self, val):
        return " ".join([str(np.round(val, self.time_delay_precision)),
                         self.time_unit])

    def calc_wavenumber(self, array):
        res = []
        for e in array:
            res.append(self._spec_conversion[self._xmode]['in'](e))
        if self._xmode == 'wavelength':
            res = res[::-1]
        return res

    def set_xlabel(self):
        if re.search('time', self._xmode, re.I):
            self.xlabel = "".join([self._xmode, " (", self.time_unit, ")"])
        else:
            if self._xmode == 'wavenumber':
                self.spec_unit_label = {'cm^(-1)': 'cm$^{-1}$',
                                        'm^(-1)': 'm$^{-1}$'}[self.spec_unit]
            else:
                self.spec_unit_label = self.spec_unit
            self.xlabel = self._xmode + ' (' + self.spec_unit_label + ')'
        return self.xlabel

    def set_ylabel(self):
        self.ylabel = self.time_unit.join(["time delay (", ")"])
        return self.ylabel

    def get_zlabel(self):
        return self.zlabel

    """ chirp correction """
    def find_time_zeros(self, irf_threshold=10, algorithm='derivative',
                        write_to_obj=True):
        def find_chirp_index(f, thresh=irf_threshold, algo=algorithm):
            if any(np.isnan(f)):
                return[np.nan]
            upper = np.where(f >= irf_threshold)[0]
            lower = np.where(f <= -irf_threshold)[0]
            if len(upper) == 0 and len(lower) > 0:
                ind = lower[0]
            elif len(upper) > 0 and len(lower) == 0:
                ind = upper[0]
            elif len(upper) == 0 and len(lower) == 0:
                ind = np.nan
            else:
                ind = np.min([upper[0], lower[0]])
            return[ind]
        
        time_zeros = np.zeros((len(self.spec_axis[self._xmode]), 2))
        time_zeros[:, 0] = self.spec_axis[self._xmode]
        if re.search('derivative', algorithm, re.I):
            irf_threshold = irf_threshold/10
        time_zeros[:, 1] = np.ones(len(self.wavelengths))*np.nan
        for i in range(self._lambda_lim_index[0],
                       self._lambda_lim_index[1] + 1):
            if re.search('derivative', algorithm, re.I):
                a = np.gradient(self.delA[:, i])
                ind = find_chirp_index(a)
            elif re.search('threshold real', algorithm, re.I):
                a = self.delA[:, i]
                ind = find_chirp_index(a)
            elif re.search('threshold abs', algorithm, re.I):
                a = np.abs(self.delA[:, i])
                n = np.where(a >= irf_threshold)
                if len(n[0]) > 0:
                    ind = np.max(n[0])
                else:
                    ind = np.nan

            if np.isnan(ind):
                time_zeros[i, 1] = np.nan
            else:
                time_zeros[i, 1] = self.time_delays_raw[ind]
        if write_to_obj:
            self.time_zeros = time_zeros
        else:
            return time_zeros

    def fit_time_zeros(self, polyorder=5,
                       outlier_thresh_rel=2,
                       ind='wlwindow',
                       time_zeros=None, write_to_obj=True,
                       forced_outliers=None,
                       non_outliers=None):
        if time_zeros is None:
            time_zeros = self.time_zeros
        if forced_outliers is None:
            forced_outliers = []
        if non_outliers is None:
            non_outliers = []
        if ind == 'wlwindow':
            lower = np.where(time_zeros[:, 0] >= self.lambda_lim[0])[0][0]
            upper = np.where(time_zeros[:, 1] <= self.lambda_lim[1])[0][-1]
            range_ind = range(lower, upper + 1)
        else:
            range_ind = range(ind[0], ind[1] + 1)
        lam = time_zeros[range_ind, 0]
        chirp = time_zeros[range_ind, 1]
        sort_ind = np.argsort(lam)
        lam = lam[sort_ind]
        chirp = chirp[sort_ind]
        lam = lam[~np.isnan(chirp)]
        chirp = chirp[~np.isnan(chirp)]
        fit_obj = LineFit(x=lam, y=chirp,
                          model='poly' + str(polyorder),
                          outlier_threshold=outlier_thresh_rel)
        fit_obj.non_outliers = non_outliers
        fit_obj.forced_outliers = forced_outliers
        fit_obj.run_fit(reject_outliers=True)
        fit_curve = fit_obj.fit_function(
            fit_obj.result.params)(self.wavelengths)
        if write_to_obj:
            self.time_zero_fit = fit_curve
            self.time_zero_abs = np.min(
                self.time_zero_fit[self.get_xlim_slice()])
            return np.array(fit_obj.outliers)
        else:
            return np.array(fit_obj.outliers), fit_curve

    def spectral_shift(self, y, data_mat=None, td=None, spec=None):
        if data_mat is None:
            data_mat = self.delA
        if td is None:
            td = self.time_delays
        if spec is None:
            spec = self.spec_axis[self._xmode]
        f_step = 1/(np.max(spec)-np.min(spec))
        f = f_step*np.arange(1, len(spec) + 1)
        f = f - np.mean(f)
        y = y - np.min(y)
        y1 = np.fft.fftshift(ifft(np.fft.ifftshift(
            np.transpose(data_mat), axes=-2), axis=-2), axes=-2)
        y1 = y1.astype(complex)
        y2 = np.zeros(np.shape(y1), dtype=complex)
        try:
            for i in range(len(td)):
                arg = -1j*(y[i]*f)
                y2[:, i] = np.multiply(y1[:, i], np.exp(arg))
        except IndexError:
            if np.shape(y) != np.shape(td):
                return 0
            else:
                raise
        except Exception:
            raise
        return np.transpose(np.real(np.fft.fftshift(fft(
            np.fft.ifftshift(y2, axes=-2), axis=-2), axes=-2)))

    def _correct_chirp(self, dat=None, correct_t0=True, queue=None,
                       truncate_wavelength=False):
        range_ind = self.get_xlim_slice()
        self.time_zero_fit = self.time_zero_fit - \
            np.min(self.time_zero_fit[range_ind])
        if truncate_wavelength:
            n = len(self.wavelengths[range_ind])
        else:
            range_ind = slice(len(self.wavelengths))
            n = len(self.wavelengths)

        if dat is None:
            if self.chirp_corrected:
                dat = self.non_filt_delA[:, range_ind]
            else:
                dat = self.raw_delA[:, range_ind]
        else:
            dat = dat[:, range_ind]

        wavenum_step = 33357/(np.max(self.time_delays)
                              - np.min(self.time_delays))
        wavenum = wavenum_step*np.arange(1, len(self.time_delays) + 1)
        wavenum = wavenum - np.mean(wavenum)
        if queue is not None:
            queue.put({'label': "Fourier Transform"})
        f1 = np.fft.fftshift(
            ifft(np.fft.ifftshift(dat, axes=-2), axis=-2), axes=-2)
        if queue is not None:
            queue.put({'label': "Correcting Chirp"})
        f1 = f1.astype(complex)
        f2 = np.zeros(np.shape(f1), dtype=complex)

        for i in range(n):
            arg = -1j*(self.time_zero_fit[i]*wavenum/(5308))
            f2[:, i] = np.multiply(f1[:, i], np.exp(arg))
            if queue:
                queue.put({'i': i, 'n': n})

        self.delA = np.real(np.fft.fftshift(fft(
            np.fft.ifftshift(f2, axes=-2), axis=-2), axes=-2))
        if queue is not None:
            queue.put({'i': i + 1, 'n': n})
        if correct_t0:
            self.time_delays = self.time_delays_raw - self.time_zero_abs
        return range_ind

    def correct_chirp(self, correct_t0=True, queue=None,
                      max_time_step=None,
                      truncate_wavelength=False):
        if not max_time_step:
            max_time_step = np.inf
        n = self.time_delay_precision
        if len(self.timesteps) == 0:
            self._get_time_steps()
        min_step = np.min(self.timesteps)
        td = self.time_delays_raw
        # check time steps and assign indices for regions to be corrected
        cc_inds = []
        non_cc_inds = []
        j = 0
        while j < len(td) - 1:
            if (np.abs(np.round(td[j + 1], n) - np.round(td[j], n))
                    <= max_time_step):
                try:
                    while (np.abs(np.round(td[j + 1], n) - np.round(td[j], n))
                           <= max_time_step):
                        cc_inds.append(j)
                        j += 1
                except Exception:
                    break
                # overlapping points
                cc_inds.append(j)
                non_cc_inds.append(j - 1)
                non_cc_inds.append(j)
                j += 1
            else:
                non_cc_inds.append(j)
                j += 1
        # last point
        if j - 1 in non_cc_inds:
            non_cc_inds.append(j)
        else:
            cc_inds.append(j)

        # write time delays for chirp correction
        del self.time_delays
        self.time_delays = []
        tstep_inds = []
        for i, ind in enumerate(cc_inds[:-1]):
            t = np.round(td[ind], n)
            while t < np.round(td[ind+1], n):
                self.time_delays.append(t)
                t = np.round(t + min_step, n)
            tstep_inds.append(len(self.time_delays) - 1)
            if queue:
                queue.put({'i': 0.01, 'n': 1})
        self.time_delays.append(t)
        tstep_inds.append(len(self.time_delays) - 1)

        # writing matrix for correction (delA_expanded) after saving copy
        if queue:
            queue.put({'label': "Writing Matrix for Correction"})
        # using either previously corrected matrix (refining) or raw data
        if self.chirp_corrected:
            delA_non_cc = np.array(self.non_filt_delA)
        else:
            delA_non_cc = np.array(self.raw_delA)
        delA_expanded = np.zeros(
            (len(self.time_delays), np.shape(delA_non_cc)[1]))
        k = 0
        for i, ind in enumerate(cc_inds[:-1]):
            for j in range(tstep_inds[i + 1] - tstep_inds[i]):
                factor = j/np.abs(tstep_inds[i+1]-tstep_inds[i])
                delA_expanded[k, :] = (delA_non_cc[cc_inds[i], :]
                                       + factor*(
                                           delA_non_cc[cc_inds[i]+1, :]
                                           - delA_non_cc[cc_inds[i], :]))
                k += 1
            if queue:
                queue.put({'i': i + 1, 'n': len(cc_inds)})
        delA_expanded[k, :] = delA_non_cc[cc_inds[i + 1], :]
        # run chirp correction for designated data range
        range_ind = self._correct_chirp(
            dat=delA_expanded,
            correct_t0=False, queue=queue,
            truncate_wavelength=truncate_wavelength)
        # recombine corrected and non-corrected regions and re-reduce data
        if queue:
            length = len(non_cc_inds) + len(tstep_inds) - 1
            queue.put({'label': "Writing new data Matrix",
                       'i': 0, 'n': length})
        delA_cc = self.delA
        self.time_delays = np.array(td)
        self.delA = np.zeros(np.shape(delA_non_cc))
        for j, ind in enumerate(tstep_inds):
            self.delA[cc_inds[j], range_ind] = delA_cc[ind, :]
            if queue:
                queue.put({'i': j, 'n': length})
        for i, ind in enumerate(non_cc_inds):
            try:
                self.delA[ind, :] = delA_non_cc[ind, :]
            except Exception:
                continue
            if queue:
                queue.put({'i': i+j, 'n': length})

        del delA_cc
        if correct_t0:
            self.time_delays = self.time_delays - self.time_zero_abs
        self.chirp_corrected = True
        return True

    def _interpolate_time(self, dat, x, n, min_step):
        if np.shape(dat)[0] != len(x):
            dat = np.transpose(dat)
        time_delays = []
        tstep_inds = []
        for i, td in enumerate(x[:-1]):
            t = np.round(td, n)
            while t < np.round(x[i+1], n):
                time_delays.append(t)
                t = np.round(t + min_step, n)
            tstep_inds.append(len(time_delays) - 1)
        time_delays.append(t)
        tstep_inds.append(len(time_delays) - 1)

        new_dat = np.zeros((len(time_delays), np.shape(dat)[1]))
        k = 0
        for i in range(len(x) - 1):
            for j in range(tstep_inds[i + 1] - tstep_inds[i]):
                factor = j/np.abs(tstep_inds[i+1]-tstep_inds[i])
                new_dat[k, :] = 0
                new_dat[k, :] = dat[i, :] + factor*(dat[i+1, :] -
                                                    dat[i, :])
                k += 1
        new_dat[k, :] = dat[i + 1, :]
        dat = np.transpose(new_dat)
        x = time_delays
        return dat, x

    def _fft_wavelet_wrap(self, case='fft', dat=None, index=None, x=None,
                          xrange=None, xunit=None,
                          round_prec=None, data_reduct_crit=0.05, **kwargs):
        if dat is None:
            dat = self.delA
        if xunit is None:
            xunit = self.time_unit
        if round_prec is None:
            try:
                n = self.time_delay_precision
            except Exception:
                n = 10
        else:
            n = round_prec
        if x is None:
            x = self.time_delays
        if xrange:
            x = x[xrange[0]:xrange[1]]
        self._get_time_steps(td=x, td_round_prec=n)
        min_step = np.round(np.min(self.timesteps), n)
        if len(self.timesteps) > 1:
            dat, x = self._interpolate_time(dat, x, n, min_step)
            xrange = None
        error = None
        if xunit in self.time_unit_factors.keys():
            min_step = self.time_unit_factors[xunit][0]*min_step
        else:
            error = ("Unable to determine time delay unit. Assuming ps.")
        if case.lower() == 'fft':
            y, f = self._fft_map(x, dat, min_step, data_reduct_crit, **kwargs)
            return y, f, error
        elif case.lower() == 'wavelet':
            mat, f = self._wavelet_tr(x, dat, min_step, data_reduct_crit,
                                      index=index, xrange=xrange, **kwargs)
            return mat, f, error

    def _wavelet_tr(self, x, dat, min_step, data_reduct_crit,
                    index=None, wavelet=morlet2, xrange=None,
                    num_scales=1000, **kwargs):
        if index is not None:
            dat = dat[index, :]
        if xrange is not None:
            dat = dat[xrange[0]:xrange[1]]
        wds = np.arange(1, num_scales)
        cwtmatr = cwt(dat, ricker, wds)
        freq = wds
        return np.abs(cwtmatr), freq

    def _fft_map(self, x, dat, min_step, data_reduct_crit, **kwargs):
        # frequency vector in wavenumbers
        f = fftfreq(len(x), d=min_step)[:len(x)//2] * 33.356
        # Calculate fft along time axis
        y = fft(np.fft.fftshift(dat, axes=-1), axis=-1)
        y = y.astype(complex)
        y = np.abs(2 * y/len(x))
        y = y[:, :len(x)//2]
        # reduce data if necessary
        if len(self.timesteps) > 1:
            f_step_fund = np.abs(f[1] - f[0])
            f_step = f_step_fund
            y_new = [y[:, 0]]
            f_new = [f[0]]
            ycurr = np.zeros(np.shape(y[:, 0]))
            av_count = 0
            for i in range(len(f)):
                while f_step < f[i] * data_reduct_crit:
                    f_step += f_step_fund
                ycurr += y[:, i]
                av_count += 1
                if f[i] >= f_new[-1] + f_step:
                    f_new.append(f[i])
                    y_new.append(ycurr / av_count)
                    ycurr = np.zeros(np.shape(y[:, 0]))
                    av_count = 0
                else:
                    continue
            y = np.transpose(np.array(y_new))
            f = np.array(f_new)
        try:
            self._get_time_steps(td=None)
        except Exception:
            pass
        return y, f

    def fft_map(self, **kwargs):
        return self._fft_wavelet_wrap(case='fft', **kwargs)

    def wavelet_analysis(self, **kwargs):
        return self._fft_wavelet_wrap(case='wavelet', **kwargs)

    """ Global Analysis """
    def svd_reduction(self, data_mat, svd_comp, return_all=False, cutoff=True):
        u, s, v = np.linalg.svd(data_mat, full_matrices=False)
        if cutoff:
            u_red = u[:, :svd_comp]
            s_red = s[:svd_comp]
            v_red = v[:svd_comp, :]
        else:
            u_red = np.transpose([u[:, i] for i in svd_comp])
            s_red = [s[i] for i in svd_comp]
            v_red = [v[i, :] for i in svd_comp]
        if return_all:
            return (np.dot(u_red, np.dot(np.diag(s_red), v_red)),
                    u_red, s_red, v_red)
        else:
            return np.dot(u_red, np.dot(np.diag(s_red), v_red))

    def init_global_analysis(self, model='parallel', td_window=None,
                             spec_window=None, inf_comp=False,
                             para_constraint=100, amp_constraint=2,
                             svd_comp=None, sine_modul=False,
                             **global_fit_kwargs):
        # time and wavelength window
        if spec_window is None:
            spec_window = self._lambda_lim_index
        if td_window is None:
            ga_fit_data = self.delA[:, spec_window[0]:spec_window[1] + 1]
            ga_ydata = self.time_delays
        else:
            ga_fit_data = self.delA[slice(td_window[0], td_window[1] + 1),
                                    spec_window[0]:spec_window[1] + 1]
            ga_ydata = self.time_delays[slice(td_window[0], td_window[1] + 1)]
        # SVD data reduction
        if svd_comp:
            ga_fit_data = self.svd_reduction(ga_fit_data, svd_comp)

        constraints = {'min_para_ratio': para_constraint/100,
                       'max_amplitude_ratio': amp_constraint,
                       'total_population': None if model == 'parallel' else 1}

        return GlobalFit(ga_ydata, ga_fit_data, model=model,
                         constraints=constraints, constant_component=inf_comp,
                         **global_fit_kwargs)

    def run_global_analysis(self, fit_obj=None, queue=None,
                            **global_fit_kwargs):
        def write_empty_attributes():
            self.ga_das = TATrace(trace_type='DAS')
            self.ga_das.tr["NaN"] = {'y': None}
            return fit_obj, self.ga_das
        if fit_obj is None:
            fit_obj = self.init_global_analysis(**global_fit_kwargs)
            run_kwargs = {}
        else:
            run_kwargs = global_fit_kwargs

        if fit_obj.error is not None:
            return write_empty_attributes()
        try:
            fit_obj.run_fit(queue=queue, **run_kwargs)
        except Exception:
            return write_empty_attributes()

        if fit_obj.error is not None:
            return write_empty_attributes()
        self.ga_das = TATrace(trace_type='DAS')
        for i in range(fit_obj.number_of_species):
            key = self.time_to_str(
                np.round(fit_obj.result.params['tau_' + str(i + 1)].value, 4))
            self.ga_das.tr[key] = {'y': fit_obj.x_comps[:, i]}
            self.ga_das.active_traces.append(key)
        if queue is not None:
            queue.task_done()
        return fit_obj, self.ga_das

    """ integration and centroid """

    def integrate_spectrum(self, xrange=None, t_range='all', label=None,
                           add_to_traces=False):
        if xrange is None:
            xslice = self.get_xlim_slice()
        else:
            xslice = slice(*xrange)
        if t_range.lower() == 'all':
            t_range = self._time_range_indices
        self.integral.xdata = self.time_delays[slice(*t_range)]
        self.integral.xlabel = self._xmode + ' (' + self.spec_unit_label + ')'
        t_range[1] = t_range[0] + len(self.integral.xdata)
        self.integral.type = 'Spectral'
        self.integral.set_x_mode(self._xmode)
        y = []
        for i in range(t_range[0], t_range[1]):
            y.append(np.sum(self.delA[i, xslice]))
        y = y/np.max(np.abs(y))
        if label not in self.integral.tr.keys():
            self.integral.tr[label] = {}
        self.integral.tr[label]['y'] = y
        if add_to_traces:
            self.integral.active_traces.append(label)
        else:
            self.integral.active_traces = [label]

    def integrate_time(self, time_range=None, spec_range='all', label=None,
                       add_to_traces=False):
        if spec_range == 'all':
            spec_range = self._lambda_lim_index
        if time_range is None:
            time_range = self._time_range_indices
        self.integral.xdata = self.spec_axis[self._xmode][slice(*spec_range)]
        self.integral.xlabel = 'time delay (' + self.time_unit + ')'
        spec_range[1] = spec_range[0] + len(self.integral.xdata)
        self.integral.type = 'Kinetic'
        self.integral.set_x_mode('time delay')
        y = []
        n = time_range[1] - time_range[0]
        for i in range(spec_range[0], spec_range[1]):
            y.append(np.sum(self.delA[slice(*time_range), i])/n)
        if label not in self.centroid.tr.keys():
            self.integral.tr[label] = {}
        self.integral.tr[label]['y'] = y
        if add_to_traces:
            self.integral.active_traces.append(label)
        else:
            self.integral.active_traces = [label]

    def calculate_centroid(self, xrange=None, t_range='all', label=None,
                           add_to_traces=False):
        def centroid(tr):
            x = self.spec_axis['wavenumber'][xrange[0]:xrange[1]]
            c = np.sum(tr * x)/np.sum(tr)
            if self._xmode == 'wavelength':
                c = 1e7 / c
            elif self._xmode == 'energy':
                c = 0.000124 * c
            return c

        if xrange is None:
            xrange = self._lambda_lim_index
        if t_range == 'all':
            t_range = self._time_range_indices
        self.centroid.xdata = self.time_delays[t_range[0]:t_range[1]]
        y = []
        for i in range(t_range[0], t_range[1]):
            c = centroid(self.delA[i, xrange[0]:xrange[1]])
            if c < np.min(self.spec_axis[self._xmode][xrange]):
                c = np.min(self.spec_axis[self._xmode][xrange])
            elif c > np.max(self.spec_axis[self._xmode][xrange]):
                c = np.max(self.spec_axis[self._xmode][xrange])
            y.append(c)
        if label not in self.centroid.tr.keys():
            self.centroid.tr[label] = {}
        self.centroid.tr[label]['y'] = y
        if add_to_traces:
            self.centroid.active_traces.append(label)
        else:
            self.centroid.active_traces = [label]

    def find_spec_maximum(self, xrange, t_range='all', x=None, y=None, z=None,
                          mode='max'):
        def find_max(row):
            return np.argmax(savgol_filter(row[slice(*xrange)], 5, 3))
        if x is None:
            x = self.spec_axis[self._xmode]
        if y is None:
            y = self.time_delays
        if z is None:
            z = self.delA
        if t_range == 'all':
            t_range = [0, len(y)]
        if mode == 'min':
            z = -z
        out = []
        for i in range(*t_range):
            out.append(x[slice(*xrange)][find_max(z[i, :])])
        return y[slice(*t_range)], out

    """ Dynamic line shape analysis """

    def init_line_shape_fit(self, fit_obj=None, num_comp=2,
                            line_shape='gaussian', spec_window=None,
                            constraints=None, constraint_inits=None,
                            offset=False, fixed_para=None, guesses=None,
                            **fit_obj_kwargs):
        if constraints is None:
            constraints = {'width': "",
                           'spacing': ""}
        if re.search('gauss', line_shape, re.I):
            model_name = 'gauss'
            wd_para_name = 'sigma'
        elif re.search('lorentz', line_shape, re.I):
            model_name = 'lorentz'
            wd_para_name = 'gamma'
        elif re.search('voigt', line_shape, re.I):
            model_name = 'voigt'
            wd_para_name = 'sigma'
        else:
            print('invalid line shape.')
            return
        model = model_name + str(num_comp)
        if offset:
            model += "_const"

        if fit_obj:
            fit_obj.set_attributes(model=model, **fit_obj_kwargs)
        else:
            fit_obj = LineFit(model=model, **fit_obj_kwargs)
        para_correl = {}
        if constraints['width'].lower() == 'linear':
            try:
                val = constraint_inits['width']
            except Exception:
                val = 1000
            fit_obj.params.add('wdfactor', value=val, min=0)
            for i in range(num_comp):
                para_correl["_".join(
                    [model_name, wd_para_name, str(i + 1)])] = (
                        str(i + 1).join(["", " * wdfactor"]))
        if constraints['spacing'].lower() == 'equal':
            try:
                val = constraint_inits['spacing']
            except Exception:
                val = 500
            fit_obj.params.add('spacing', value=val, min=0)
            for i in range(1, num_comp):
                para_correl["_".join([model_name, 'x0', str(i + 1)])] = (
                    str(i).join([model_name + "_x0_1 + ", " * spacing"]))
        if guesses is not None:
            fit_obj.set_guesses(guesses)
        fit_obj.set_para_correl(para_correl=para_correl)
        fit_obj.set_fixed_para(fixed_para=fixed_para)
        return fit_obj

    def single_line_shape(self, time_index=0, ydata=None, zdata=None,
                          xdata=None, xinds=None, fit_obj=None, guesses=None,
                          init_to_result=False, init_variance=0.005,
                          **fit_kwargs):
        if xdata is None:
            xdata = self.wavenumbers
            if xinds is not None:
                xdata = xdata[xinds]
        if xinds is None:
            xinds = slice(0, len(xdata))
        if ydata is None:
            if zdata is None:
                zdata = self.delA
            ydata = zdata[time_index, xinds]
        if not fit_obj:
            fit_obj = self.init_line_shape_fit(x=xdata, y=ydata, **fit_kwargs)
        else:
            fit_obj.set_attributes(
                x=xdata, y=ydata, update=False, **fit_kwargs)
            if init_to_result:
                fit_obj.set_init_to_result(variance=init_variance)
        success, report = fit_obj.run_fit(guesses=guesses)
        return fit_obj, success, report

    def dynamic_line_shape(self, fit_obj=None, xinds=None, yrange=None,
                           reverse=True, guesses=None, queue=None,
                           variance=0.005, **fit_obj_kwargs):
        if xinds is None:
            xinds = slice(0, len(self.wavenumbers))
            xdata = self.wavenumbers
        else:
            xdata = self.wavenumbers[xinds]
        if yrange is None:
            yrange = [self.time_delays[0], self.time_delays[-1]]
        t_start = np.where(self.time_delays >= yrange[0])[0][0]
        t_end = np.where(self.time_delays <= yrange[1])[0][-1]
        fit_matrix = np.zeros((np.abs(t_end - t_start) + 1, len(xdata)))
        residual = np.zeros((np.abs(t_end - t_start) + 1, len(xdata)))
        fit_paras = {}
        fit_para_err = []
        for p in fit_obj.params:
            fit_paras[p] = {
                'value': np.zeros((np.abs(t_end - t_start) + 1)),
                'stderror': np.zeros((np.abs(t_end - t_start) + 1))}
        time_ind = t_end if reverse else t_start
        it = range(t_end - 1, t_start - 1, -
                   1) if reverse else range(t_start + 1, t_end + 1)
        time_points = [self.time_delays[time_ind]]
        try:
            fit_obj, success, report = self.single_line_shape(
                time_index=time_ind, xdata=xdata, xinds=xinds, fit_obj=fit_obj,
                guesses=guesses, **fit_obj_kwargs)
            fit_matrix[time_ind - t_start, :] = fit_obj.curve
            for p in fit_obj.result.params:
                fit_paras[p]['value'][time_ind -
                                      t_start] = fit_obj.result.params[p].value
                try:
                    fit_paras[p]['stderror'][time_ind -
                                             t_start] = fit_obj.stderrors[p]
                except Exception:
                    fit_paras[p]['stderror'][time_ind - t_start] = np.nan
        except Exception:
            return fit_paras, fit_matrix, fit_para_err, residual, time_points
        time_points.extend([self.time_delays[i] for i in it])
        n = len(time_points)
        i = 1
        for time_ind in it:
            fit_obj, success, report = self.single_line_shape(
                time_index=time_ind, xdata=xdata, xinds=xinds,
                fit_obj=fit_obj, init_to_result=True,
                init_variance=variance, **fit_obj_kwargs)
            i += 1
            try:
                paras = {}
                for p in fit_obj.result.params:
                    paras[p] = fit_obj.result.params[p].value
                for p in fit_obj.result.params:
                    fit_paras[p]['value'][time_ind - t_start] = paras[p]
            except Exception:
                for p in fit_paras.keys():
                    fit_paras[p]['value'][time_ind - t_start] = np.nan
                    fit_paras[p]['stderror'][time_ind - t_start] = np.nan
            else:
                fit_matrix[time_ind - t_start, :] = fit_obj.curve
                residual[time_ind - t_start, :] = fit_obj.residual
                for p in fit_obj.result.params:
                    try:
                        fit_paras[p]['stderror'][
                            time_ind - t_start] = fit_obj.stderrors[p]
                    except Exception:
                        fit_paras[p]['stderror'][time_ind - t_start] = np.nan
                if queue is not None:
                    queue.put({'fit': fit_obj.curve,
                               'dat': fit_obj.y,
                               'point': time_ind,
                               'prog': i/n})
        return fit_paras, fit_matrix, fit_para_err, residual, time_points


# %%
class TATraceCollection():
    """ a collection of trace objects of class TATrace.
    class TATrace does not allow for individual x values,
    so use this class if that is required. """

    def __init__(self, init_trace=False, key=0):
        self.traces = {}
        self.active_members = []
        self.xlabel = ""
        if init_trace:
            self.traces[key] = TATrace()
            self.active_members = [key]

    def add_trace(self, trace, key, overwrite=False):
        if (not overwrite) and (key in self.traces.keys()):
            j = 2
            new_key = "_".join([key, str(j)])
            while new_key in self.traces.keys():
                j += 1
                new_key = "_".join([new_key.strip("_" + str(j - 1)), str(j)])
            key = new_key
        self.traces[key] = trace
        self.active_members.append(key)

    def remove_trace(self, key):
        del self.traces[key]
        if key in self.active_members:
            self.active_members.remove(key)

    def remove_active_traces(self):
        for member in self.active_members:
            for tr in self.traces[member].active_traces:
                self.traces[member].delete_trace(tr)
            if not self.traces[member].tr:
                self.remove_trace(member)

    def remove_all_traces(self):
        keys = list(self.traces.keys())
        for key in keys:
            del self.traces[key]
            try:
                self.active_members.remove(key)
            except Exception:
                pass

    def init_exp_fit(self, *args, **kwargs):
        try:
            keys = list(self.traces.keys())
        except Exception:
            return
        else:
            return self.traces[keys[0]].init_exp_fit(*args, **kwargs)

    def init_line_shape_fit(self, *args, **kwargs):
        try:
            keys = list(self.traces.keys())
        except Exception:
            return
        else:
            return self.traces[keys[0]].init_line_shape_fit(*args, **kwargs)

    def get_number_of_active_traces(self):
        n = 0
        for member in self.active_members:
            n += len(self.traces[member].active_traces)
        return n

    def set_x_mode(self, mode='wavelength'):
        unit = 'nm'
        for member in self.active_members:
            trace = self.traces[member]
            try:
                trace.set_x_mode(mode)
            except Exception:
                traceback.print_exc(limit=-5)
            try:
                unit = trace.spec_unit_label
            except Exception:
                pass
        return unit

    def plot(self, *args, ax=None, include_fit_comps=True, color_order=None,
             color_cycle=None, interpolate_colors=False, reverse_z=False,
             **kwargs):
        if ax is None:
            return
        else:
            ax.cla()
        if color_order:
            start_inds = np.array(color_order['ind'])
            cord = {'ind': start_inds,
                    'reverse': color_order['reverse']}
        kwargs['show_legend'] = False
        color_offset = 0
        numlines_total = len([key for member in self.active_members
                              for key in self.traces[member].active_traces])
        zord, cycle = set_line_cycle_properties(
            numlines_total, reverse_z=reverse_z, color_order=color_order,
            cycle=color_cycle, interpolate=interpolate_colors)
        for key in self.active_members:
            num_lines = len(self.traces[key].active_traces)
            if include_fit_comps:
                for k in self.traces[key].active_traces:
                    try:
                        for comp in self.traces[key].tr[k]['fit_comps'].keys():
                            num_lines += 1
                    except KeyError:
                        pass
                    except Exception:
                        raise
            if not color_order:
                cord = {'ind': np.array([i for i in range(num_lines)])
                        + color_offset,
                        'reverse': False}
            else:
                cord['ind'] = start_inds + color_offset
            self.traces[key].plot(*args, ax=ax, cla=False,
                                  include_fit_comps=include_fit_comps,
                                  color_order=cord,
                                  color_cycle=cycle, interpolate_colors=False,
                                  **kwargs)
            color_offset += num_lines

    def plot_single(self, i, *args, **kwargs):
        def get_member():
            j = 0
            for member in self.active_members:
                for key in self.traces[member].active_traces:
                    if j == i:
                        return j, key, member
                    j += 1
            return -1, None, None
        ind, current_trace, current_member = get_member()
        if ind != i:
            return
        try:
            self.traces[current_member].plot(
                *args, active_traces=[current_trace], **kwargs)
        except KeyError:
            pass
        except Exception:
            raise

    def get_legend(self):
        legend = []
        for member in self.active_members:
            legend.extend(self.traces[member].active_traces)
        return legend

    def plot_residual(self, *args, fig=None, ax=None, range_ind=None,
                      **plot_kwargs):
        return self.plot_for_active_members(plot_func='resid', *args, fig=fig,
                                            ax=ax, range_ind=range_ind,
                                            **plot_kwargs)

    def plot_resid_fft(self, *args, **kwargs):
        return self.plot_for_active_members(*args, plot_func='resid_fft',
                                            **kwargs)

    def plot_for_active_members(self, *args, plot_func=None, fig=None, ax=None,
                                range_ind=None, **plot_kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        func_dict = {'plot': (lambda obj, *args, **kwargs:
                              obj.plot(*args, **kwargs)),
                     'resid': (lambda obj, *args, **kwargs:
                               obj.plot_residual(*args, **kwargs)),
                     'resid_fft': (lambda obj, *args, **kwargs:
                                   obj.plot_residual_fft(*args, **kwargs))}
        try:
            plot_func = func_dict[plot_func]
        except Exception:
            plot_func = func_dict['plot']
        lines = {}
        ax.cla()
        for member in self.active_members:
            r_ind = None if range_ind is None else range_ind[member]
            lines[member] = plot_func(self.traces[member], *args, fig=fig,
                                      ax=ax, range_ind=r_ind, clear_ax=False,
                                      set_ax_labels=False, **plot_kwargs)
        return lines

    def residual_fft(self):
        for member in self.active_members:
            self.traces[member].residual_fft()

    def save(self, file, *args, save_fit=False, **kwargs):
        return self.save_traces(file, *args, save_fit=save_fit, **kwargs)

    def save_traces(self, file, trace_keys='all', spec_quantity='wavelength',
                    spec_unit='nm', trace_type='Kinetic', ylabel=None,
                    custom_para=None, save_fit=True, range_ind=None):
        if type(file) is str:
            fname = file
        else:
            fname = file.name
        trace_type = str(trace_type).title()
        if trace_keys == 'all':
            trace_keys = self.active_members
        integral_prefix = ["Spectral Trace of Temporal ",
                           "Kinetic Trace of Spectral "]
        save_dict = {'Kinetic': ['Kinetic Trace(s)', spec_quantity],
                     'CPM': ['CPM Trace(s)', spec_quantity],
                     'Spectral': ['Spectral Trace(s)', 'time_delays'],
                     'Integral': [trace_keys[0], spec_quantity],
                     'Centroid': [trace_keys[0], spec_quantity],
                     'Lineshape': ['Lineshape Trace(s)', 'time_delays'],
                     'DAS': ['Decay-associated spectra (by decay times)',
                             'tau'],
                     'SAS': ['Species-associated spectra (by life times)',
                             'tau'],
                     'None': ['Traces(s)', 'Name']}
        errors = {}
        if fname.endswith('txt'):
            matrix = []
            column_headers = []
            for key in trace_keys:
                tr_info = [s for s in save_dict[trace_type]]
                if re.search('integ', trace_type, re.I):
                    if re.search('time', self.traces[key].xlabel, re.I):
                        tr_info[0] = integral_prefix[1] + tr_info[0]
                    else:
                        tr_info[0] = integral_prefix[0] + tr_info[0]
                try:
                    mat, head = self.traces[key].save_traces_txt(
                        fname, tr_info, self.traces[key].active_traces,
                        range_ind=range_ind, save_fit=save_fit, ylabel=ylabel,
                        writefile=False)
                except KeyError as e:
                    if save_fit:
                        errors[key] = 'no_fit'
                    else:
                        errors[key] = e
                except Exception as e:
                    errors[key] = e
                else:
                    col_head = [h.strip()
                                for h in head.split('\n')[-1].split(",")]
                    column_headers.extend(
                        [": ".join([key, k])
                         for k in col_head])
                    matrix.append(mat)
            if len(matrix) == 0:
                return errors
            header = head.split('\n')
            header[0] = trace_type + " Trace Collection"
            header.pop(3)
            header[2] = "Collection start columns: "
            header[-1] = ", ".join(column_headers)
            # find maximum dimension of all traces
            dim = [np.shape(matrix[0])[0], 0]
            for i in range(np.shape(matrix)[0]):
                d = np.shape(matrix[i])
                if d[0] > dim[0]:
                    dim[0] = d[0]
                dim[1] += d[1]
            # write matrix for saving
            mat = np.empty((dim))
            mat[:] = np.nan
            j = 0
            for i in range(np.shape(matrix)[0]):
                d = np.shape(matrix[i])
                mat[:d[0], j:j + d[1]] = matrix[i]
                header[2] += str(j) + ", "
                j += d[1]
            header[2] = header[2][:-1]
            header = '\n'.join(header)
            with open(fname, 'w') as f:
                np.savetxt(f, mat, delimiter=',',
                           header=header)
                f.close()

        elif fname.split('.')[-1] == 'mat':
            vars_to_save = {}
            if len(trace_keys) == 1:
                key = trace_keys[0]
                trace_keys = ['data']
                self.traces[trace_keys[0]] = self.traces[key]
            for key in trace_keys:
                self.traces[key].xlabel = self.xlabel
                try:
                    vars_to_save[key] = self.traces[key].save_traces_mat(
                        fname,
                        save_fit=save_fit,
                        trace_keys=self.traces[key].active_traces,
                        trace_type=trace_type,
                        trace_label_name=save_dict[trace_type][1],
                        custom_para=custom_para,
                        trace_obj=self.traces[key],
                        writefile=False)
                except Exception as e:
                    errors[key] = e
            sio.savemat(fname, vars_to_save)
        else:
            errors['0'] = 'Invalid file type'
        return errors

    def load_traces(self, file, member_offset=0):
        def read_txt_matrix(fname):
            with open(fname) as f:
                header = []
                line = f.readline()
                while line[0] == "#":
                    header.append(line)
                    line = f.readline()
                mat = [line]
                mat.extend(f.readlines())
                f.close()
            try:
                mat = np.array([x.strip().split(",")
                               for x in mat]).astype(np.float64)
            except Exception:
                try:
                    mat = np.array([x.strip().split("\t") for x in mat])
                    mat = mat.astype(np.float64)
                except Exception:
                    raise
            return mat, header
        if type(file) is str:
            fname = file
        else:
            fname = file.name
        if fname.endswith("txt"):
            start_columns = [0]
            type_dct = {'kinetic': 'Kinetic',
                        'lineshape': 'Spectral',
                        'spectra': 'Spectral'}
            mat, header = read_txt_matrix(fname)
            tr_type = 'kinetic'
            for tracetype in type_dct.keys():
                if re.search(re.compile(tracetype, re.I), header[0]):
                    tr_type = type_dct[tracetype]
                    break
            xlabel = None
            ylabel = None
            for line in header:
                if re.search(re.compile('X-Axis', re.I), line):
                    xlabel = line[-re.search(":", line[::-1]).end()+1:]
                if re.search(re.compile('Y-Axis', re.I), line):
                    ylabel = line[-re.search(":", line[::-1]).end()+1:]

            if re.search('collect', header[0], re.I):
                for line in header:
                    if re.search(re.compile('start.+columns', re.I), line):
                        start_columns_str = re.findall(
                            '(?<=:).+', line)[0].strip().split(',')
                for d in start_columns_str:
                    try:
                        start_columns.append(int(d))
                    except Exception:
                        pass
                trace_keys = re.findall(
                    '(?<=:).+', header[-1])[0].strip().split(';')
                trace_keys = header[-1].split(',')
                trace_keys = [key.strip('#') for key in trace_keys]
                collect_dict = {}
                for i, key in enumerate(trace_keys):
                    split_key = [k.strip() for k in key.split(":")]
                    if split_key[0] not in collect_dict.keys():
                        collect_dict[split_key[0]] = [(split_key[1],i)]
                    else:
                        collect_dict[split_key[0]].append((split_key[1],i))
                k = 0
            else:
                trace_keys = header[-1].strip().split(",")
                k = 1
                try:
                    collect_name = re.findall('(?<=/)[^/]+(?=\.txt)',fname)[-1]
                except IndexError:
                    collect_name = fname
                except Exception:
                    raise
                collect_dict = {collect_name: []}
                for i, key in enumerate(trace_keys):
                    collect_dict[collect_name].append((key, i))
            # write trace objects
            for member, traces in collect_dict.items():
                self.traces[member] = TATrace(xdata=mat[:, traces[0][1]],
                                              ylabel=ylabel, xlabel=xlabel,
                                              trace_type=tr_type)
                self.active_members.append(member)
                tr = self.traces[member]
                tr.read_xunit()
                tr.get_x_mode()
                k = 1
                for i in range(1, len(traces)):
                    try:
                        key = re.search(
                            '.+(?=fit)|.+(?=y)',traces[i][0]).group(0).strip()
                    except Exception:
                        key = " ".join(["Curve", str(k)])
                    if key not in tr.tr.keys():
                        tr.tr[key] = {}
                        tr.active_traces.append(key)
                    if re.search('fit x', traces[i][0]):
                        tr.tr[key]['fit_x'] = mat[:, traces[i][1]]
                    elif re.search('fit y', traces[i][0]):
                        tr.tr[key]['fit'] = mat[:, traces[i][1]]
                    elif re.search('y', traces[i][0]):
                        tr.tr[key]['y'] = mat[:, traces[i][1]]
                # remove trailing zeros in fit
                try:
                    j = len(tr.tr[key]['fit']) - 1
                except KeyError:
                    pass
                except Exception:
                    raise
                else:
                    while j > 1:
                        if (tr.tr[key]['fit'][j] == 0
                                and tr.tr[key]['fit_x'][j] == 0):
                            j -= 1
                        else:
                            break
                    tr.tr[key]['fit'] = tr.tr[key]['fit'][:j]
                    tr.tr[key]['fit_x'] = tr.tr[key]['fit_x'][:j]
                        
    
        elif fname.endswith("mat"):
            def _todict(obj):
                dct = {}
                for name in obj._fieldnames:
                    field = obj.__dict__[name]
                    if isinstance(field, sio.matlab.mio5_params.mat_struct):
                        dct[name] = _todict(field)
                    else:
                        dct[name] = field
                return dct

            struct = sio.loadmat(fname, mdict=None, appendmat=True,
                                 chars_as_strings=True,
                                 struct_as_record=False, squeeze_me=True)
            for key, val in struct.items():
                if key[0] == "_":
                    continue

                if not isinstance(val, sio.matlab.mio5_params.mat_struct):
                    continue
                fields = _todict(val)
                self.traces[key] = TATrace(xdata=np.array(fields['xdata']))
                ydata = fields['ydata']

                if len(np.shape(ydata)) > 1:
                    numtr = np.shape(ydata)[0]
                    ydata = np.array(ydata)
                    names = np.array(fields['Name'])
                else:
                    numtr = 1
                    ydata = np.array([ydata])
                    names = np.array([fields['Name']])
                for i in range(numtr):
                    name = names[i].strip()
                    kwargs = {}
                    # try:
                    if 'parameters' in fields.keys():
                        kwargs['fit_paras'] = lmfit.Parameters()
                    for k, para in fields['parameters'].items():
                        try:
                            kwargs['fit_paras'].add(k, value=para[i])
                        except TypeError:
                            kwargs['fit_paras'].add(k, value=para)
                        except Exception:
                            raise
                    try:
                        if numtr > 1:
                            kwargs['fit'] = fields['fits_ydata'][i]
                        else:
                            kwargs['fit'] = fields['fits_ydata']
                    except Exception:
                        pass
                    try:
                        if numtr > 1:
                            kwargs['fit_x'] = fields['fits_xdata'][i]
                        else:
                            kwargs['fit_x'] = fields['fits_xdata']
                    except Exception:
                        pass
                    for k, field in fields['misc'].items():
                        kwargs[k] = field
                    self.traces[key].add_trace(ydata[i, :], name, **kwargs)
                for unit in ('xunit', 'yunit'):
                    try:
                        getattr(self.traces[key], "set_" + unit)(fields[unit])
                    except Exception:
                        pass
                self.active_members.append(key)
                if re.search('cm|nm', self.traces[key].get_xunit(), re.I):
                    self.traces[key].type = 'spectral'
                    if re.search('cm', self.traces[key].get_xunit(), re.I):
                        self.traces[key]._xmode = 'wavenumber'
                    else:
                        self.traces[key]._xmode = 'wavelength'
                else:
                    self.traces[key].type = 'kinetic'

    def combine_trace_keys(self, member):
        def get_key_by_name(labels):
            ref_lbl = labels[0]
            commons = []
            for i in range(1, len(labels)):
                for j in range(len(ref_lbl), 0, -1):
                    if ref_lbl[:j - 1] == labels[i][:j - 1]:
                        commons.append(ref_lbl[:j - 1])
                        break
            if len(commons) > 0:
                ind = np.argmin([len(c) for c in commons])
                return commons[ind]

        keys = list(self.traces[member].tr.keys())
        if len(keys) == 0:
            return
        elif len(keys) == 1:
            key = keys[0]
        else:
            key = get_key_by_name(keys)
            if key is None:
                return
        self.add_trace(self.traces[member], key)
        self.remove_trace(member)
        keys = list(self.traces[key].tr.keys())
        for k in keys:
            new_key = k[len(key):]
            self.traces[key].tr[new_key] = {}
            for dct_key, val in self.traces[key].tr[k].items():
                self.traces[key].tr[new_key][dct_key] = val
            self.traces[key].delete_trace(k)


# %%
class TATrace(TAData):
    def __init__(self, trace_type='Kinetic', xdata=None, xlabel="",
                 ylabel="$\Delta$ Abs. (mOD)", xunit=None, yunit="mOD",
                 xrange="all"):
        self.init_common_attributes()
        self._xslice = None
        if xdata is None:
            self.xdata = []
            self.xrange = xrange
        else:
            self.xdata = xdata
            self.set_xrange(xrange)
        self.active_traces = []
        self.values = []
        self.fit_done = False
        self.type = trace_type
        self.xlabel = xlabel
        self.ylabel = ylabel

        if xunit is None:
            self.read_xunit()
        else:
            self.set_xunit(xunit)
        self._yunit = yunit
        self.tr = OrderedDict()

    def add_trace(self, ydata, key, overwrite=True, **misc_data):
        if (not overwrite) and (key in self.tr.keys()):
            j = 2
            new_key = "_".join([key, str(j)])
            while new_key in self.tr.keys():
                j += 1
                new_key = "_".join([new_key.strip("_" + str(j - 1)), str(j)])
            key = new_key
        self.tr[key] = {'y': ydata}
        for k, val in misc_data.items():
            self.tr[key][k] = val
        self.active_traces.append(key)

    def rename_trace(self, in_key, out_key):
        if in_key in self.tr.keys() and in_key != out_key:
            try:
                self.tr[out_key] = {}
                for key, val in self.tr[in_key].items():
                    self.tr[out_key][key] = val
            except Exception:
                return
            else:
                del self.tr[in_key]

    def delete_trace(self, key):
        trace = self.tr.pop(key, None)
        if trace is not None:
            if key in self.active_traces:
                self.active_traces.remove(key)
            return True
        else:
            return False

    def re_order_traces(self, keys):
        for key in self.tr.keys():
            if key not in keys:
                keys.append(key)
        tr = OrderedDict()
        for key in keys:
            try:
                tr[key] = self.tr[key]
            except Exception:
                pass
        self.tr = tr

    def set_all_active(self):
        self.active_traces = []
        for key in self.tr.keys():
            self.active_traces.append(key)

    def clear_all_traces(self):
        keys = list(self.tr.keys())
        for key in keys:
            self.delete_trace(key)
        self.active_traces = []

    def set_xrange(self, xrange=None):
        if xrange is not None:
            self.xrange = xrange
        else:
            xrange = self.xrange
        if type(xrange) is str:
            if xrange == "all" and len(self.xdata) > 0:
                ind = [0, len(self.xdata)]
            else:
                return
        else:
            try:
                lower = np.where(self.xdata >= xrange[0])[0]
                upper = np.where(self.xdata <= xrange[1])[0]
                inds = np.sort([lower[0], lower[-1], upper[0], upper[-1]])
                ind = [inds[1], inds[2] + 1]
            except Exception:
                return
        self._xslice = slice(*ind)
        try:
            return np.sort([self.xdata[ind[0]], self.xdata[ind[1] - 1]])
        except Exception:
            return

    def auto_ylim(self, xrange="all"):
        if not self._xslice:
            self.set_xrange(xrange=xrange)
        tr_max = []
        tr_min = []
        for k in self.active_traces:
            try:
                y = self.tr[k]['y'][self._xslice]
                y_mask = y[np.isfinite(y)]
                tr_max.append(np.max(y_mask))
                tr_min.append(np.min(y_mask))
            except Exception:
                pass
        if len(tr_max) > 0:
            mn = np.min(tr_min)
            mx = np.max(tr_max)
            return [mn*0.9 if mn > 0 else mn*1.1,
                    mx*1.1 if mx > 0 else mx*0.9]
        else:
            return None

    def get_value_list(self):
        try:
            return [val['val'] for val in self.tr.values()]
        except Exception:
            return []

    def get_value(self, key):
        try:
            return self.tr[key]['val']
        except Exception:
            return None

    def sort_by_value(self):
        try:
            values = self.get_value_list()
            keys = list(self.tr.keys())
            ind = np.argsort(values)
        except Exception:
            return None
        else:
            self.re_order_traces([keys[i] for i in ind])
            return [keys[i] for i in ind], [values[i] for i in ind]

    def move_trace(self, ind, target):
        self.active_traces.insert(target, self.active_traces.pop(ind))

    def move_trace_by_step(self, ind, step=-1):
        try:
            self.move_trace(ind, ind + step)
        except Exception:
            pass

    def get_nearest_trace(self, val):
        values = np.array(self.get_value_list())
        try:
            next_trace_up = np.where(values >= val)[0][0]
        except Exception:
            # ind = -1
            raise
        else:
            try:
                next_trace_down = np.where(values <= val)[0][-1]
            except Exception:
                ind = 0
            else:
                if (np.abs(values[next_trace_up] - val)
                        < np.abs(values[next_trace_down] - val)):
                    ind = next_trace_up
                else:
                    ind = next_trace_down
        return self.active_traces[ind], ind

    def get_x_mode(self):
        if re.search('kin|time', self.type, re.I):
            self._xmode = 'time'
        else:
            if re.search('wavelen', self.xlabel, re.I):
                self._xmode = 'wavelength'
            elif re.search('wavenum', self.xlabel, re.I):
                self._xmode = 'wavenumber'
            elif re.search('energ', self.xlabel, re.I):
                self._xmode = 'energy'
            else:
                self.xlabel = 'wavelength (nm)'
                self._xmode = 'wavelength'
        mode = self._xmode
        return mode

    def set_x_mode(self, mode='wavelength'):
        try:
            keys = self._spec_conversion.keys()
        except Exception:
            self._init_spec_conversion_dict()
            keys = self._spec_conversion.keys()
        if mode in keys:
            std_units = {'wavelength': 'nm',
                         'wavenumber': 'cm^(-1)',
                         'energy': 'eV'}
            try:
                xdata = self.convert_spec_values(
                    self.xdata, x_in=self._xmode, x_out=mode)
            except Exception:
                self.get_x_mode()
                try:
                    xdata = self.convert_spec_values(
                        self.xdata, x_in=self._xmode, x_out=mode)
                except Exception:
                    raise
            if xdata is not None:
                self.xdata = xdata
            self._xmode = mode
            self.spec_unit = std_units[mode]

        elif re.search('time', mode, re.I):
            self._xmode = mode
        else:
            return
        self.xlabel = self.set_xlabel()

    def assign_axes_labels(self, case_dict):
        for key in self.active_traces:
            for case in case_dict.keys():
                if re.search(case, key, re.I):
                    for k in case_dict[case].keys():
                        self.tr[key][k] = case_dict[case][k]
                    continue

    def init_exp_fit(self, fit_obj=None, num_exp=1, offset=True, irf=None,
                     irf_fixed=False,
                     t0=0, cos_modul=None, fix_modul_paras=False,
                     damp_comp=None, fix_t0=False, func_type='exp',
                     **line_fit_kwargs):
        # translate model options to model name for LineFit object
        fixed_para = {}
        if func_type == 'exp':
            model = 'exp' + str(num_exp)
        else:
            model = func_type
            if re.search('inf', func_type.lower()):
                try:
                    num = re.findall('(?<=inf)\d+', model)[0]
                except Exception:
                    num = num_exp
                fixed_para['kineticexp_tau_' + num] = np.inf
        if cos_modul is not None:
            model += 'osc' + str(cos_modul)
        if damp_comp is not None and damp_comp > 0:
            model += 'damp' + str(damp_comp)
        if offset:
            model += '_const'
        if fix_t0:
            fixed_para['x0'] = t0
        if irf is not None:
            model += '_conv'
            if irf_fixed:
                fixed_para['sigma_1'] = irf
        if fit_obj is None:
            return LineFit(model=model, fixed_para=fixed_para,
                           **line_fit_kwargs)
        else:
            fit_obj.set_attributes(model=model, fixed_para=fixed_para,
                                   **line_fit_kwargs)
            return fit_obj

    def run_exp_fit(self, fit_obj=None, fit_range=None, inits=None,
                    bounds=None, **fit_obj_kwargs):
        # initialize exponential fit object
        if fit_obj is None:
            fit_obj = self.init_exp_fit(**fit_obj_kwargs)
        # run fit and return results
        return self.run_fit(fit_obj, fit_range=fit_range, inits=inits,
                            bounds=bounds)

    def run_fit(self, fit_obj, fit_range=None, inits=None, bounds=None,
                xdata=None, queue=None, calculate_components=True,
                recursive_guess=False, **kwargs):
        # process inputs
        if inits is None or inits == 'auto':
            guesses = {}
            for key in self.active_traces:
                guesses[key] = inits
        else:
            guesses = inits
        if recursive_guess:
            for i in range(1, len(self.active_traces)):
                guesses[self.active_traces[i]] = None
        if bounds is None or bounds == 'auto':
            bounds = {}
            for key in self.active_traces:
                bounds[key] = 'auto'
        prev_inits = {}
        paras = {}
        report = {}
        success = {}
        if not xdata:
            xdata = self.xdata
        if fit_range:
            xdata = xdata[fit_range[0]:fit_range[1]]
        n = len(self.active_traces)
        # run fit loop over active traces
        for i, key in enumerate(self.active_traces):
            # set x and y data
            ydata = np.array(self.tr[key]['y'])
            if fit_range:
                ydata = ydata[fit_range[0]:fit_range[1]]

            valid = ~np.isnan(ydata)
            fit_obj.x = np.array(xdata)[valid]
            fit_obj.y = np.array(ydata)[valid]
            # run fit
            success[key], report[key] = fit_obj.run_fit(
                dy=None, guesses=guesses[key], bounds=bounds[key])
            # write attributes and output
            prev_inits[key] = fit_obj.params.valuesdict()
            try:
                paras[key] = fit_obj.result.params
            except Exception:
                continue

            if calculate_components and success[key]:
                curve, comps = fit_obj.fit_function(
                    fit_obj.result.params, return_comps=True)(fit_obj.x)
                if not type(comps) is dict:
                    self.tr[key]['fit_comps'] = {}
                    for i in range(np.shape(comps)[0]):
                        try:
                            c = comps[i, :]
                        except Exception:
                            try:
                                c = comps[i]
                            except Exception:
                                break
                        self.tr[key]['fit_comps']['Comp. ' + str(i + 1)] = c
                else:
                    self.tr[key]['fit_comps'] = comps
            else:
                curve = fit_obj.curve
            if recursive_guess:
                fit_obj.set_init_to_result()

            self.tr[key]['fit_paras'] = fit_obj.result.params
            self.tr[key]['fit_subfunctions'] = fit_obj.sub_functions
            self.tr[key]['fit'] = curve
            self.tr[key]['residual'] = fit_obj.residual
            self.tr[key]['fit_x'] = fit_obj.x
            self.tr[key]['fit_outliers'] = np.array(fit_obj.outliers)
            if fit_obj.stderrors is None:
                (self.tr[key]['fit_error_lower'],
                 self.tr[key]['fit_error_upper']) = None, None
            else:
                (self.tr[key]['fit_error_lower'],
                 self.tr[key]['fit_error_upper']) = fit_obj.get_error_curves()
            self.tr[key]['fit_para_errors'] = fit_obj.stderrors
            if queue is not None:
                queue.put({'point': i, 'n': n})
        return paras, prev_inits, report, success

    def residual_fft(self, xunit=None):
        if not xunit:
            xunit = self.get_xunit()
        for key in self.active_traces:
            try:
                self.tr[key]['residual']
            except Exception:
                continue
            fft, self.tr[key]['resid_fft_x'], err = self.fft_map(
                dat=[self.tr[key]['residual']],
                x=self.tr[key]['fit_x'],
                xunit=xunit)
            if err:
                print(err)
            self.tr[key]['resid_fft'] = fft[0]

    def plot_fit_error(self, ax, key, color='black', alpha=0.15,
                       transpose=False):
        fill_between(ax, self.tr[key]['fit_x'],
                     self.tr[key]['fit_error_lower'],
                     self.tr[key]['fit_error_upper'],
                     transpose=transpose,
                     label='_nolegend_',
                     alpha=alpha, color=color)

    def plot_residual(self, *args, **kwargs):
        if 'plot_case' in kwargs.keys():
            if not re.search('resid', kwargs['plot_case'], re.I):
                kwargs['plot_case'] = 'residual'
        return self.plot_for_active_traces(*args, **kwargs)

    def plot_for_active_traces(self, *plot_args, plot_case='residual',
                               fig=None, ax=None, range_ind=None, fill=0,
                               reverse_z=False, color_order=None,
                               interpolate_colors=False, color_cycle=None,
                               clear_ax=True, set_ax_labels=True,
                               transpose=False,  **plot_kwargs):
        if re.search('resid(.*)fft', plot_case, re.I):
            def y(key): return self.tr[key]['resid_fft']
            def x(key): return self.tr[key]['resid_fft_x']
            title = 'FFT of residual'
        elif re.search('resid', plot_case, re.I):
            def y(key): return self.tr[key]['residual']
            def x(key): return self.tr[key]['fit_x']
            title = 'Residual'
        else:
            return
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        if clear_ax:
            ax.cla()
        lines = []
        fill = get_fill_list(fill, len(self.active_traces))
        for i, key in enumerate(self.active_traces):
            if range_ind is None:
                ydata = y(key)
                xdata = x(key)
            else:
                ydata = y(key)[slice(*range_ind)]
                xdata = x(key)[slice(*range_ind)]
            lines.append(
                plot(ax, xdata, ydata, transpose=transpose, **plot_kwargs))
            if fill[i]:
                fill_between(ax, xdata, ydata,
                             label='_nolegend_',
                             transpose=transpose, alpha=fill[i])
        if set_ax_labels:
            ax.set_title(title)
            try:
                ax.set_xlabel(self.xlabel)
            except Exception:
                pass
            try:
                ax.set_ylabel(self.ylabel)
            except Exception:
                pass
        return lines

    def plot_residual_fft(self, *args, **kwargs):
        return self.plot_for_active_traces(
            *args, plot_case='residual_fft', **kwargs)

    def run_line_shape_fit(self, fit_obj=None, fit_range=None, inits=None,
                           bounds=None, convert_x=False, **fit_obj_kwargs):
        if fit_obj is None:
            fit_obj = self.init_line_shape_fit(**fit_obj_kwargs)
        if convert_x:
            if re.search('wavenum', convert_x, re.I):
                x = self.set_wavenumbers(self.xdata)
            else:
                x = self.xdata
        else:
            x = self.xdata
        return self.run_fit(fit_obj, fit_range=fit_range, inits=inits,
                            bounds=bounds, xdata=x)

    def read_xunit(self):
        try:
            xunit = self.xlabel[
                re.search(re.compile("\("), self.xlabel).end():
                    re.search(re.compile("\)"), self.xlabel).end()-1]
        except Exception:
            self._xunit = "a.u."
        else:
            self.set_xunit(xunit)
        return self._xunit

    def set_xunit(self, xunit):
        self._xunit = xunit

    def set_yunit(self, yunit):
        self._yunit = yunit

    def get_xunit(self):
        return self._xunit

    def get_yunit(self):
        return self._yunit

    def save(self, file, save_fit=False, **kwargs):
        return self.save_traces(file, save_fit=save_fit, **kwargs)

    def save_traces(self, file, trace_keys='all', spec_quantity='wavelength',
                    spec_unit='nm', trace_type='Kinetic', ylabel=None,
                    custom_para=None, save_fit=True, range_ind=None):
        """ save all or selected traces

            positional arguments:
                file is either a string containing the file path or
                an object with attribute .name = file path (string)
        """
        if type(file) is str:
            fname = file
        else:
            fname = file.name
        if trace_keys == 'all':
            trace_keys = self.active_traces
        integral_prefix = ["Spectral Trace of Temporal ",
                           "Kinetic Trace of Spectral "]
        if re.search('time', self.xlabel, re.I):
            int_prefix_ind = 1
        else:
            int_prefix_ind = 0
        save_dict = {'Kinetic': ['Kinetic Trace(s)', spec_quantity],
                     'CPM': ['CPM Trace(s)', spec_quantity],
                     'Spectral': ['Spectral Trace(s)', 'time_delays'],
                     'Integral': [integral_prefix[int_prefix_ind]
                                  + trace_keys[0],
                                  spec_quantity],
                     'Centroid': [trace_keys[0], spec_quantity],
                     'Lineshape': ['Lineshape Trace(s)', 'time_delays'],
                     'DAS': ['Decay-associated spectra (by decay times)',
                             'tau'],
                     'SAS': ['Species-associated spectra (by life times)',
                             'tau'],
                     'Left SV': ['Left Singular Values', 'time delay'],
                     'Right SV': ['Right Singular Values', spec_quantity],
                     None: ['Trace(s)', 'Name']}

        if fname.split('.')[-1] == 'txt':
            self.save_traces_txt(fname, save_dict[trace_type], trace_keys,
                                 range_ind=range_ind, save_fit=save_fit,
                                 ylabel=ylabel)
        elif fname.split('.')[-1] == 'mat':
            self.save_traces_mat(fname, save_fit=save_fit,
                                 trace_keys=trace_keys, trace_type=trace_type,
                                 trace_label_name=save_dict[trace_type][1],
                                 custom_para=custom_para)
        else:
            print('Invalid file type.')

    def save_traces_txt(self, fname, tr_info, trace_keys,
                        range_ind=None, save_fit=False, ylabel=None,
                        trace_obj=None, writefile=True):
        if trace_obj is None:
            trace_obj = self
        header = tr_info[0]
        if save_fit:
            header += ' fit\n\n'
        else:
            header += '\n\n'
        header += 'First column: x values\nFollowing column(s): '
        if save_fit:
            header += (' ' + tr_info[0] +
                       ', fit x values, fit y values for each trace.')
        else:
            header += tr_info[0] + '.'
        header += '\nX-Axis: ' + trace_obj.xlabel + '\n'
        if ylabel is None:
            ylabel = trace_obj.ylabel
        header += 'Y-Axis: ' + ylabel + '\n'
        if save_fit:
            save_matrix = np.zeros(
                (3*len(trace_keys) + 1, len(trace_obj.tr[trace_keys[0]]['y'])))
            save_matrix[0, :len(trace_obj.xdata)] = trace_obj.xdata
            for i, key in enumerate(trace_keys):
                header += "".join([key, ' x, ', key, ' y,'])
                tr = trace_obj.tr[key]
                save_matrix[3*i + 1, :len(tr['y'])] = tr['y']
                try:
                    save_matrix[3*i + 2, :len(tr['fit_x'])] = tr['fit_x']
                    save_matrix[3*i + 3, :len(tr['fit'])] = tr['fit']
                    header = " ".join([header, key, 'fit x,', key, 'fit y, '])
                except Exception:
                    pass
        else:
            if range_ind is None:
                range_ind = slice(len(trace_obj.xdata))
            save_matrix = np.concatenate(
                ([trace_obj.xdata[range_ind]],
                 [trace_obj.tr[trace_keys[0]]['y'][range_ind]]),
                axis=0)
            header += " ".join(['x,', trace_keys[0], 'y, '])
            for i in range(1, len(trace_keys)):
                save_matrix = np.concatenate(
                    (save_matrix,
                     [trace_obj.tr[trace_keys[i]]['y'][range_ind]]), axis=0)
                header += trace_keys[i] + ' y, '
        header = header[:-2]
        if writefile:
            with open(fname, 'w') as f:
                np.savetxt(f, np.transpose(save_matrix), delimiter=',',
                           header=header)
                f.close()
        return np.transpose(save_matrix), header

    def save_traces_mat(self, fname, save_fit=False, trace_keys=None,
                        trace_type='Kinetic', save_all_fit_attr=True,
                        trace_label_name='wavelengths', custom_para=None,
                        trace_obj=None, writefile=True):
        if trace_obj is None:
            trace_obj = self
        if trace_keys is None:
            trace_keys = []
        if custom_para is None:
            custom_para = {}
        vars_to_save = {
            'xdata': np.array(trace_obj.xdata),
            'ydata': [trace_obj.tr[key]['y'] for key in trace_keys],
            'xunit': self.get_xunit(),
            'yunit': self.get_yunit()}
        if save_fit:
            vars_to_save['parameters'] = {}
            if save_all_fit_attr:
                vars_to_save['misc'] = {'fit_subfunctions': [],
                                        'fit_para_errors': [],
                                        'fit_error_lower': [],
                                        'fit_error_upper': [],
                                        'fit_comps': [],
                                        'residual': []}
            else:
                vars_to_save['misc'] = {}
            vars_to_save['fits_ydata'] = []
            vars_to_save['fits_xdata'] = []
            for key in trace_keys:
                try:
                    vars_to_save['fits_ydata'].append(trace_obj.tr[key]['fit'])
                    vars_to_save['fits_xdata'].append(
                        trace_obj.tr[key]['fit_x'])
                    for k, para in trace_obj.tr[key]['fit_paras'].items():
                        if k in vars_to_save['parameters'].keys():
                            vars_to_save['parameters'][k].append(para.value)
                        else:
                            vars_to_save['parameters'][k] = [para.value]
                    for k, field in vars_to_save['misc'].items():
                        try:
                            field.append(trace_obj.tr[key][k])
                        except Exception as e:
                            print(e)
                except Exception:
                    pass
        try:
            vars_to_save[trace_label_name] = np.array(
                [np.double(trk) for trk in trace_keys])
        except Exception:
            vars_to_save[trace_label_name] = np.array(
                [trk for trk in trace_keys])
        for k in custom_para.keys():
            vars_to_save['parameters'][k] = custom_para[k]
        if writefile:
            sio.savemat(fname, vars_to_save)
        return vars_to_save

    def plot(self, *args, fig=None, ax=None, fill=0, fit_kwargs=None,
             include_fit=False,
             plot_fit_error=False, xmode='values', ymode=None,
             show_legend=False, reverse_z=False, color_order=None,
             color_cycle=None, interpolate_colors=False,
             active_traces='all', cla=True, include_fit_comps=False,
             fit_comp_alpha=0.3, show_fit_legend=False,
             transpose=False,
             plot_outliers=False,
             **plot_kwargs):
        def log_values(x, y):
            try:
                f = getattr(np, re.search('log\d*', xmode, re.I).group(0))
                x = np.array([f(val) for val in x])
            except Exception:
                return [x, y]
            else:
                return [x[np.isfinite(x)], y[np.isfinite(x)]]
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        if fit_kwargs is None:
            fit_kwargs = {}
        if 'color' not in fit_kwargs.keys():
            fit_kwargs['color'] = 'black'
        if not show_fit_legend:
            fit_kwargs['label'] = '_nolegend_'
        if active_traces == 'all':
            active_traces = self.active_traces
        if cla:
            ax.cla()
        fill = get_fill_list(fill, len(active_traces))

        if color_order is None:
            cyc_len = len(active_traces)
            if include_fit_comps:
                for k in active_traces:
                    for comp in self.tr[k]['fit_comps'].keys():
                        cyc_len += 1
        else:
            cyc_len = np.max(color_order['ind']) + len(active_traces)
        zord, cycle = set_line_cycle_properties(
            cyc_len, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)
        lines = []
        i = 0
        if re.search('val', xmode, re.I):
            if re.search('log', xmode, re.I):
                xfunc = log_values
            else:
                def xfunc(x, y): return [x, y]
        else:
            def xfunc(x, y): return [np.arange(len(x)), y]
        for j, k in enumerate(active_traces):
            lines.append(plot(ax, *xfunc(self.xdata, self.tr[k]['y']),
                              transpose=transpose, zorder=zord[i],
                              color=cycle[i], **plot_kwargs))
            if fill[j]:
                lines.append(fill_between(ax,
                                          *xfunc(self.xdata, self.tr[k]['y']),
                                          zorder=zord[i],
                                          color=cycle[i],
                                          alpha=fill[j],
                                          transpose=transpose,
                                          label='_nolegend_'))
            if include_fit:
                try:
                    lines.append(plot(ax, *xfunc(self.tr[k]['fit_x'],
                                                 self.tr[k]['fit']),
                                      transpose=transpose, **fit_kwargs))
                except Exception:
                    pass
                if plot_fit_error:
                    try:
                        self.plot_fit_error(ax, k, transpose=transpose,
                                            color=fit_kwargs['color'])
                    except Exception:
                        pass
                if plot_outliers:
                    try:
                        lines.append(plot(ax, self.tr[k]['fit_outliers'][0, :],
                                          self.tr[k]['fit_outliers'][1, :],
                                          marker='o', color='red',
                                          linestyle='None'))
                    except Exception:
                        traceback.print_exc()
            i += 1
            if include_fit_comps:
                try:
                    for comp in self.tr[k]['fit_comps'].values():
                        lines.append(fill_between(
                            ax, *xfunc(self.tr[k]['fit_x'], comp),
                            zorder=zord[i], color=cycle[i],
                            alpha=fit_comp_alpha, transpose=transpose,
                            label='_nolegend_'))
                        i += 1
                except KeyError:
                    pass
                except Exception:
                    raise
        return lines

    def plot_single(self, i, *args, **kwargs):
        try:
            key = self.active_traces[i]
        except Exception:
            return
        else:
            return self.plot(*args, active_traces=[key], **kwargs)

    def run_cpm_fit(self, fit_obj=None, fit_range=None, inits=None,
                    bounds=None, queue=None, calculate_components=True,
                    recursive_guess=True, **fit_obj_kwargs):
        if fit_obj is None:
            fit_obj = self.init_cpm_fit(**fit_obj_kwargs)
        return [fit_obj,
                *self.run_fit(fit_obj, fit_range=fit_range, inits=inits,
                              bounds=bounds, queue=queue,
                              calculate_components=calculate_components,
                              recursive_guess=recursive_guess)]

    def init_cpm_fit(self, func='hermitepoly', n=5, offset=False, fit_obj=None,
                     modul_n=1, **fit_obj_kwargs):
        if func == 'hermitepoly':
            model = 'hermite' + str(n)
        elif func == 'gaussian':
            model = 'gauss' + str(n)
        elif func == 'gaussiansine':
            model = str(n).join(['gauss', 'osc']) + str(modul_n)
        if offset:
            model += 'const'
        if not fit_obj:
            fit_obj = LineFit(model=model, **fit_obj_kwargs)
        else:
            fit_obj.set_attributes(model=model, **fit_obj_kwargs)
        return fit_obj


# %%
class DataToPlot():
    def __init__(self, plot_type='2D'):
        self.x = []
        self.y = []
        self.z = []
        self.plot_type = plot_type
        self.xlims = []
        self.ylims = []
        self.clims = [-10, 10]
        self.cmap = plt.get_cmap('RdBu_r')
        self.flag = False
        self.xlabel = " "
        self.ylabel = " "
        self.clabel = '$\Delta$ Abs.'
        self.plot_kwargs = {}

    def write_properties(self, **kwargs):
        for key, val in kwargs.items():
            exec('self.' + key + ' = val')

    def plot_2d(self, *args, fig=None, ax=None, canv=None,
                x=None, y=None, z=None, transpose=False, **kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        ax.cla()
        try:
            for key, val in zip(('vmin', 'vmax'),
                                (self.clims[0], self.clims[1])):
                if key not in kwargs.keys():
                    kwargs[key] = val
        except Exception:
            pass
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = self.cmap
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z
        self.image = pcolor(ax, x, y, z, transpose=transpose, **kwargs)
        return self.image

    def plot_line(self, *args, fig=None, ax=None, fill=0, reverse_z=False,
                  color_order=None, color_cycle=None, interpolate_colors=False,
                  transpose=False,  **kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        ax.cla()
        numlines = np.shape(self.y)[0]
        fill = get_fill_list(fill, numlines)
        zord, cycle = set_line_cycle_properties(
            numlines, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)
        for i in range(numlines):
            plot(ax, self.x, self.y[i], zorder=zord[i], color=cycle[i],
                 transpose=transpose, **kwargs)
            if fill[i]:
                fill_between(ax, self.x, self.y[i], zorder=zord[i],
                             label='_nolegend_',
                             color=cycle[i],
                             transpose=transpose, alpha=fill[i])
        try:
            ax.set_xlim(self.xlims)
        except Exception:
            pass
        try:
            ax.set_ylim(self.ylims)
        except Exception:
            pass
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


# %%
class TAParameterDependence(TAData):
    def __init__(self, *args, **kwargs):
        TAData.__init__(self, *args, **kwargs)
        self.all_timesteps_equal = True
        self.all_timesteps_same = True
        self.sortmode = 'folder'
        self.x_para = "Power"
        self.x_para_original = "Power"
        self.x_unit = 'mW'
        self.sorted_scans = {}
        self.x = []
        self.y = []
        self.rep_rate = 1e3         # Hz
        self.beam_size = 1e-4       # m
        self.fit_plot_resolution = 50
        self.fit_obj = None
        self.disp_fit_func = False
        self.data_loaded = False

    def load_scans(self, path, filetypes, **kwargs):
        self.all_timesteps_equal, self.all_timesteps_same = (
            self.read_scan_files(path, filetypes, import_subfolders=True,
                                 **kwargs))
        self.data_loaded = True
        self.fit_obj = None

    def sort_scans_and_average(self, out_queue=None, sortmode='folder',
                               **averagingopts):
        self.sorted_scans = {}
        if re.search('power', sortmode, re.I):
            self.sortmode = 'power'
        else:
            self.sortmode = sortmode
        if self.sortmode == 'folder' or self.sortmode == 'power':
            for k, scan in self.raw_scans.items():
                try:
                    if self.sortmode == 'folder':
                        key = k.split('\\')[0]
                        key = key.split('/')[0]
                    else:
                        key = str(np.mean(scan['power']))
                except Exception:
                    continue
                else:
                    if key not in self.sorted_scans.keys():
                        self.sorted_scans[key] = {
                            'scan': TAData(), 'paths': [k]}
                    else:
                        self.sorted_scans[key]['paths'].append(k)
            for key, val in self.sorted_scans.items():
                if out_queue is not None:
                    out_queue.put([key, val['paths']])
                self.average_scans(
                    self.all_timesteps_equal, self.all_timesteps_same,
                    files_to_average=val['paths'], **averagingopts)
                val['scan'].delA = self.delA
                val['scan'].wavelengths = self.wavelengths
                val['scan'].time_delays = self.time_delays
            if self.sortmode == 'folder':
                self.read_unit()
            else:
                self.x_unit = 'V'
        else:
            return
        self.auto_xlabel()
        self.x_para_original = self.x_para
        if out_queue is not None:
            out_queue.put(False)

    def read_unit(self):
        keys = list(self.sorted_scans.keys())
        try:
            self.x_unit = re.search("[a-zA-Z]+", keys[0])[0]
        except Exception:
            return
        else:
            for k in keys:
                try:
                    unit = re.search("[a-zA-Z]+", k)[0]
                except Exception:
                    continue
                if unit != self.x_unit:
                    print(" ".join(
                        ["Multiple units identified:", self.x_unit, unit])
                        + '.\nUsing ' + self.x_unit)

    def auto_xlabel(self):
        if self.x_unit[-1] == 'W':
            self.x_para = "Power"
        elif self.x_unit[-1] == 'J' or self.x_unit[-2:] == 'eV':
            self.x_para = 'Energy'
        elif re.search('Jcm\-2', self.x_unit):
            self.x_para = 'Fluence'
        elif self.x_unit[-1] == 'V':
            self.x_para = "Voltage"
        else:
            return
        self.x_label = self.x_para + ' (' + self.x_unit + ')'
        return self.x_label

    def convert_para(self, convert_to="power", convert_to_unit='mW',
                     rep_rate=None, beam_size=None):
        def get_unit_factor(case, unit):
            if case.lower() == "power" or case.lower() == "energy":
                base_unit = "W" if case.lower() == "power" else "J"
                try:
                    pre = re.search('[a-zA-Z](?=' + base_unit + ')', unit)[0]
                    factor = prefixes[pre]
                except Exception:
                    return
                else:
                    return factor
            elif case.lower() == "fluence":
                try:
                    pre1 = re.search('[a-zA-Z](?=J)', unit)[0]
                    pre2 = re.search('[a-zA-Z](?=m(\^|\-|2))', unit)[0]
                    factor = prefixes[pre1]/(prefixes[pre2]**2)
                except Exception:
                    return
                else:
                    return factor
            else:
                return

        if rep_rate is None:
            rep_rate = self.rep_rate
        else:
            self.rep_rate = rep_rate
        if beam_size is None:
            beam_size = self.beam_size
        else:
            self.beam_size = beam_size
        area = 0.5*np.pi*(beam_size*0.8493218)**2
        conversions = {"power":  {"power": lambda x: x,
                                  "energy": lambda x: x/rep_rate,
                                  "fluence": lambda x: x/(rep_rate*area)},
                       "energy": {"power": lambda x: x*rep_rate,
                                  "energy": lambda x: x,
                                  "fluence": lambda x: x/area},
                       "fluence": {"power": lambda x: x*area*rep_rate,
                                   "energy": lambda x: x*area,
                                   "fluence": lambda x: x}}
        prefixes = {"p": 1e-12,
                    "n": 1e-9,
                    "u": 1e-6,
                    "m": 1e-3,
                    "c": 1e-2,
                    "d": 1e-1,
                    "k": 1e3,
                    "M": 1e6,
                    "G": 1e9}
        if not self.x_para.lower() in conversions.keys():
            return
        self.x = get_unit_factor(self.x_para, self.x_unit) * \
            conversions[self.x_para.lower()][convert_to.lower()](self.x)
        factor = get_unit_factor(convert_to, convert_to_unit)
        if factor is not None:
            self.x = self.x / factor
            self.x_unit = convert_to_unit
        else:
            if convert_to.lower() == "power":
                self.x_unit = 'mW'
            elif convert_to.lower() == "energy":
                self.x_unit = 'nJ'
            elif convert_to.lower() == 'fluence':
                self.x_unit = 'mJcm^-2'
            self.x = self.x / get_unit_factor(convert_to, self.x_unit)
        self.x_para = convert_to
        self.x_label = self.x_para + ' (' + self.x_unit + ')'

    def calculate_values(self, method='svd', x_limits=None, y_limits=None,
                         out_queue=None, num_comp=1):
        def get_xvalues_from_str():
            x = []
            for k in self.sorted_scans.keys():
                try:
                    val = np.double(re.search('\d+', k)[0])
                except Exception:
                    return
                else:
                    x.append(val)
            return np.array(x)
        if self.sortmode == 'folder':
            self.x = get_xvalues_from_str()
        else:
            self.x = np.array([np.double(k) for k in self.sorted_scans.keys()])
        self.y = []
        for key, scan in self.sorted_scans.items():
            if x_limits is None:
                scan['scan']._lambda_lim_index = [
                    0, len(scan['scan'].wavelengths)]
            else:
                scan['scan'].set_xlimits(x_limits)
            if y_limits is None:
                scan['scan']._time_range_indices = [
                    0, len(scan['scan'].time_delays)]
            else:
                scan['scan'].set_time_win_indices(limits=y_limits)
        if method.lower() == 'svd':
            for key, scan in self.sorted_scans.items():
                u, s, v = np.linalg.svd(
                    scan['scan'].delA[slice(*scan['scan']._time_range_indices),
                                      slice(*scan['scan']._lambda_lim_index)],
                    full_matrices=False)
                try:
                    self.y.append(np.sum(s[0:num_comp]))
                except Exception as e:
                    print(e)
        elif re.search('integr', method, re.I):
            for key, scan in self.sorted_scans.items():
                s = np.sum(np.sum(
                    scan['scan'].delA[slice(*scan['scan']._time_range_indices),
                                      slice(*scan['scan']._lambda_lim_index)]))
                self.y.append(s)
        ind = np.argsort(self.x)
        self.x = self.x[ind]
        self.y = np.array(self.y)[ind]

    def plot_function(self, *args, ax=None, fig=None, marker='x',
                      linestyle=' ', transpose=False, zord=None, markersize=10,
                      color='k', **kwargs):
        if ax is None:
            if fig is None:
                return
            else:
                ax = fig.add_subplot(111)
        ax.cla()
        plot(ax, self.x, self.y, marker=marker, linestyle=linestyle,
             transpose=transpose, zorder=1, label='data',
             color=color, markersize=markersize)
        if self.fit_obj is not None:
            plot(ax, self.fit_x_plot, self.fit_y_plot, transpose=transpose,
                 zorder=2, label='fit')
            plot(ax, self.fit_x, self.fit_y, 'o', transpose=transpose,
                 zorder=0, label='fitpoints')
            if self.disp_fit_func:
                text_in_plot(ax, 0.1, 0.9, self.fit_disp,
                             transform=ax.transAxes, transpose=transpose)

    def fit(self, fit_func='linear', xrange=None, fit_plot_range=None):
        if not xrange:
            self.fit_x = self.x
            self.fit_y = self.y
        else:
            try:
                xlower_ind = np.where(self.x >= xrange[0])[0][0]
                xupper_ind = np.where(self.x <= xrange[1])[0][-1] + 1
                self.fit_x = self.x[slice(xlower_ind, xupper_ind)]
                self.fit_y = self.y[slice(xlower_ind, xupper_ind)]
            except Exception:
                self.fit_x = self.x
                self.fit_y = self.y
        fit_obj = LineFit(x=self.fit_x, y=self.fit_y, model=fit_func)
        fit_obj.run_fit()
        self.fit_obj = fit_obj
        if fit_plot_range is None:
            self.fit_x_plot = np.linspace(self.fit_obj.x[0],
                                          self.fit_obj.x[-1],
                                          num=self.fit_plot_resolution)
        else:
            self.fit_x_plot = np.linspace(fit_plot_range[0], fit_plot_range[1],
                                          num=self.fit_plot_resolution)
        self.fit_y_plot = self.fit_obj.fit_function(
            self.fit_obj.result.params)(self.fit_x_plot)
        if fit_func in ('linear', 'poly2', 'poly3'):
            try:
                func = re.match('\D+', fit_func).group(0)
            except Exception:
                func = fit_func
            p = self.fit_obj.result.params.valuesdict()
            numbers = [str(np.round(p['_'.join([func, 'a_1'])], 3))]
            for key in list(p.keys())[1:]:
                n = np.round(p[key], 3)
                if n < 0:
                    numbers.append("- " + str(np.abs(n)))
                else:
                    numbers.append("+ " + str(n))

            if fit_func == 'linear':
                self.fit_disp = "{} x {}".format(*numbers)
            elif fit_func == 'poly2':
                self.fit_disp = "{} $x^2$ {} x {}".format(*numbers)
            elif fit_func == 'poly3':
                self.fit_disp = "{} $x^3$ {} $x^2$ {} x {}".format(*numbers)
            self.disp_fit_func = True
        else:
            self.disp_fit_func = False


# %%
class StaticAbsData():
    def __init__(self):
        self.wavelengths = None
        self.abs = None
        self.ref = None
        self.bkg = None
        self.trans = None
        self.baseline = 0
        self.abs_spectra = {}
        self.abs_nonfilt = {}
        self.magnif_factors = {}
        self.diff_abs = {}
        self._active_key = ""

    def set_active_spec(self, index=0):
        try:
            self._active_key = list(self.abs_spectra.keys())[index]
        except IndexError:
            self._active_key = self.abs_spectra.keys()
        except Exception:
            pass
        else:
            self.abs = self.abs_spectra[self._active_key][1]

    def get_active_key(self):
        return self._get_abs_keys('active')[0]

    def calc_properties_crystal(self, od=1.0, length=1e-7, num_dens=1e21,
                                photon_energy=5e-19, fluence=2.0):
        # length in m
        # number density in cm^-3
        # photon energy in J
        # fluence in mJ*(cm^-2)
        # od = -log10(Transmission)
        self.abs_coeff = od/length*1e-2
        self.abs_cross = self.abs_coeff/num_dens
        self.penetr_depth = 1/(self.abs_coeff)*1e7  # nm
        self.photon_flux = fluence*1e-3/photon_energy
        photon_abs_ratio = 1 - np.power(10, -od)
        self.exc_frac = float(
            self.photon_flux*photon_abs_ratio/(length*1e2*num_dens)*100)
        self.fluence = fluence

    def calc_properties_solution(self, od=1.0, length=1e-3, conc=1e-3,
                                 photon_energy=5e-19, fluence=2.0):
        # input parameter units: length in meters,
        # conc (= concentration) in mol/L
        # photon_energy in Joules, fluence in mJcm^(-2)
        self.fluence = fluence
        self.abs_coeff = od/length*1e-2
        self.abs_coeff_mol = od/(length*conc)*1e-2
        num_dens = conc*6.022e23*1e-3  # cm^-3
        self.abs_cross = self.abs_coeff/num_dens
        self.penetr_depth = 1/(self.abs_coeff)*10  # mm
        self.photon_flux = fluence*1e-3/photon_energy
        photon_abs_ratio = 1-np.power(10, -od)
        self.exc_frac = float(
            self.photon_flux*photon_abs_ratio/(length*1e2*num_dens)*100)
        return num_dens

    def clear_active_spec(self):
        del self.abs_spectra[self._active_key]
        self.set_active_spec(index=0)

    def clear_data(self, *args):
        self.abs_spectra = {}
        self.abs_nonfilt = {}
        self.abs = None

    def load_abs_spectrum(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            return False
        except Exception:
            raise
        else:
            self.abs_fname = re.split('\.', re.split('\/', path)[-1])[-2]
            self.wavelengths = loaded_data[:, 0]
            self._active_key = self.abs_fname
            self.abs = loaded_data[:, 1]
            self.abs_spectra[self.abs_fname] = np.array(
                [loaded_data[:, 0], loaded_data[:, 1]])
            return True

    def filter_abs_spectrum(self, mode='gauss', parameters=None,
                            case='active'):
        if parameters is None:
            parameters = {}
        std_para = {"savgol": {"order": 5, "framelength": 3},
                    "gauss": {"sigma": 1, "order": 0}}
        func = {"savgol": savgol_filter_wrap, "gauss": gaussian_filter}
        for key, val in std_para[mode].items():
            if key not in parameters.keys():
                parameters[key] = val
        keys = self._get_abs_keys(case)
        for key in keys:
            self.abs_nonfilt[key] = np.array(
                [a for a in self.abs_spectra[key][1, :]])
            self.abs_spectra[key][1, :] = func[mode](
                self.abs_spectra[key][1, :], **parameters)

    def clear_filter(self):
        for key, val in self.abs_nonfilt.items():
            try:
                self.abs_spectra[key][1, :] = val
            except KeyError:
                pass
            except Exception:
                raise

    def _get_abs_keys(self, case):
        if re.search('active', case, re.I):
            return [self._active_key]
        elif re.search('all', case, re.I):
            return self.abs_spectra.keys()
        else:
            return [case]

    def magnify_abs(self, factor=1, case='active'):
        if factor == 0:
            return
        keys = self._get_abs_keys(case)
        for key in keys:
            if key in self.magnif_factors.keys():
                fac = factor/self.magnif_factors[key]
            else:
                fac = factor
            self.magnif_factors[key] = factor
            self.abs_spectra[key][1, :] = fac*self.abs_spectra[key][1, :]

    def reset_magnif(self):
        for key, fac in self.magnif_factors.items():
            try:
                self.abs_spectra[key][1, :] = self.abs_spectra[key][1, :]/fac
            except Exception:
                continue
            else:
                self.magnif_factors[key] = 1

    def load_ref_spec(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.ref = loaded_data[:, 1]

    def load_bkg_spectrum(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.bkg = loaded_data[:, 1]

    def load_transmiss(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.trans = loaded_data[:, 1]
            self.trans_fname = re.split('\/', path)[-1]

    def load_spec_txt(self, path, load_all=True):
        m = []
        with open(path) as f:
            while True:
                line = f.readline()
                if not line.startswith('#'):
                    break
            while line:
                m.append(line)
                line = f.readline()
            f.close()
        m = np.array([x.strip().split("\t") for x in m])

        try:
            return m.astype(np.float64)
        except Exception:
            try:
                m = np.array([[x.replace('n. def.', 'nan')
                             for x in i] for i in m])
                return m.astype(np.float64)
            except Exception:
                try:
                    m = np.array([x[0].split(",") for x in m])
                except Exception:
                    raise
                else:
                    try:
                        return m.astype(np.float64)
                    except Exception:
                        raise

    def save(self, path, case='active'):
        with open(path, 'w+') as f:
            np.savetxt(f, np.transpose(
                self.abs_spectra[self._active_key]), delimiter='\t')
            f.close()

    def sort_spectra(self, mode="name"):
        new_dct = {}
        for attr in ["abs_spectra", "diff_abs"]:
            new_dct[attr] = {}
            dct = getattr(self, attr)
            keys = list(dct.keys())
            if re.search("name|alph", mode, re.I):
                sorted_inds = sorted(range(len(keys)), key=keys.__getitem__)
            elif re.search("val", mode, re.I):
                vals = []
                for key in keys:
                    try:
                        vals.append(np.double(re.findall("[\d\.]+", key)[0]))
                    except Exception:
                        vals.append(np.inf)
                sorted_inds = np.argsort(vals)
            for i in sorted_inds:
                new_dct[attr][keys[i]] = dct[keys[i]]
        for key, attr in new_dct.items():
            setattr(self, key, attr)

    def save_abs_traces(self, path=None, case='active'):
        self._save_traces(path, case, case='abs')

    def _save_traces(self, path, keys, case='abs'):
        traces = TATraceCollection()
        if re.search('diff', case, re.I):
            spectra = self.diff_abs
            ylabel = "$\Delta$ Abs. (OD)"
        else:
            spectra = self.abs_spectra
            ylabel = "Abs. (OD)"
        if not type(keys) is list:
            if re.search('active', keys, re.I):
                keys = [self._active_key]
            elif re.search('all', keys, re.I):
                keys = spectra.keys()
        for key in keys:
            trace = TATrace(xdata=spectra[key][0, :],
                            trace_type='Spectral',
                            yunit='OD', ylabel=ylabel,
                            xlabel='wavelength (nm)', xunit='nm')
            trace.add_trace(spectra[key][1, :], "")

            traces.add_trace(trace, key)
        traces.save(trace_type='Spectral',
                    fname=path)

    def save_diff_abs(self, path, case='active'):
        key = list(self.diff_abs.keys())[-1]
        with open(path, 'w+') as f:
            np.savetxt(f, np.transpose(self.diff_abs[key]), delimiter='\t')
            f.close()

    def save_diff_abs_traces(self, path=None, case='active'):
        self._save_traces(path, case, case='diff')

    def calc_abs_spectrum(self):
        try:
            self.abs = np.log10(
                np.divide((self.ref-self.bkg), (self.trans-self.bkg)))
        except Exception:
            if self.ref is None:
                return "Reference spectrum not loaded.", None
            elif self.bkg is None:
                return "Background spectrum not loaded.", None
            elif self.trans is None:
                return "Transmitted spectrum not loaded.", None
            else:
                return "Error calculating absorption spectrum", None
        else:
            wl = [w for w in self.wavelengths]
            self.abs_spectra[self.trans_fname] = np.array([wl, self.abs])
            self._active_key = self.trans_fname
            return None, self._active_key

    def calc_diff_abs(self, spec_keys, ref_keys):
        ref = self.abs_spectra[ref_keys][1, :]
        for key in spec_keys:
            try:
                self.diff_abs[" - ".join([key, ref_keys])] = np.array(
                    [self.abs_spectra[key][0, :],
                     self.abs_spectra[key][1, :] - ref])
            except KeyError:
                pass
            except Exception:
                raise

    def subtr_baseline(self, val, case='abs', from_all=False):
        if case == 'abs':
            if from_all:
                for spec in self.abs_spectra.values():
                    spec[1, :] = spec[1, :] - val
            else:
                self.abs_spectra[self._active_key][1, :] = (
                    self.abs_spectra[self._active_key][1, :] - val)
        elif case == 'ref':
            self.ref = self.ref - val
        elif case == 'trans':
            self.trans = self.trans - val
        else:
            return
        self.baseline = self.baseline + val

    def get_abs_xind(self, key, xrange, mode='slice'):
        x = self.abs_spectra[key][0, :]
        try:
            xlower = np.where(x >= xrange[0])[0][0]
        except Exception:
            xlower = 0
        try:
            xupper = np.where(x <= xrange[1])[0][-1]
        except Exception:
            xupper = len(x)
        if re.search('slice', mode, re.I):
            return slice(xlower, xupper)
        elif re.search('ind', mode, re.I):
            return [xlower, xupper - 1]

    def calc_centroid(self, xrange=None, spectrum='active'):
        def centroid(tr, x):
            return np.sum(tr * x)/np.sum(tr)

        if re.search('active', spectrum, re.I):
            xrange = self.get_abs_xind(self._active_key, xrange, mode='slice')
            s = self.abs_spectra[self._active_key]
            return centroid(s[1, xrange], s[0, xrange])
        elif re.search('all', spectrum, re.I):
            c = {}
            for key, val in self.abs_spectra.items():
                xrange = self.get_abs_xind(key, xrange, mode='slice')
                c[key] = centroid(val[1, xrange], val[0, xrange])
            return c

    def integrate_spectrum(self, xrange=None, spectrum='current'):
        if re.search('curr', spectrum, re.I):
            xrange = self.get_abs_xind(self._active_key, xrange, mode='slice')
            return (np.sum(self.abs_spectra[self._active_key][1, xrange]),
                    np.mean(self.abs_spectra[self._active_key][1, xrange]))
        elif re.search('all', spectrum, re.I):
            integ = {}
            mean = {}
            for key, val in self.abs_spectra.items():
                xrange = self.get_abs_xind(key, xrange, mode='slice')
                integ[key] = np.sum(val[1, xrange])
                mean[key] = np.mean(val[1, xrange])
            return integ, mean

    def plot_abs_spectra(self, ax, plot_which='All', zero_line=True,
                         **kwargs):
        if plot_which == 'All':
            plot_which = self.abs_spectra.keys()
            for k in plot_which:
                ax.plot(self.abs_spectra[k][0, :],
                        self.abs_spectra[k][1, :], **kwargs)
            if zero_line:
                ax.plot([self.abs_spectra[k][0, 0],
                         self.abs_spectra[k][0, -1]],
                        [0, 0],
                        color='grey', linestyle='-.', dashes=(5, 1),
                        **kwargs)
        else:
            dat = self.abs_spectra[self._active_key]
            ax.plot(dat[0, :], dat[1, :])
            if zero_line:
                ax.plot([dat[0, 0], dat[0, -1]], [0, 0], color='grey',
                        linestyle='-.', dashes=(5, 1), **kwargs)
