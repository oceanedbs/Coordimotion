#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:24:24 2022

@author: dubois
"""
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, filtfilt
#from dtw import *

def remove_file(path):
     try : 
        os.remove(path)
     except OSError:
        pass
    
def append_data(path, data):
    if os.path.isfile(path):
        data.to_csv(path, mode='a', header=False)
    else:
        data.to_csv(path)

def extract_data_name(filename):
    """
    Extract name of the data from the filename.

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    text : TYPE
        DESCRIPTION.

    """
    text = filename.rsplit("/", 1)[-1]
    text = text.rsplit(".", 1)[0]
    return text


def butter_lowpass_filter(data, cutoff, fs, order):
    for c in data.columns:
        if len(data[c]) > 18:
            nyq = 0.5 * fs  # Nyquist Frequency
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            data[c] = filtfilt(b, a, data[c])
    return data


def zero_crossing(data):
    return np.where(np.diff(np.sign(np.array(data))))[0]

def time_wrap_data(data, mean):
    alignment = dtw(data, mean, keep_internals=True)
    return
def normalize_data_in_time(data_m, step=0.1, dropna=False):
    data = data_m.copy()
    # data.index = data['time']
    data.index = data.index - data.index[0]
    data.index = data.index/data.index.max()*100
    data['time'] = data.index
    data.index.name = 'Index'
    df_new_time = pd.DataFrame(np.arange(0, 100+step, step), columns=['time'])
    # print(data)
    data = pd.merge(
        data, df_new_time, how="outer", on=["time"])
    data = data.sort_values('time')
    # data = data.fillna(0)
    data = data.interpolate()
    data = data[data['time'].isin(np.arange(
        0, 100+step, step))]

    # data = data.drop_duplicates(subset='time')
    data.index = data['time']
    data = data.drop(columns=['time'])
    if dropna:
        data = data.dropna()
    return data


def normalize_data_in_amplitude(data):
    data = data-data.min()
    max_data = data.max().max()
    data = data/max_data
    data = data-data.mean()

    return data


def save_multi_image(filename, path, svg=False):
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    if svg :
        for i, fig in enumerate(figs):
            fig.savefig(path+filename+ '_'+ str(i)+'.svg', format='svg')

    pdf = matplotlib.backends.backend_pdf.PdfPages(
        path + filename + ".pdf")
    for fig in plt.get_fignums():
        pdf.savefig(fig)
    pdf.close()


def able_jacobian(q, list_angle, l1, l2):

    # ABLE Jacobian
    # This function computes the jacobian of ABLE, given a specific joint
    # configuration q=[q1[n],q2[n],q3[n],q4[n]] and the end-effector/elbow distance l2 in mm and l1 the shoulder/elbow distance.

    q1 = q[list_angle[0]]
    q2 = q[list_angle[1]]
    q3 = q[list_angle[2]]
    q4 = q[list_angle[3]]
    print(q1)

    J = np.zeros((3, 4, len(q)))

    for n in np.arange(len(q)):
        J[:, :, n] = np.array([
            # first line
            [- (l1*np.cos(q1[n])*np.sin(q3[n]))/1000 - (l2*np.cos(q4[n])*(np.cos(q1[n])*np.sin(q3[n]) + np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))) - (l2*np.sin(q4[n])*(np.cos(q1[n])*np.cos(q3[n]) - np.cos(q2[n])*np.sin(q1[n])*np.sin(q3[n]))) - (l1*np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))/1000,

             np.cos(q1[n])*((l1*np.cos(q3[n])*np.sin(q2[n]))/1000 - (l2*np.sin(q2[n]) *
                                                                     np.sin(q3[n])*np.sin(q4[n])) + (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n]))),

             - np.cos(q2[n])*((l1*np.cos(q1[n])*np.sin(q3[n]))/1000 + (l2*np.cos(q4[n])*(np.cos(q1[n])*np.sin(q3[n]) + np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))) + (l2*np.sin(q4[n])*(np.cos(q1[n])*np.cos(q3[n]) - np.cos(q2[n])*np.sin(q1[n])*np.sin(
                 q3[n]))) + (l1*np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))/1000) - np.sin(q1[n])*np.sin(q2[n])*((l1*np.cos(q3[n])*np.sin(q2[n]))/1000 - (l2*np.sin(q2[n])*np.sin(q3[n])*np.sin(q4[n])) + (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n]))),

             np.sin(q1[n])*np.sin(q2[n])*((l2*np.sin(q2[n])*np.sin(q3[n])*np.sin(q4[n])) - (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n]))) - np.cos(q2[n])*((l2*np.cos(q4[n])*(np.cos(q1[n])*np.sin(q3[n]) + np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))) + (l2*np.sin(q4[n])*(np.cos(q1[n])*np.cos(q3[n]) - np.cos(q2[n])*np.sin(q1[n])*np.sin(q3[n]))))],

            # second line
            [0,

             np.sin(q1[n]) * ((l1*np.cos(q1[n])*np.sin(q3[n]))/1000 + (l2*np.cos(q4[n])*(np.cos(q1[n])*np.sin(q3[n]) + np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))) + (l2*np.sin(q4[n])*(np.cos(q1[n])*np.cos(q3[n]) - np.cos(q2[n])*np.sin(q1[n])*np.sin(q3[n]))) + (l1*np.cos(q2[n])*np.cos(q3[n])*np.sin(q1[n]))/1000) -
                np.cos(q1[n]) * ((l1*np.sin(q1[n])*np.sin(q3[n]))/1000 + (l2*np.cos(q4[n])*(np.sin(q1[n])*np.sin(q3[n]) - np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))) +
                                 (l2*np.sin(q4[n])*(np.cos(q3[n])*np.sin(q1[n]) + np.cos(q1[n])*np.cos(q2[n])*np.sin(q3[n]))) - (l1*np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))/1000),

                np.cos(q1[n]) * np.sin(q2[n])*((l1 * np.cos(q1[n]) * np.sin(q3[n]))/1000 + (l2 * np.cos(q4[n])*(np.cos(q1[n]) * np.sin(q3[n]) + np.cos(q2[n]) * np.cos(q3[n]) * np.sin(q1[n]))) + (l2 * np.sin(q4[n])*(np.cos(q1[n]) * np.cos(q3[n]) - np.cos(q2[n]) * np.sin(q1[n]) * np.sin(q3[n]))) + (l1 * np.cos(q2[n])
                                                                                                                                                                                                                                                                                                          * np.cos(q3[n]) * np.sin(q1[n]))/1000) + np.sin(q1[n]) * np.sin(q2[n])*((l1 * np.sin(q1[n]) * np.sin(q3[n]))/1000 + (l2 * np.cos(q4[n])*(np.sin(q1[n]) * np.sin(q3[n]) - np.cos(q1[n]) * np.cos(q2[n]) * np.cos(q3[n]))) + (l2 * np.sin(q4[n])*(np.cos(q3[n]) * np.sin(q1[n]) + np.cos(q1[n]) * np.cos(q2[n]) * np.sin(q3[n]))) - (l1 * np.cos(q1[n]) * np.cos(q2[n]) * np.cos(q3[n]))/1000),

                np.cos(q1[n]) * np.sin(q2[n])*((l2 * np.cos(q4[n])*(np.cos(q1[n]) * np.sin(q3[n]) + np.cos(q2[n]) * np.cos(q3[n]) * np.sin(q1[n]))) + (l2 * np.sin(q4[n])*(np.cos(q1[n]) * np.cos(q3[n]) - np.cos(q2[n]) * np.sin(q1[n]) * np.sin(q3[n])))) + np.sin(q1[n]) * np.sin(q2[n])*((l2 * np.cos(q4[n])*(np.sin(q1[n]) * np.sin(q3[n]) - np.cos(q1[n]) * np.cos(q2[n]) * np.cos(q3[n]))) + (l2 * np.sin(q4[n])*(np.cos(q3[n]) * np.sin(q1[n]) + np.cos(q1[n]) * np.cos(q2[n]) * np.sin(q3[n]))))],

            # third line
            [(l1*np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))/1000 - (l2*np.cos(q4[n])*(np.sin(q1[n])*np.sin(q3[n]) - np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))) - (l2*np.sin(q4[n])*(np.cos(q3[n])*np.sin(q1[n]) + np.cos(q1[n])*np.cos(q2[n])*np.sin(q3[n]))) - (l1*np.sin(q1[n])*np.sin(q3[n]))/1000,

             np.sin(q1[n])*((l1*np.cos(q3[n])*np.sin(q2[n]))/1000 - (l2*np.sin(q2[n]) *
                                                                     np.sin(q3[n])*np.sin(q4[n])) + (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n]))),

             np.cos(q1[n])*np.sin(q2[n])*((l1*np.cos(q3[n])*np.sin(q2[n]))/1000 - (l2*np.sin(q2[n])*np.sin(q3[n])*np.sin(q4[n])) + (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n]))) - np.cos(q2[n])*((l1*np.sin(q1[n])*np.sin(q3[n]))/1000
                                                                                                                                                                                                    + (l2*np.cos(q4[n])*(np.sin(q1[n])*np.sin(q3[n]) - np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))) + (l2*np.sin(q4[n])*(np.cos(q3[n])*np.sin(q1[n]) + np.cos(q1[n])*np.cos(q2[n])*np.sin(q3[n]))) - (l1*np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))/1000),

             - np.cos(q2[n])*((l2*np.cos(q4[n])*(np.sin(q1[n])*np.sin(q3[n]) - np.cos(q1[n])*np.cos(q2[n])*np.cos(q3[n]))) + (l2*np.sin(q4[n])*(np.cos(q3[n])*np.sin(q1[n]) + np.cos(q1[n])*np.cos(q2[n])*np.sin(q3[n])))) - np.cos(q1[n])*np.sin(q2[n])*((l2*np.sin(q2[n])*np.sin(q3[n])*np.sin(q4[n])) - (l2*np.cos(q3[n])*np.cos(q4[n])*np.sin(q2[n])))]])

    return J


def recompute_xd_from_J(J, qd):
    Xd = np.zeros((J.shape[2], 3))
    for n in np.arange(J.shape[2]):
        Xd[n, :] = np.matmul(J[:, :, n],  qd.loc[n, :])
    print(Xd)
    return Xd


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))