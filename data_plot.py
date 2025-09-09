#!/usr/bin/env python3
# This module provides various plotting functions for visualizing coordination metrics.
# Functions
# ---------
# get_cmap(n, name='hsv'):
#     Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color.
# plot_joints_info(coord_metric, mean_data):
#     Create a plot with joints information, showing a figure with 2 subplots: one with the joints values over time and another with angular velocities values over time.
# plot_angle_angle(coord_metric, data_list, mean_data=False, position=True, width=False):
#     Plot Angle/Angle Graphs, optionally showing mean data and width.
# plot_ee_info(list_data, name):
#     Create a plot with end effector information, showing a figure with 2 subplots: one with the end-effector trajectory in the 3D space and another one with the end-effector linear velocity over time.
# plot_crp(angle_name, data, data_norm, max_vel, fig):
#     Plot Continuous Relative Phase (CRP) results.
# plot_corr_matrix(data, title, sym=False, annot=None, vmax=None, vmin=None):
#     Plot a correlation matrix with optional symmetry and annotations.
# plot_atypical_kinematics(data, title):
#     Plot atypical kinematics data in a heatmap format.
# plot_emg_info(coord_metric, mean_data):
#     Plot Electromyography (EMG) information, optionally showing mean data.
# -*- coding: utf-8 -*-

"""
Created on Wed Nov  3 17:32:08 .

@author: oceane
"""
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import numpy as np
import math


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_joints_info(coord_metric, mean_data):
    """
    Create a plot with joints infos.

    Show a figure with 2 subplot : one with the joints values over time
    and another one with angular velocities values over time

    Returns
    -------
    None.

    """

    cmap = get_cmap(coord_metric.get_n_dof()+1)


    data = coord_metric.get_data_ang()
    data_velocity = coord_metric.get_data_vel()

    if not mean_data :
        # Plot angles
        fig = plt.figure()
        fig.suptitle(' Data : ' + coord_metric.get_data_name())
        ax1 = fig.add_subplot(2, 1, 1)
        for d in data:
            for i, n in enumerate(coord_metric.get_n_columns_ang()):
                print(d['time'])
                ax1.plot(d['time']/24, np.rad2deg(d[n].to_numpy()), c=cmap(i))
        ax1.legend(coord_metric.get_n_columns_ang())
        ax1.set_ylabel('Angle (deg)')
          # Plot angular velocities
        ax1 = fig.add_subplot(2, 1, 2)
        for d in data_velocity:
            for i, n in enumerate(coord_metric.get_n_columns_vel()):
                ax1.plot(d['time'], np.rad2deg(d[n].to_numpy()), c=cmap(i))
        ax1.legend(coord_metric.get_n_columns_vel())
        # ax1.set_ylim([-3, 3])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (rad/sec)')
        
    else:
        
        #Plot mean and std movement
        fig = plt.figure()
        fig.suptitle(' Data : ' + coord_metric.get_data_name())
        ax1 = fig.add_subplot(2, 1, 1)
        data = coord_metric.get_concatenated_data(horizontal=True, position=True)
        for i, c in enumerate(coord_metric.get_n_columns_ang()):
            data_mean = data[c].mean(axis=1)
            data_std= data[c].std(axis=1)
            ax1.plot(data_mean['time'], data_mean, c=cmap(i))
            ax1.fill_between(data.index, data_mean-data_std, data_mean+data_std, color=cmap(i), alpha=0.2)
        ax1.set_ylabel('Joint Position (rad)')
        ax1.set_xlabel('Normalized time')
        ax1 = fig.add_subplot(2, 1, 2)
        data_velocity = coord_metric.get_concatenated_data(horizontal=True, position=False)
        for i, c in enumerate(coord_metric.get_n_columns_vel()):
                 data_mean = np.rad2deg(data_velocity[c].mean(axis=1))
                 data_std= np.rad2deg(data_velocity[c].std(axis=1))
                 ax1.plot(data_mean, c=cmap(i))
                 ax1.fill_between(data.index, data_mean-data_std, data_mean+data_std, color=cmap(i), alpha=0.2)
        ax1.legend(coord_metric.get_n_columns_vel())
        # ax1.set_ylim([-3, 3])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (rad/sec)')

  

    plt.show(block=False)


def plot_angle_angle(coord_metric, data_list, mean_data=False, position=True, width=False):
    """
    Plot Angle/Angle Graphs.

    Parameters
    ----------
    metrics : TYPE
        DESCRIPTION.
    position : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """

    if position:
        columns = coord_metric.get_n_columns_ang()

    else:
        columns = coord_metric.get_n_columns_vel()


    df_data = pd.DataFrame()
    t = np.arange(-200, 200, 0.01)
    ax = []
    maximum = []

    # for each dataset
    for i, d in enumerate(data_list):

        # get angle or velocity data
        d_temp = d[columns]
        # if not position get velocity in centimeter per seconds
        if not position:
            d_temp[columns] = 100*d_temp[columns]

        d_temp = d_temp.dropna()
        # set trial number
        d_temp['trial'] = i
        df_data = pd.concat([df_data, d_temp])

    if position:
        df_data[columns] = np.rad2deg(df_data[columns])
    df_data = df_data.reset_index()
    #
    # # do a pair plot
    ax = sns.pairplot(df_data[columns+['trial']].sample(2000), corner=False,
                      # diag_kind='kde',
                      hue='trial')
    ax.fig.suptitle('Angle/Angle Plot' + ' Data : ' +str(coord_metric.get_data_name()) + '\n Position : ' + str(position))
    ax3 = sns.pairplot(df_data[columns].sample(2000), corner=False,
                       diag_kind='kde',
                       kind="reg",
                       plot_kws={'line_kws': {'color': 'black'}})
    ax3.fig.suptitle('Angle/Angle Plot' + ' Data : ' +str(coord_metric.get_data_name()) + '\n Position : ' + str(position))

    df_slope = pd.DataFrame(np.zeros((len(df_data['trial'].unique()), len(coord_metric.get_combination_theta(position)))), columns=[
                            ' '.join(i) for i in coord_metric.get_combination_theta(position)], index=df_data['trial'].unique(), dtype=float)

    df_data = df_data.dropna()
   
    for i, n in enumerate(columns):
        offset = df_data[columns].std().max()
        max_lim = df_data[columns].max().max()+offset
        min_lim = df_data[columns].min().min()-offset
        for aa in [ax, ax3]:
            aa = aa.axes[i, i]
            aa.set_axis_off()
            if width :
                aa.annotate("Width = \n" + str(width_tab[n].iloc[0]),
                                # + '\n Max = \n' + str(maximum_dist),
                                xy=(0.01, 0.9),
                                size=10, xycoords=aa.transAxes)
            aa.set(xlim=(min_lim, max_lim))
            aa.set(ylim=(min_lim, max_lim))

    plt.show(block=False)

    if width :
        return width_tab, df_width_ratio, df_slope
    else :
        return None, None, df_slope


def plot_ee_info(list_data, name):
    """
    Create one plot end effector infos.

    It creates a figure with 2 subplots : one with the end-effector
    trajectory in the 3D space and another one with the end-effector
    linear velocity over time.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(' Data : ' + name)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    for d in list_data:
        ax1.plot3D(d['ee_pos1'],
                   d['ee_pos2'],
                   d['ee_pos3'], c='b')
        ax2.plot(d['time'],
                 d['ee_velocity'], '.')

    ax1.set_title('End effector trajectory \n \n')
    ax1.set_xlabel('Position  x (m)')
    ax1.set_ylabel('Position  y (m)')
    ax1.set_zlabel('Position  z (m)')
    ax1.view_init(0, 0)

    ax2.set_title('End effector velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    plt.show(block =False)


def plot_crp(angle_name, data, data_norm, max_vel, fig):
    """
    Plot CRP results.

    Parameters
    ----------
    metrics : Metrics
        Class that handels all the variables we need to plot.

    Returns
    -------
    None.

    """

    combinations_theta = list(
        itertools.combinations(angle_name, 2))


    ax = fig.get_axes()
    print(angle_name)
    ax[0].plot(data.index,
               np.rad2deg(data[angle_name]))
    ax[0].legend(angle_name)
    ax[0].set_ylabel("Joint's angle (deg)")
    ax[0].set_xlabel("Normalized time")


    for i, (item) in enumerate(combinations_theta):
        ax[i+1].plot(data_norm.index,  # [1:],
                     data_norm['rel_phase_' + item[0]+'_' + item[1]])  # .iloc[1:])

        ax[i+1].set_ylabel('CRP : ' + item[0]+' / ' + item[1])
        ax[i+1].set_ylim([-500, 500])
        ax[i+1].axvline(x=max_vel, color='b', label='max_ee_vel')

      
    ax[i+1].set_xlabel('Time')
    ax[i+1].legend('')



    plt.show(block=False)

    return ax[i+1]


def plot_corr_matrix(data, title, sym=False, annot=None, vmax=None, vmin=None):
    """
    Plots a correlation matrix using seaborn's heatmap.
    Parameters:
    data (pd.DataFrame or np.ndarray): The correlation matrix data to be plotted.
    title (str): The title of the plot.
    sym (bool, optional): If True, masks the upper triangle of the matrix to maintain symmetry. Default is False.
    annot (pd.DataFrame or None, optional): If provided, annotations for each cell in the heatmap. Default is None.
    vmax (float or None, optional): Maximum value for the heatmap color scale. Default is None.
    vmin (float or None, optional): Minimum value for the heatmap color scale. Default is None.
    Returns:
    None
    """

    fig = plt.figure()
    mask = None
    if sym:
        mask = np.triu(np.array(data), k=1)
    if isinstance(annot, pd.DataFrame):
        # print(data)
        # print(annot)
        sns.heatmap(data, mask=mask, annot=annot.values,
                    fmt="", vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(data, mask=mask, annot=True, vmin=0, vmax=1)

    fig.suptitle(title)
    plt.show(block=False)


def plot_atypical_kinematics(data, title):
    """
    Plots atypical kinematics data using heatmaps.
    Parameters:
    data (list of tuples): A list where each tuple contains a title (str) and a 2D numpy array.
                           The 2D numpy array should be transposed before plotting.
    title (str): The main title for the entire figure.
    The function creates a figure with subplots arranged in a grid. Each subplot is a heatmap
    representing the kinematics data for a specific degree of freedom (DOF) over normalized time.
    The heatmaps use the "Blues" colormap, with values ranging from 0 to 1.
    The function also adjusts the font sizes for various plot elements to ensure readability.
    """

    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)     # fontsize of the axes title

    plt.rc('xtick', labelsize=8)     # fontsize of the axes title
    plt.rc('ytick', labelsize=8)     # fontsize of the axes title

    num_fig = len(data)
    num_row = math.ceil(num_fig / 2)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Atypical kinematics \n' + title)
    for i, d in enumerate(data):
        ax = fig.add_subplot(num_row, 2, i+1)
        d_data = d[1].transpose()
        ax.set_title(d[0])
        ax = sns.heatmap(d_data, cmap="Blues", vmax=1,
                         vmin=0)
        #  vmin=0, vmax=100, ax=ax)
        ax.set_ylabel('DOF')
        ax.set_xlabel('Time (normalized)')
    plt.tight_layout()

def plot_emg_info(coord_metric, mean_data): 
    """
    Plots EMG information from the given coordination metric object.
    Parameters:
    coord_metric (CoordinationMetric): An object containing coordination metrics and data.
    mean_data (bool): If True, plots the mean and standard deviation of the EMG data. 
                      If False, plots the raw EMG data.
    Returns:
    None
    """

    cmap = get_cmap(len(coord_metric.get_n_columns_emg())+1)
    data = coord_metric.get_data_emg()


    if not mean_data :
            # Plot angles
            fig = plt.figure()
            fig.suptitle(' Data : ' + coord_metric.get_data_name())
            ax1 = fig.add_subplot(1, 1, 1)
            for d in data:
                for i, n in enumerate(coord_metric.get_n_columns_emg()):
                    ax1.plot(d['Time,s'], d[n], c=cmap(i))
            ax1.legend(coord_metric.get_n_columns_emg())
            ax1.set_ylabel('EMG [%MVC]')
    else:
        
        #Plot mean and std movement
        fig = plt.figure()
        fig.suptitle(' Data : ' + coord_metric.get_data_name())
        ax1 = fig.add_subplot(1, 1, 1)
        data = coord_metric.get_concatenated_data(horizontal=True, position=True, emg=True)
        for i, c in enumerate(coord_metric.get_n_columns_emg()):
            data_mean = data[c].mean(axis=1)
            data_std= data[c].std(axis=1)
            ax1.plot(data_mean, c=cmap(i), label=c)
            ax1.fill_between(data.index, data_mean-data_std, data_mean+data_std, color=cmap(i), alpha=0.2)
        ax1.set_ylabel('Joint Position (rad)')           
        ax1.hlines(0.3, 0,100, color='k', label='Threshold')
        ax1.legend()

        ax1.set_xlabel('Normalized time')
    
    plt.show(block=False)
    return 





   
        
    