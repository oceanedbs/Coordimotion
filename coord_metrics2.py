# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 27 11:30:12 2022

# @author: dubois
# """
import itertools
import math
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_plot as dp
import pca
import utils as ut
from scipy_func import argrelextrema
import random
import crp as crp
import seaborn as sns
import copy
from scipy_func import distance_correlation


class CoordinationMetric2():
    """ Compute and manage coordination metrics of a dataset."""

    ee_columns = ['ee_pos1', 'ee_pos2', 'ee_pos3']

    def __init__(self, n_dof, list_files_angles=None, list_data=None, list_files_joints=None, name=None, list_angles=None, deg=True, freq=None):
        """
        Initialize a CoordinationMetric Object.

        Parameters
        ----------
        n_dof : int
            numbers of degrees of freedom.
        list_files_angles : list
            list of the data filenames.
        list_files_joints : list, optional
            list of the intermediate joint's data filenames. The default is None.
        name : string, optional
            Name of the dataset. The default is None.
        list_angles : list, optional
            List of the joints name in the files. If nothing is specified, it will be assumed
            that the joints angles are named '1', '2', '3' ...
        deg : bool, optional
            True if the files contain the joints angles in degrees, False if angles
            are in radians.
        freq : int, optional
            sampling frequency of the data.

        Raises
        ------
        ValueError
            ValueError is raised if the file format is incorrect.

        Returns
        -------
        None.
        """        

        # init fields
        self.list_files_angles = list_files_angles
        self.list_files_joints = list_files_joints
        if list_data: 
            self.list_data_angles = list_data
        else: 
            self.list_data_angles = []

        self.list_data_emg = []

        self.n_dof = n_dof
        self.deg = deg
        self.freq = freq

        if list_angles and (len(list_angles) != n_dof):
            raise ValueError(
                "There should be as many angles names as n_dof")

        # set name
        if name is not None:
            self.name = name
        # if no name is given, use the filename
        else:
            self.name = self.__set_data_name()

        # initialize empty members
        # list of data frames
        self.list_data_joints = []

        self.n_columns_ang = []
        self.n_columns_vel = []
        self.n_columns_joints = []

        self.mean_movement_ang = pd.DataFrame(
            columns=self.n_columns_ang + self.n_columns_vel)
        self.mean_movement_joints = pd.DataFrame()
        self.joints_metrics = pd.DataFrame()

        # set the columns name
        if not list_angles:
            self.__create_n_columns()
        # if columns names are given, create corresponding velocity columns
        else:
            self.n_columns_ang = list_angles
            self.__create_n_vel_columns()

        # load angles
        if self.list_files_angles:
            if self.list_files_angles[0].endswith(".csv"):
                self.__load_csv_ang_file()
            else:
                raise ValueError(
                    "file format not known. Should be a 'csv' file")

     
        self.theta_combination = list(
            itertools.combinations(self.n_columns_ang, 2))

        # compute angular velocity
        self.compute_angular_velocity(freq=self.freq)

        # compute mean trajectory
        self.compute_mean_trajectory()



        self.theta_vel_combination = list(
            itertools.combinations(self.n_columns_vel, 2)) 
        
        

# %% Load data

    def __load_csv_ang_file(self):
        """
        Load data from a csv file.

        Data should be arranged as follows:
            [Time, Theta1 ...n, [ee_pos1, ee_pos2, ee_pos3]]
        The first column is the Time. The following ones contain the
        joints values. There is as many columns as angles. The last 3 columns
        are optional and contain the end-effector position.

        Returns
        -------
        data : pd.DataFrame
            DataFrame containing the time, the angles, and the end-effector
            position if this info was provided in the source file.

        """
        target_number = []
        target_height = []

        # for each csv file, read data as dataframe
        for files in self.list_files_angles:
            data_ang = pd.read_csv(files,  # index_col=0,
                                   header=[0])
            data_ang = data_ang.rename(columns={"tgt": "target",
                                                "intermediate_score": "interm_score"})

            # remove toolbox
            # get order and target height
            if 'score' in data_ang.columns:
                target_number.append(int(data_ang['score'].head(1).values[0]))
                target_height.append(int(data_ang['target'].head(1).values[0]))

            # if data are in degrees, convert them to radians
            if self.deg:
                data_ang[self.n_columns_ang] = np.deg2rad(
                    data_ang[self.n_columns_ang])

            if 'time' in data_ang.columns:
                t = 'time'
            else:
                t = 'index'

            # reset time so each movement starts at 0
            print(data_ang[t])
            data_ang[t] = data_ang[t] - \
                data_ang[t].head(1).to_numpy()[0]

            # set time as index
            data_ang.index = data_ang[t]

            # add data to the list of data
            self.list_data_angles.append(data_ang)

        self.joints_metrics['Condition'] = self.name

        return self.list_data_angles

    

    # %% Compute velocity

    def compute_angular_velocity(self, freq=None):
        """
        Compute angular velocity for each joints.

        Create a list named vel_columns with all the columns name that contains
        joint velocity.
        Updates the dataframe containing all data with two new column for each
        joint.The first new column contains diff() for each joint.
        The second columns contains the joint's velocity.

        Returns
        -------
        None.

        """

        # create columns name
        self.n_columns_vel = []
        n_delta_ang = []
        for col_name in self.n_columns_ang:
            name_speed = "vel_" + col_name
            self.n_columns_vel.append(name_speed)
            name_delta = "delta_" + col_name
            n_delta_ang.append(name_delta)

        for i, d in enumerate(self.list_data_angles):
            d.reset_index(drop=True, inplace=True)
            d['time'] = d.index
            # resample and interpolate
            time_resample = np.arange(d['time'].head(
                1).values[0], d['time'].tail(1).values[0]+0.01, 0.01)
            d_resample = pd.DataFrame(time_resample, columns=['time'])
            # d.reset_index(drop=True, inplace=True)

            d = pd.merge(d, d_resample, on='time', how='outer')
            d = d.sort_values(by=['time'])
            d = d.interpolate()
            d = d[d['time'].isin(time_resample)]

            if freq is None:
                d['delta_time'] = d['time'].diff()
            else:
                d["delta_time"] = 1/freq
            d.index = d['time']

            d[n_delta_ang] = d[self.n_columns_ang].diff()

            d[self.n_columns_vel] = d[n_delta_ang].div(d["delta_time"], axis=0)

            d[self.n_columns_vel] = ut.butter_lowpass_filter(
                d[self.n_columns_vel].dropna(), 10, 120*100, 5)

            self.list_data_angles[i] = d



        # %% Compute joints metrics

    def compute_joints_metrics(self, mode='optitrack'):
        self.compute_rom()
#        self.compute_elbow_height()

        return self.joints_metrics

    def compute_rom(self):
        for j in self.n_columns_ang:
            self.joints_metrics[j+'_rom'] = 0.0

        for d in self.list_data_angles:
            for j in self.n_columns_ang:
                self.joints_metrics.at[int(d['score'].head(
                    1).values[0]), j+'_rom'] = np.rad2deg(np.rad2deg(d[j].max()-d[j].min()))
        return self.joints_metrics
                
# %% mean trajectory

    def compute_mean_trajectory(self, step=0.1):
        """
        Compute mean data from the list of datas

        Parameters
        ----------
        step : int, optional
            duration of the timestep (0 = start of the movement, 1 = end of the
            movement). The default is 1 (so 100 points for the mean movement).

        Returns
        -------
        self.mean_movement_ang : pd.DataFrame
            DataFrame containing mean movement trajectory.

        """

        # create a big data frame containing all data

        for d in self.list_data_angles:
            # print(d.columns)
            temp_data = ut.normalize_data_in_time(d)
            # print(temp_data)
            temp_data['time_t'] = temp_data.index
            if self.mean_movement_ang.empty:
                self.mean_movement_ang = temp_data
            else:
                self.mean_movement_ang = pd.concat(
                    [self.mean_movement_ang, temp_data])

        # create dataframe to reduce number of data : timestep = step
        df_new_time = pd.DataFrame(
            np.arange(0, 100+step, step), columns=['time_t'])
        self.mean_movement_ang = pd.concat(
            [self.mean_movement_ang, df_new_time])

        # sort and interpolate values
        self.mean_movement_ang = self.mean_movement_ang.sort_values(by=[
            'time_t'])
        self.mean_movement_ang = self.mean_movement_ang.reset_index()

        self.mean_movement_ang.index = self.mean_movement_ang['time_t']
        self.mean_movement_ang = self.mean_movement_ang.groupby(
            self.mean_movement_ang.index).mean()

        # keep only data that has a timestamp that correspond to the wished timestamp
        self.mean_movement_ang = self.mean_movement_ang[self.mean_movement_ang.index.isin(np.arange(
            0, 100+step, step))]

        # reset index and time
        self.mean_movement_ang = self.mean_movement_ang.set_index([
            'time_t'])
        self.mean_movement_ang['time'] = self.mean_movement_ang.index

        return self.mean_movement_ang
    




# %% Plot global information

    def plot_joints_info(self, mean_data=False):
        """
        Plot joint position and velocity in time.

        Returns
        -------
        None.

        """
        dp.plot_joints_info(self, mean_data)

# %% Coordination metrics
    def compute_angle_angle(self, mean_data=False, position=True, width=False):
        """
        Plot the angle/angle diagram.

        Each joint is ploted against another one. On the diagonal,
        the distribution of each angle is ploted.

        The diagonoal uses a kernel density estimation (KDE). A kernel density
        estimate plot is a method for visualizing the distribution of
        observations in a dataset, analagous to a histogram. KDE represents
        the data using a continuous probability density curve in one or
        more dimensions.

        Parameters
        ----------
        mean_data : bool, optional
            If true, the function plots the angle-angle diagram using the mean data.
            The default is False.
        position : bool, optional
            If true, computes the angle-angle diagram with the joints position.
            If false, the angle-angle diagram is computed with the velocity
            data. The default is True.
        Returns
        -------
        None.

        """
        if mean_data:
            data = [self.mean_movement_ang]
        else:
            data = self.list_data_angles

        return dp.plot_angle_angle(
            self, data, mean_data=mean_data, position=position, width=width)

    def compute_pca(self, position=True, plot=True, spca=False,
                    verbose_plot=False, last=False):
        """
         Compute principal components analysis decomposition for the dataset.

        Parameters
        ----------
        plot : bool, optional
            If True, a figure is ploted with the data ploted according to the
            2 last components. The default is True.
        position : bool, optional
            If True the PCA is run with the position data, otherwise the pca is
            computed with the velocity data. The default is True.
        spca : bool, optional
            If True, sparse PCA is run. This PCA doesn't take into account very
            small variance. Otherwise, normal PCA is computed.
            The default is False.
        verbose_plot : bool, optional
            If True, intermediate plot will be displayed. Here the data and the
            explained variance per componants will be shown.
            The default is False.
        last : bool, optional
            If True, the last componants of the PCA is used. Otherwise the
            first components are used.

        Returns
        -------
        pca : object
            PCA result of the dataset.

        """

        return pca.compute_pca(self, plot=plot, position=position,
                               spca=spca, verbose_plot=verbose_plot,
                               last=last)

    def compute_crp(self, anim=False, mean=False, plot=True, verbose_plot=False, split = 1, emg=False):
        """
        Compute Continuous Relative Phase.

        Parameters
        ----------
        anim : bool, optional
            If True a video is displayed to show each step of the CRP
            computation.
            The default is False.
        plot : bool, optional
            If True, the CRP plot will be displayed. The default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame with all the CRP value for each joints pair.

        """
        if 'ee_velocity' in self.mean_movement_ang.columns:
            max_ee_vel = self.mean_movement_ang['ee_velocity'].idxmax()
        else:
            max_ee_vel = 50 
        if math.isnan(max_ee_vel) :
            max_ee_vel = 50

        if emg:
            return crp.compute_crp(self.get_data_emg(), self.n_columns_emg, self.n_columns_emg_vel, 100, self.emg_combination, str(self.get_data_name()), anim=False, mean=False, plot=False, verbose_plot=False, split=1)
        
        if mean:
            return crp.compute_crp([self.get_mean_data(velocity=True)], self.n_columns_ang, self.n_columns_vel, max_ee_vel,
                                   self.theta_combination,  str(self.get_data_name()), anim=anim, mean=mean, plot=plot, verbose_plot=verbose_plot)

        else:

            return crp.compute_crp(self.list_data_angles, self.n_columns_ang, self.n_columns_vel, max_ee_vel,
                                   self.theta_combination,  str(self.get_data_name()), anim=anim, mean=mean, plot=plot, verbose_plot=verbose_plot, split=split)

    
    def compute_crp_diff(self, coord_metric= None, n=2, verbose_plot=False, emg=False):
        """
        Compute the difference between Continuous Relative Phase signals (CRP) of coordination metrics.

        Parameters:
        coord_metric (object, optional): Another instance of the coordination metric to compare against. 
                                             If None, the function will compute CRP differences within the same instance.
        n (int, optional): Number of splits for internal CRP computation. Default is 2.
        verbose_plot (bool, optional): If True, enables verbose plotting. Default is False.
        emg (bool, optional): If True, computes CRP difference for EMG data. Default is False.

        Returns:
        list: A list of CRP differences. If coord_metric is provided, the list contains a single CRP difference.
          Otherwise, it contains CRP differences for all combinations of splits.
        """
        if coord_metric:
            res1 = self.compute_crp(plot=False, verbose_plot=verbose_plot, emg=emg)
            print(res1[0][1])
            res2 = coord_metric.compute_crp(plot=False, verbose_plot=verbose_plot, emg=emg)
            if emg:
                res = crp.crp_difference(res1[0][1], res2[0][1], self.n_columns_emg, self.get_data_name(), coord_metric.get_data_name())
            else:
                res = crp.crp_difference(res1[0][1], res2[0][1], self.get_n_columns_ang(), self.get_data_name(), coord_metric.get_data_name())
        else :
            result = self.compute_crp(plot=False, verbose_plot=verbose_plot, split = n)
            res = []
            for j in itertools.combinations(result,2):
                base1 = j[0][1]
                base2 = j[1][1]
                temp, _ = crp.crp_difference(base1, base2, self.get_n_columns_ang(), self.get_data_name(), self.get_data_name())
                res.append(temp)

        return res

        
   
   
    def compute_matrix_correlation(self, method='pearson',  position=True,
                                   plot=True):
        """
        Compute the correlation matrix.

        Parameters
        ----------
        method : string, optional
            The correlation method to use. The default is 'pearson'.
        plot : bool, optional
            If True, a plot with the correlation matrix will be displayed.
            The default is True.
        position : bool, optional
            If True, the correlation matrix is computed with the joints
            position. Otherwise, the calculation is done with the velocity
            data.
            The default is True.
pca
        Returns
        -------
        corr : pandas.DataFrame
            DataFrame containing the corerlation matrix.

        """
        # data, columns = self.__get_data_columns(position, mean_data=True)
        # data = data[0].dropna()

        # corr = data.corr(method=method)
        # annotation = pd.DataFrame(columns=columns, index=columns)
        # for col in list(product(columns, columns)):
        #     if method == 'pearson':
        #         r, p = stats.pearsonr(data[col[0]], data[col[1]])

        #     else:
        #         r, p = stats.kendalltau(data[col[0]], data[col[1]])

        #     annotation.at[col[0], col[1]] = str(
        #         round(r, 4)) + '\n' + str(round(p, 4))

        # if plot:
        #     dp.plot_corr_matrix(corr, 'Correlation matrix - ' +
        #                         method + '  \n ' + self.get_data_name() +
        #                         '\n Position : ' + str(position), sym=(True),
        #                         annot=annotation, vmax=1, vmin=-1)
        return None #corr

    def compute_mic(self, plot=True, position=True):
        """
        Compute the Maximum Index Coefficient Matrix for the dataset.

        Parameters
        ----------
        plot : bool, optional
            If True, a plot with the maximum index coefficient matrix will
            be displayed.
            The default is True.
        position : bool, optional
            If True, the correlation matrix is computed with the joints
            position. Otherwise, the calculation is done with the velocity
            data.
            The default is True.

        Returns
        -------
        mic_matrix : pandas.DataFrame
            Maximum Index Coefficient Matrix.

        """
        data, columns = self.__get_data_columns(position, mean_data=True)
        for d in data:
            mic_matrix = pd.DataFrame(
                np.zeros((len(columns), len(columns))), columns=columns,
                index=columns)
            mine = MINE(alpha=0.6, c=15)
            for (i, j) in itertools.product(columns, columns):
                mine.compute_score(d[i], d[j])
                mic_matrix.loc[i, j] = mine.mic()
            if plot:
                dp.plot_corr_matrix(
                    mic_matrix, 'Maximum Index Coefficient Matrix \n ' +
                    self.get_data_name() + '\n Position :' + str(position),
                    sym=True, vmax=1)
        return mic_matrix

    def compute_distance_correlation(self, plot=True, position=True):
        """
        Compute the Distance Correlation Matrix.

        Parameters
        ----------
        plot : bool, optional
            If True, a plot with the maximum distance correlation matrix will
            be displayed. The default is True.
        position : bool, optional
            If True, the distance correlation matrix is computed with the
            joints position. Otherwise, the calculation is done with the velocity data.
            The default is True.

        Returns
        -------
        dist_corr_matrix : pandas.DataFrame
            DataFrame containing the distance correlation matrix.

        """
        data, columns = self.__get_data_columns(position, mean_data=True)
        for d in data:
            d = d.dropna()

            dist_corr_matrix = pd.DataFrame(
                columns=columns, index=columns, dtype='float64')
            annot = pd.DataFrame(columns=columns,
                                 index=columns, dtype='float64')

            for c in dist_corr_matrix.columns:
                for r in dist_corr_matrix.index:
                    dcor, pval = distance_correlation(
                        d[r], d[c])
                    annot.loc[r, c] = str(round(dcor, 3)) + \
                        '\n' + str(round(pval, 3))
                    dist_corr_matrix.loc[r, c] = dcor

        if plot:
            dp.plot_corr_matrix(dist_corr_matrix,
                                'Distance Correlation Matrix \n '
                                + self.get_data_name() + '\n Position : '
                                + str(position), sym=True, annot=annot, vmax=1)
        return dist_corr_matrix

    def compute_cross_correlation(self, plot=True, position=True):
        """
        Compute the Cross Correlation Matrix.

        The number displayed in the matrix correspond to the maximum of
        each cross-correlation computation.

        Parameters
        ----------
        plot : bool, optional
            If True, a plot with the cross correlation matrix will be displayed.
            The default is True.
        position : bool, optional
            If True, the distance correlation matrix is computed with the
            joints position. Otherwise, the calculation is done with the
            velocity data.
            The default is True.

        Returns
        -------
        cross_correlation_matrix : pd.DataFrame
            DataFrame containing the cross-correlation matrix.

        """
        data, columns = self.__get_data_columns(position, mean_data=(True))
        for d in data:

            d = d.dropna()

            cross_correlation_matrix = pd.DataFrame(
                np.zeros((len(columns), len(columns))), columns=columns,
                index=columns)
            cross_correlation_matrix_index = pd.DataFrame(
                np.zeros((len(columns), len(columns))), columns=columns,
                index=columns)

            for c in cross_correlation_matrix.columns:
                for r in cross_correlation_matrix.index:
                    test_result = np.correlate(
                        d[r].values, d[c].values)
                    # cross_corr = max(abs(test_result))
                    # cross_corr_index = np.argmax(test_result)
                    cross_corr = test_result[0]
                    cross_corr_index = 0
                    cross_correlation_matrix.loc[r, c] = cross_corr
                    cross_correlation_matrix_index.loc[r, c] = str(
                        round(cross_corr, 3)) + ' \n ' + str(cross_corr_index)
            cross_correlation_matrix.columns = [
                var + "_x" for var in columns]
            cross_correlation_matrix.index = [
                var + "_y" for var in columns]
            if plot:
                # print(self.cross_correlation_matrix_index)
                dp.plot_corr_matrix(cross_correlation_matrix,
                                    'Cross Correlation Matrix \n '
                                    + self.get_data_name() + '\n Position : '
                                    + str(position),
                                    annot=cross_correlation_matrix_index,
                                    sym=True, vmax=3)

        return cross_correlation_matrix

       

    def compute_single_acc(self, position=True, plot=True):
        def compute_single_acc(self, position=True, plot=True):
            """
            Computes the ACC metric which is a ratio of the joint angle trajectories.

            Parameters:
            -----------
            position : bool, optional
                If True, use position data; otherwise, use velocity data. Default is True.
            plot : bool, optional
                If True, generate and display plots for the computed metrics. Default is True.

            Returns:
            --------
            None
                The function does not return any value. It computes the ACC and optionally plots them.

            Notes:
            ------
            - The function retrieves mean data and computes the difference between consecutive rows.
            - It generates a list of column combinations and computes the ratio of differences for each combination.
            - If plotting is enabled, it creates subplots for each combination and displays the computed ratios.
            """
        d = self.get_all_mean_data()
        if position:
            columns = self.results_matrix()
        else:
            columns = self.get_n_columns_vel()
        d = d[columns]
        d = d.diff()
        result = pd.DataFrame(
            np.zeros(len(d)))
        if plot:
            fig = plt.figure()
            fig.suptitle(self.get_data_name())

        # get list of combination with position or velocity
        list_tuple = []
        for i in itertools.combinations(columns, 2):
            list_tuple.append(i)

        # compute acc
        for i, tuple_name in enumerate(list_tuple):
            result[tuple_name[0]+'/'+tuple_name[1]
                   ] = d[tuple_name[0]]/d[tuple_name[1]]
            if plot:
                ax = fig.add_subplot(len(list_tuple), 1, i+1)
                ax.plot(result[tuple_name[0]+'/'+tuple_name[1]], '.')
                ax.plot(result[tuple_name[0]+'/'+tuple_name[1]], '-')

                ax.set_ylim([-5, 5])
                ax.set_title(tuple_name[0]+'/'+tuple_name[1])
        # print(result)

    def compute_angle_ratio(self, position=True, plot=True):
        d = self.get_all_mean_data()
        if position:
            columns = self.get_n_columns_ang()
        else:
            columns = self.get_n_columns_vel()
        d = d[columns]
        list_tuple = []
        result = pd.DataFrame(
            np.zeros(len(d)))

        for i in itertools.combinations(columns, 2):
            list_tuple.append(i)

        if plot:
            fig = plt.figure()
            fig.suptitle(self.get_data_name())

        for i, tuple_name in enumerate(list_tuple):
            result[tuple_name[0]+'/'+tuple_name[1]
                   ] = d[tuple_name[1]]/d[tuple_name[0]]
            if plot:
                ax = fig.add_subplot(len(list_tuple), 1, i+1)
                ax.plot(result[tuple_name[0]+'/'+tuple_name[1]], '.')
                ax.plot(result[tuple_name[0]+'/'+tuple_name[1]], '-')

                ax.set_ylim([-10, 10])
                ax.set_title(tuple_name[0]+'/'+tuple_name[1])

    def compute_angle_ratio_at_max_vel(self, plot=True):
        data = self.list_data_angles

        ar_res = pd.DataFrame(columns=self.theta_combination)

        for i, d in enumerate(data):
            d = d.loc[d['ee_velocity'].idxmax()]
            temp_res = []
            for jc in self.theta_combination:
                temp_res.append(d[jc[0]]/d[jc[1]])
            temp_res = pd.Series(np.array(temp_res),
                                 index=self.theta_combination)
            ar_res = ar_res.append(temp_res, ignore_index=True)

        if plot:
            fig = plt.figure()
            fig.suptitle(self.get_data_name())

        for i, tuple_name in enumerate(self.theta_combination):
            if plot:
                ax = fig.add_subplot(1, len(self.theta_combination), i+1)
                ax.boxplot(ar_res[tuple_name])
                ax.set_ylim([ar_res.min().min(), ar_res.max().max()])

                ax.set_title(tuple_name[0]+'/'+tuple_name[1])

    def compute_temporal_coordination(self, plot=True, percent_task=True):
        """
        Compute the temporal delay of activation between each joint

        Parameters
        ----------
        list_data : list of coordination_metrics object
            List of all the coordination metrics object, one for each
            movement/trial.
            The function plots the mean temporal delay with the std bars.
        plot : bool, optional
            Boolean to tell if the plot is shown or not. The default is True.
        percent_task : bool, optional
            If true, the temporal coordination is computed relatively to the
            task achievement.
            The temporal coordination is a percentage of the task achievement.
            The default is True.

        Returns
        -------
        temporal_coordination_list : list of tuples
            list of tuples containing the joint's name and their respective
            delay of activation.

        """

        df_tc = pd.DataFrame(
            columns=['joints', 'temporal_coordination', 'ignore'])
        df_threshold = pd.DataFrame(columns=self.get_n_columns_vel())

        # for each data
        # based on the assumption that the begining of the movement is directly the begining of the dataset
        for d in self.list_data_angles:
            # get data
            # plt.figure()
            # d[self.n_columns_joints].plot()
            # plt.figure()
            # d[self.n_columns_vel].plot()
            data = d[self.n_columns_vel]

            # convert time to task percentage if needed
            if percent_task:
                data.index = data.index/data.index[-1]

            # get max velocity  of each joints and 5% max vel of each joints
            max_vel = data.max()
            # print("max")
            # print(data.max())
            min_vel_mvmt = 5*max_vel/100
            # print("min")
            # print(min_vel_mvmt)
            # get first index where value is greater than min_vel and append it to the dataframe
            for (n_joint, n_vel) in zip(self.n_columns_ang, min_vel_mvmt.index):
                # print(n_joint)
                # print(data[n_vel].max())
                # print(min_vel_mvmt[n_vel])
                # print(data[data[n_vel] > min_vel_mvmt[n_vel]])
                if abs(data[n_vel].max()-data[n_vel].min()) <= 0.005:
                    df_tc = df_tc.append({'joints': n_joint,
                                          'temporal_coordination': 0}, ignore_index=True)
                else:
                    # if one joint's velocity never crosses max vel
                    # if(data[data[n_vel] > min_vel_mvmt[n_vel]].empty)
                    df_tc = df_tc.append({'joints': n_joint,
                                          'temporal_coordination': data[data[n_vel] > min_vel_mvmt[n_vel]].head(1).index.values[0]}, ignore_index=True)
    
        # if plot:
        #     plt.figure(figsize=(8, 8))
        #     bplot = sns.barplot(x='joints', y='temporal_coordination',
        #                         data=df_tc, capsize=.2)
        #     xlabels = ['theta_' + str(x.get_text())
        #                for x in bplot.get_xticklabels()]
        #     bplot.set_xticklabels(xlabels)
          

        #     plt.ylim(0, 1)
        #     plt.title(self.get_data_name()+' \n Temporal coordination')
        #     if percent_task:
        #         plt.ylabel('Percentage of task achievement')
        #     else:
        #         plt.ylabel('Time (s)')

        #     num_plot = len(df_tc['joints'].unique())
        #     fig, ax = plt.subplots(num_plot)
        #     if num_plot == 1:
        #         ax = [ax]
        #     for i, t in enumerate(df_tc['joints'].unique()):
        #         data = df_tc[df_tc['joints'] == t]
        #         if data['temporal_coordination'].any() == 0:
        #             data = data.append({'condition': self.name,
        #                                 'temporal_coordination': 1, 'joints': t},
        #                                ignore_index=True)
        #         # print(data)
        #         sns.histplot(data=data,
        #                      x='temporal_coordination',  ax=ax[i], binwidth=0.01)
        #         ax[i].set_xlabel('Time (s)')
        #         ax[i].set_xlim([0, 1])
        #         ax[i].set_title('Theta ' + t)
        #         # ax[i].set_ylim([0, len(data)])

        #     fig.suptitle('Temporal Coordination : ' + self.name)

        return df_tc

    def compute_interjoint_coupling_interval(self, plot=True):
        res = pd.DataFrame(columns=['joint', 'value', 'condition'])
        # print(res)

        for d in self.list_data_angles:
            # get end movement time = end of end effector
            ee_max_vel = d['ee_velocity'].max()
            ee_begin_vel = 5*ee_max_vel/100
            t_end_ee = d[d['ee_velocity'] >
                         ee_begin_vel]['time'].tail(1).values[0]

            # get end time for each joint
            for j in self.n_columns_vel:
                j_max_vel = d[j].max()
                j_begin_vel = 5*j_max_vel/100
                t_end_j = d[d[j] > j_begin_vel]['time'].tail(1).values[0]

                # compute difference
                res = res.append(
                    {'condition': self.name, 'value': t_end_ee-t_end_j,
                     'joint': j},
                    ignore_index=True)

        if plot:
            # print(res)
            num_plot = len(res['joint'].unique())
            fig, ax = plt.subplots(num_plot)
            if num_plot == 1:
                ax = [ax]
            for i, t in enumerate(res['joint'].unique()):
                data = res[res['joint'] == t]
                # if data['value'].any() == 0:
                #     data = data.append({'condition': self.name,
                #                         'value': 1, 'joint': t},
                #                        ignore_index=True)
                # print(data)
                sns.histplot(data=data,
                             x='value',  ax=ax[i])  # , binwidth=20)
                ax[i].set_xlabel('Time (s)')
                ax[i].set_xlim([res['value'].min(), res['value'].max()])
                ax[i].set_title('Theta ' + t)
                ax[i].set_ylim([0, len(data)])

            fig.suptitle('Interjoint Coupling Interval : ' + self.name)

            fig, ax = plt.subplots(1)
            bplot = sns.barplot(x='joint', y='value',
                                data=res, capsize=.2, ax=ax)
            ax.set_ylim([res['value'].min(), res['value'].max()])
            xlabels = ['theta_' + str(x.get_text())
                       for x in bplot.get_xticklabels()]
            bplot.set_xticklabels(xlabels)
           
            plt.title(self.get_data_name() +
                      ' \n Interjoint Coupling interval')
            plt.ylabel('Time (s)')

        return 0

    def compute_interjoint_coupling_interval_old(self, plot=True):
        """
        Computes interjoint coupling interval

        Parameters
        ----------
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        # create a data frame to store the results
        res = pd.DataFrame(columns=['tuple', 'value', 'condition'])
        # print(self.n_columns_ang)

        for d in self.list_data_angles:
            # print(d.columns)
            d.index = d['time']-d['time'].head(1).values

            for tup_theta in itertools.combinations(self.n_columns_vel, 2):
                # get max velocity index
                d.index = np.round(d.index, 4)

                max_1 = d[tup_theta[0]].idxmax()
                max_2 = d[tup_theta[1]].idxmax()

              
                if not d[tup_theta[0]][max_1:].empty:

                    min_1 = d[tup_theta[0]][max_1:].idxmin()
                else:
                    min_1 = d[tup_theta[0]].index[0]
                if not d[tup_theta[1]][max_1:].empty:
                    min_2 = d[tup_theta[1]][max_2:].idxmin()
                else:
                    min_2 = d[tup_theta[1]].index[0]

                if abs(d[tup_theta[0]].max()-d[tup_theta[0]].min()) <= 0.005:
                   
                    min_1 = d.index[-1]
                    min_1 = 0
                if abs(d[tup_theta[1]].max()-d[tup_theta[1]].min()) <= 0.005:
                    
                    min_2 = d.index[-1]
                    min_2 = 0

                
                res = res.append(
                    {'condition': self.name, 'value': min_1-min_2,
                     'tuple': tup_theta[0]+'/'+tup_theta[1]},
                    ignore_index=True)
        if plot:
            # print(res)
            num_plot = len(res['tuple'].unique())
            fig, ax = plt.subplots(num_plot)
            if num_plot == 1:
                ax = [ax]
            for i, t in enumerate(res['tuple'].unique()):
                data = res[res['tuple'] == t]
                if data['value'].any() == 0:
                    data = data.append({'condition': self.name,
                                        'value': 1, 'tuple': t},
                                       ignore_index=True)
                # print(data)
                sns.histplot(data=data,
                             x='value',  ax=ax[i], binwidth=20)
                ax[i].set_xlabel('Time (s)')
                ax[i].set_xlim([res['value'].min(), res['value'].max()])
                ax[i].set_title('Theta ' + t)
                ax[i].set_ylim([0, len(data)])

            fig.suptitle('Interjoint Coupling Interval : ' + self.name)

            fig, ax = plt.subplots(1)
            bplot = sns.barplot(x='tuple', y='value',
                                data=res, capsize=.2, ax=ax)
            ax.set_ylim([res['value'].min(), res['value'].max()])
            xlabels = ['theta_' + str(x.get_text())
                       for x in bplot.get_xticklabels()]
            bplot.set_xticklabels(xlabels)
           
            plt.title(self.get_data_name()+' \n Interjoint Coupling interval')
            plt.ylabel('Time (s)')

    def compute_relative_joint_angle_corr(self, list_data, plot=True):
        """
        Computes relative joints angle correlation

        Parameters
        ----------
        list_data : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        RJAC : TYPE
            DESCRIPTION.

        """

        rjac_res = pd.DataFrame(columns=['condition', 'rjac', 'segment'])

        for cond in list_data:

            segment_list = []
         
            # for each condition, load intermediate joint list
            if len(cond.get_joint_data()) == 0:
                cond.load_intermdiate_joints()

            # get the list of joints and create corresponding headers
            joints = cond.n_columns_joints
            header = np.empty((2, 3*(len(joints)-1)), dtype='object')
            for i in range(len(joints)-1):
                segment_list.append('segment_'+str(i))
                for k, pos in enumerate(['X', 'Y', 'Z']):
                    header[0, 3*i+k] = segment_list[-1]
                    header[1, 3*i+k] = pos

            # for each dataset in the condition
            for d, j in zip(cond.get_all_data_ang(), cond.get_joint_data()):

                # create a dataframe to compute the segments
                segments = pd.DataFrame(np.zeros(
                    (j.shape[0], (len(joints)-1)*3)), columns=header.transpose())
                segments.columns = pd.MultiIndex.from_tuples(
                    segments.columns, names=['Joints', 'Position'])

                # for each joint for each time step compute corresponding
                # segment vector
                for index, (joint, s) in enumerate(zip(joints, segment_list)):
                    # get next joint
                    if index+1 < len(joints):
                        next_j = str(joints[index+1])

                        # compute X, Y, Z position
                        for pos in ['X', 'Y', 'Z']:
                            segments[s, pos] = (j[next_j] -
                                                j[joint])[pos].tolist()

                # shift all segment data to be able to vectorize the angle
                # calculation
                for s in segment_list:
                    for pos in ['X', 'Y', 'Z']:
                        segments[s+'_s', pos] = segments[s, pos].shift(-1)

                # create a dataframe to store the theta
                theta = pd.DataFrame(
                    np.zeros((segments.shape[0], len(segment_list))),
                    columns=segment_list)

                # compute angle for first segment (that has no proximal segment)
                s0 = segments[segment_list[0]]
                s1 = segments[segment_list[0]+'_s']

                pos = ['X', 'Y', 'Z']

                theta[segment_list[0]] = np.arccos(dot(
                    s0[pos], s1[pos])/(np.linalg.norm(s0[pos], axis=1)*np.linalg.norm(s1[pos], axis=1)))

                # compute other angles (not the first one and stop before the last segment since the last segment is used to compute theta)
                for i, s in enumerate(segment_list[1:]):
                    # get previous segment
                    if i+1 > 0:
                        prev_s = str(segment_list[i])
                        # compute angle

                        a = dot(segments[prev_s+'_s'][pos], segments[s+'_s'][pos])/(
                            np.linalg.norm(segments[prev_s+'_s'][pos], axis=1)*np.linalg.norm(segments[s+'_s'][pos], axis=1))
                        b = dot(segments[prev_s][pos], segments[s][pos]) / (np.linalg.norm(segments[prev_s][pos], axis=1) *
                                                                            np.linalg.norm(segments[s][pos], axis=1))
                        theta[s] = np.arccos(a)-np.arccos(b)

                theta = theta.dropna()
                # theta = ut.normalize_data_in_time(theta)
                # create a matrix to store RJAC
                RJAC = np.zeros(len(segment_list)-1)

                # for each pair of proximal and distal segments
                for n in range(len(segment_list)-1):
                    covMatrix = np.cov(theta[segment_list[n]],
                                       theta[segment_list[n+1]])
                    covMatrix_i = np.cov(
                        theta[segment_list[n]], theta[segment_list[n]])
                    covMatrix_j = np.cov(
                        theta[segment_list[n+1]], theta[segment_list[n+1]])

                    r = covMatrix/np.sqrt(covMatrix_i*covMatrix_j)
                    RJAC[n] = np.linalg.det(r)
                    rjac_res = rjac_res.append({'segment': n, 'rjac': RJAC[n],
                                                'condition': cond.get_data_name()},
                                               ignore_index=True)
        # plot result
        if plot:
            plt.figure(figsize=(8, 10))
            sns.barplot(data=rjac_res,
                        x='condition', y='rjac', hue='segment')
            plt.xticks(rotation=70)
            # labels = itertools.combinations(segment_list, 2)
            # labels = [''.join(i) for i in labels]
            # plt.bar(labels, RJAC)
            plt.title('Relative Joint Angle Correlation')

        return RJAC
# %% comparative metrics

    def compute_acc(self, list_coord_metrics, plot=True):
        # To do : a automatiser pour n articulations
        result = pd.DataFrame(columns=["condition", "ACC"])
        for cond in list_coord_metrics:
            # compute delta
            for d in cond.get_data_ang():
                da = ut.normalize_data_in_time(d)
                df_diff = da.diff()
                df_diff_2 = df_diff.pow(2)
                df_diff['l'] = df_diff_2[cond.get_n_columns_ang()].sum(
                    axis=1).apply(np.sqrt)
                df_diff['time'] = d['time']
                df_diff["cos_theta"] = df_diff[cond.get_n_columns_ang()[0]] / \
                    df_diff['l']
                df_diff["sin_theta"] = df_diff[cond.get_n_columns_ang()[1]] / \
                    df_diff['l']
                mean_res_2 = df_diff.mean()**2
                A = np.sqrt(mean_res_2["cos_theta"]+mean_res_2["sin_theta"])
                dict_res = {"condition": cond.get_data_name(), "ACC": A}
                r = pd.Series(dict_res)
                result = result.append(r, ignore_index=True)
        # print(result)
        plt.figure()
        sns.barplot(data=result, x="condition", y="ACC").set(
            title='Angular Coefficient of Correspondance')

    def compute_pca_distance(self, list_coord_metrics, plot=True,
                             position=True):
        """
        Compute distance between pc's with  Bockemuhl's formula

        Parameters
        ----------
        list_coord_metrics : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is True.
        position : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return pca.compute_pca_distance(self, list_coord_metrics, plot=plot,
                                        position=position)

    def compute_pca_distance_baseline(self, plot=True,
                                      position=True, n=2):
        """
        Compute distance between pc's with  Bockemuhl's formula

        Parameters
        ----------
        list_coord_metrics : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is True.
        position : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return pca.compute_pca_distance_baseline(self, plot=plot,
                                                 position=position, n=n)

    def compute_pca_reprojection(self, list_coord_metrics, position=False):
        """
        Reproject PCA of list_coord_metrics into the base of self

        Parameters
        ----------
        list_coord_metrics : TYPE
            DESCRIPTION.
        position : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        return pca.compute_pca_reprojection(self, list_coord_metrics, position=position)

    def compute_pca_reprojection2(self, coord_metric_2, nPCA=2, position=False, absolute=True, plot=False, emg=False):
        return pca.compute_pca_reprojection2(self, coord_metric_2, position=position, absolute=absolute, plot=plot, emg=emg, nPCA=nPCA)

    def compute_pca_diff_no_reprojection(self, coord_metric_2, nPCA = 2, position=False, absolute=True, plot=False):

        return pca.compute_pca_diff_no_reprojection(self, coord_metric_2, position=position, absolute=absolute, plot=plot, nPCA=nPCA)

    def compute_pca_reprojection_base(self, position=True, n=2):
        return pca.compute_pca_reprojection_noise(self, position, n=n)

    def compute_base_crp(self, n=2):

        matrix_area_res = self.compute_crp_diff(n=n)

        return matrix_area_res

    def compute_fpca(self, position=False):
        return pca.compute_fpca(self, position)

    def compute_nmf(self, position=False):
        return pca.compute_nmf(self, position)

    def compute_zero_crossing_time(self, plot=True):
        """
        Compute zero crossing

        Parameters
        ----------
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        # get velocities data

        result = pd.DataFrame()
        columns = self.n_columns_vel

        for d in self.list_data_angles:
            res = {'condition': self.name}
            data = ut.normalize_data_in_time(d[self.n_columns_vel])
            data['num'] = np.arange(len(data))
            # data.plot()

            # compute 0 crossing time
            for j in columns:
                # if one articulation doesn't move
                # data[j].plot()
                # print(data[j].min())
                if abs(data[j].min()-data[j].max()) <= 0.005:
                    res['joint'] = j
                    res['task_achievement'] = 0
                    result = result.append(res, ignore_index=True)
                else:

                    zero_crossings = ut.zero_crossing(data[j])
                    # get index of the max value
                    idx_max_time = data[j].idxmax()
                    idx_max = data['num'][idx_max_time]

                    # if no zero crossing get the first local minima after max value
                    if zero_crossings.size == 0:
                        # get all local minima
                        zerocrossing = data.iloc[argrelextrema(
                            data[j].values, np.less)]

                        # if no local minima or there is no local minima after max value get the last value
                        if (zerocrossing.empty) or (zerocrossing[zerocrossing["num"] > idx_max].empty):
                            zerocrossing = data["num"].tail(1).values[0]
                        # get the first local minima after max value
                        else:
                            zerocrossing = zerocrossing[zerocrossing["num"] > idx_max]["num"].head(
                                1).values[0]

                    elif len(zero_crossings) >= 1:
                        # get index that is AFTER max

                        zerocrossing = zero_crossings[np.argmax(
                            zero_crossings > idx_max)]

                    # get corresponding normalized time
                    data_zc = data[data['num'] == zerocrossing]
                    res['joint'] = j
                    res['task_achievement'] = data_zc.index.to_numpy()[0]
                    result = result.append(res, ignore_index=True)
        # print(result)

        if plot:
            plt.figure()
            plt.suptitle('Zero Crossing : ' + self.name)
            sns.barplot(data=result, x='condition',
                        y='task_achievement', hue='joint')

        return result

    def compute_angle_ratio(self):
        fig, ax = plt.subplots(len(self.theta_combination), 1)

        for d in self.list_data_angles:
            for i, n in enumerate(self.theta_combination):
                data_normalized_1 = ut.normalize_data_in_time(
                    d[[n[0], 'time']])
                data_normalized_2 = ut.normalize_data_in_time(
                    d[[n[1], 'time']])

                ratio = data_normalized_1[n[0]]/data_normalized_2[n[1]]
                ax[i].plot(ratio)
                ax[i].set_ylabel(n)

# %% Getter

    def get_n_dof(self):
        """
        Get number of dof

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.n_dof

    def get_data_ang(self):
        """
        Get angle's data as a list

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """
        res = []
        for d in self.list_data_angles:
            res.append(d[self.n_columns_ang+['time']])
        return res

    def get_all_data_ang(self):
        """
        Get all datas as a list

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.list_data_angles

    def get_data_vel(self):
        """
        Get data velocity as a list

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """
        res = []
        for d in self.list_data_angles:
            res.append(d[self.n_columns_vel+['time']])
        return res

    def get_mean_data(self, velocity=False):
        """
        Get mean trajectory angles

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if velocity:
            return self.mean_movement_ang[self.n_columns_ang+self.n_columns_vel+['time']]
        else:
            return self.mean_movement_ang[self.n_columns_ang+['time']]

    def get_all_mean_data(self):
        """
        Get all mean trajectory data (mean joints position, mean velocity ....)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.mean_movement_ang

    def get_mean_data_vel(self):
        """
        Get mean data velocity.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.mean_movement_ang[self.n_columns_vel + ['time']]

    def get_data_name(self):
        """
        Get data name

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.name

    def get_n_columns_ang(self):
        """
        Get columns' name

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.n_columns_ang
    
    
    def get_n_columns_vel(self):
        """
        Get name of velocity columns

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.n_columns_vel

    def get_pca(self, position=True):
        """
        Get pca results

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.compute_pca(position=position)

    def get_joint_list(self):
        """
        Get list of joints

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.n_columns_joints

    def get_joint_data(self):
        """
        Get intermediate joints data

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.list_data_joints


    def get_combination_theta(self, position):
        if position:
            return self.theta_combination
        else:
            return self.theta_vel_combination

    def get_concatenated_data(self, position, horizontal = False, normalized=False, emg=False):
        
        if horizontal :
            normalized = True 
            
        if emg:
            columns = self.n_columns_emg

        elif position:
            columns = self.get_n_columns_ang()
        else:
            columns = self.get_n_columns_vel()
        # append all datas to a dataframe
        df_data = pd.DataFrame(columns=columns)

        if emg:
            all_data = self.list_data_emg
        else:
            all_data = self.get_all_data_ang()

        for d in all_data:
            data = d[columns].iloc[1:]
            if normalized:
                data = ut.normalize_data_in_time(data)
                if not horizontal : 
                    df_data = df_data.append(data)
                else : 
                    if df_data.empty:
                        df_data = data
                    else:
                        df_data = pd.concat((df_data, data),axis=1)
            else:
                df_data = pd.concat([df_data, data], ignore_index=True)

        return df_data

# %% Utils

    def __create_n_columns(self):
        for n in range(self.n_dof):
            self.n_columns_ang.append(str(n+1))

    def __create_n_vel_columns(self):
        # for n in range(self.n_columns_ang):
        for n in self.n_columns_ang:
            self.n_columns_vel.append('vel_' + n)

    def __set_data_name(self):
        """
        If no data name is given this function sets the name of the data as the name
        of the first joint angle file given.

        Returns
        -------
        name : string
            name of the first joint angle file name.

        """
        text = self.list_files_angles[0].rsplit("/", 1)[-1]
        directory = self.list_files_angles[0].rsplit("/")[-2]
        text = text.rsplit(".", 1)[0]
        name = directory+'/'+text
        return name

    def __get_data_columns(self, position, mean_data=False):
        if mean_data:
            data = [self.mean_movement_ang]
        else:
            data = self.list_data_angles
        if position:
            columns = self.get_n_columns_ang()
        else:
            columns = self.get_n_columns_vel()

        list_data = []
        for d in data:
            list_data.append(d[columns])

        return list_data, columns
    
    #%% Plot functions
    
    def plot_angle_angle(self):
        for i, data in enumerate(self.list_data_angles):
            fig = sns.pairplot(data[self.n_columns_ang], corner=True)
            fig.fig.suptitle('Angle Angle Plot for : ' + self.get_data_name(), y=1.02)
        return fig
    
    def plot_continuous_relative_phase(self):
        results = crp.compute_crp(self.list_data_angles, self.n_columns_ang, self.n_columns_vel, 50,
                                   self.theta_combination,  str(self.get_data_name()), plot=False)
        print(results)
        data_norm = results[0][1]
        real_phase_col_list = [
            col for col in data_norm if col.startswith('rel_phase')]
        fig, axes = plt.subplots(len(self.theta_combination), 1, figsize=(8, 4 * len(self.theta_combination)))
        
        for idx, combination in enumerate(self.theta_combination):
            ax = axes[idx]
            column_name = f'rel_phase_{combination[0]}_{combination[1]}'
            if column_name in data_norm.columns:
                ax.plot(data_norm.index, data_norm[column_name], label=f'{combination[0]} vs {combination[1]}')
                ax.set_title(f'CRP: {combination[0]} vs {combination[1]}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Relative Phase')
                ax.legend()
        
    
        return fig 
    
    def plot_pca(self):
        result, fig = pca.compute_pca(self, plot=True, position=True, spca=False, verbose_plot=False, last=False)
        
        return fig 
    
    def plot_jcvpca(self, coord_metric_comparison, nPCA):
        print('Compute JcvPCA')
        if nPCA not in [1, 2, 3, 4, 5, 10]:
            raise ValueError("nPCA must be 1 or 10")
        pca_res, res, sub, fig = pca.compute_pca_reprojection2(self, coord_metric_comparison, position=True, absolute=True, plot=True, emg=False, nPCA=nPCA)
        print(pca_res, res, sub, fig)
        return fig, sub
    
    def plot_jsvcrp(self, coord_metric_comparison):
        res1 = self.compute_crp(plot=False, verbose_plot=False, emg=False)
        print(res1[0][1])
        res2 = coord_metric_comparison.compute_crp(plot=False, verbose_plot=False, emg=False)
        result_list, results_matrix_area, fig, fig2 = crp.crp_difference(res1[0][1], res2[0][1], self.get_n_columns_ang(), self.get_data_name(), coord_metric_comparison.get_data_name())
        return fig, fig2, result_list

    def plot_statistical_correlation(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        data = pd.concat(self.list_data_angles, ignore_index=True)
        corr = data[self.n_columns_ang].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix for Joint Angles")
        return fig 

def dot(v1, v2):
    """
    Compute dot products of 2 vectors

    Parameters
    ----------
    v1 : TYPE
        DESCRIPTION.
    v2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    v1['res'] = 0
    for i in range(max(len(v1), len(v2))):
        a = v1.iloc[i]
        b = v2.iloc[i]
        v1['res'].iloc[i] = sum([c1 * c2 for c1, c2 in zip(a, b)])
    return v1['res']
