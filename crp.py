#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:58:50 2023

@author: dubois
"""
import data_plot as dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as sig
import utils as ut
import math
import itertools
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
import matplotlib.cm as cm
from shapely.geometry import Polygon

def compute_crp(data, angle_name, angle_vel_name, max_ee_vel, theta_combination, name, anim=False, mean=False, plot=True, verbose_plot=False, split = 1):
    # plot raw data
    # print(len(data))
    list_result = []

    if split != 1 :
        plot = False
        anim = False
        mean = False
        temp_data = data
        random.shuffle(temp_data)

        list_data = ut.split(temp_data, split)


    else :
        list_data = [data]

    if plot:
        fig = plt.figure(figsize=(8, 10))

        fig.suptitle('Continuous Relative Phase' + ' Data : ' +
                     str(name))
    # numbers of subfig
    max_i = len(theta_combination)+5

    # empty lists
    mean_list = []
    dp_list = []
    max_list = []
    min_list = []
    deriv_list = []
    max_vel_list =[]
    all_crp_data = pd.DataFrame()

    # for each subfigure plot blue and green area defined as in-phase and out of phase
    if plot :
        for i in range(max_i-4):
            ax = fig.add_subplot(max_i, 1, i+1)
            if i != 0:
                in_phase = ax.axhspan(-30, 30, facecolor='b', alpha=0.2)
                out_of_phase = ax.axhspan(-210, -150, facecolor='g', alpha=0.2)
                ax.axhspan(210, 150, facecolor='g', alpha=0.2)
                ax.legend([in_phase, out_of_phase], [
                    'in-phase', 'out-of-phase'], bbox_to_anchor=(1.05, 0.5), loc='lower center')

    for l_d in list_data:

        # for each trial
        for k, d in enumerate(l_d):
            # normalize data
            if not mean :
                raw_data = ut.normalize_data_in_time(d, dropna=True)
            else :
                raw_data = data[0]

            # range normalization
            data_norm = 2*((raw_data-raw_data.min()) /
                           (raw_data.max()-raw_data.min()))-1

            data_norm = data_norm.iloc[2:]
            if mean:
                data_norm = data_norm.dropna()


            # for each joint
            for n in angle_name:

                # if joint is not moving, then phase angle is null
                # TO DO : seuil adaptatif
                if(raw_data[n].std() < 0.0005):
                    data_norm["phase_" + n] = np.zeros(len(data_norm))
                else:
                    # remove first data point that has no velocity
                    data = data_norm.iloc[1:, :]

                    # get velocity data
                    dataspeed = data["vel_" + n].to_numpy()
                    a = [dataspeed]
                    # get position data
                    datatheta = data[n].to_numpy()
                    b = [datatheta]


                    # compute phase angle
                    data_norm["phase_"+n] = np.hstack((np.zeros(1),
                                                       np.rad2deg(np.arctan2(a[0], b[0]))))


                    data_norm["phase_"+n] = np.unwrap(
                        data_norm["phase_"+n], period=np.rad2deg(math.pi))

            # compute crp
            for i in list(itertools.combinations(angle_name, 2)):
                data_norm["rel_phase_" + "_".join(i)] = round(
                    data_norm["phase_" + i[0]] - data_norm["phase_" + i[1]], 5)

                data_norm["rel_phase_" + "_".join(i)]= np.unwrap(
                    data_norm["rel_phase_" + "_".join(i)])

            # get mean, max dp, deriv, min for each columns
            real_phase_col_list = [
                col for col in data_norm if col.startswith('rel_phase')]
            phase_col_list = [col for col in data_norm if col.startswith('phase')]
            vel_col_list = [col for col in data_norm if col.startswith('vel')]

            mean_crp = data_norm[real_phase_col_list].mean()
            dp_crp = data_norm[real_phase_col_list].std()
            max_crp = data_norm[real_phase_col_list].max()
            min_crp = data_norm[real_phase_col_list].min()
            deriv_crp = data_norm[real_phase_col_list].diff().mean()

            data_norm['time_t2'] = data_norm.index
            data_norm['diff'] = (data_norm['time_t2']- max_ee_vel).abs()

            if not data_norm['diff'].empty:
                val_max_ee_crp = pd.DataFrame(data_norm[real_phase_col_list].loc[data_norm['diff'].idxmin()])
            # concatenate all crp values
            if all_crp_data.empty:
                all_crp_data = data_norm[real_phase_col_list+phase_col_list+vel_col_list+angle_name]
            else:
                all_crp_data = all_crp_data.append(
                    data_norm[real_phase_col_list+phase_col_list+vel_col_list+angle_name])

            # save extracted features
            mean_list.append(mean_crp)
            dp_list.append(dp_crp)
            max_list.append(max_crp)
            min_list.append(min_crp)
            deriv_list.append(deriv_crp)
            max_vel_list.append(pd.DataFrame(val_max_ee_crp))
            # print(max_vel_list)

            # for each trial plot its crp
            if plot :
                ax = dp.plot_crp(angle_name, raw_data, data_norm[real_phase_col_list],
                                 max_ee_vel, fig)
        if plot :
            ax.legend(np.arange(len(data)))
            data_norm[real_phase_col_list].plot(subplots=True, layout=(1, 6), figsize=(15, 6))

        if anim:
            animation(data_norm[real_phase_col_list], raw_data, angle_name, name)

        if verbose_plot:
            all_data = all_crp_data
            for n in angle_name:
                fig3, ax = plt.subplots(4, figsize=(8,12))
                fig3.suptitle(str(name)+ ' '+ n)
                ax[0].set_title('Normalized positions')
                ax[0].plot(all_data[n], '.')
                ax[0].set_ylim([-1.2, 1.2])

                ax[1].set_title('Normalized velocity')
                ax[1].plot(all_data['vel_'+n], '.')
                ax[1].set_ylim([-1.2, 1.2])

                ax[2].set_title('Position / Velocity')
                ax[2].plot(all_data[n], all_data['vel_'+n], '.')
                ax[2].set_ylim([-1.2, 1.2])
                ax[2].set_xlim([-1.2, 1.2])
                ax[2].set_aspect('equal', 'box')
                ax[2].axhline(0)  # x-axis line
                ax[2].axvline(0)  # y-axis line

                ax[3].set_title('Phase')
                ax[3].plot(all_data['phase_'+n], '.')
                ax[3].set_ylim([-250, 250])

        # plot global informations about crp
        mean_crp = pd.DataFrame(mean_list).mean()
        dp_crp = pd.DataFrame(dp_list).mean()
        deriv_crp = pd.DataFrame(deriv_list).mean()
        max_crp = pd.DataFrame(max_list).max()
        min_crp = pd.DataFrame(min_list).min()
        if mean:
            max_vel_list=max_vel_list[0]
            val_max_ee = pd.DataFrame(max_vel_list)
        else:
            val_max_ee = pd.DataFrame(pd.concat(max_vel_list, axis=1)).transpose().dropna().mean()

        if plot:
            ax = fig.add_subplot(max_i, 1, max_i-3)

            # table plot
            # hide axes
            ax.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')

            ax.table(cellText=[mean_crp],
                     colLabels=pd.DataFrame(mean_list).columns, loc='center')
            ax.set_title('Mean')

            ax = fig.add_subplot(max_i, 1, max_i-2)

            # hide axes
            ax.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=[dp_crp],
                     colLabels=pd.DataFrame(dp_list).columns, loc='center')
            ax.set_title('Deviation Phase')

            ax = fig.add_subplot(max_i, 1, max_i-1)

            # hide axes
            ax.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=[deriv_crp],
                     colLabels=pd.DataFrame(deriv_list).columns, loc='center')
            ax.set_title('Deriv')

            ax = fig.add_subplot(max_i, 1, max_i)

            # hide axes
            ax.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')

            pd.plotting.table(ax=ax, data=val_max_ee)
            ax.set_title('Value at max vel')

        list_result.append((pd.DataFrame([mean_crp, dp_crp, max_crp, min_crp, deriv_crp, val_max_ee], index=['mean', 'dp', 'max', 'min', 'deriv', 'crp_max_ee_vel']), all_crp_data[real_phase_col_list]))

    if mean : 
        return pd.concat([mean_crp, dp_crp, max_crp, min_crp, deriv_crp, pd.Series(val_max_ee.transpose().values[0], index=val_max_ee.transpose().columns)], axis=1), data_norm[real_phase_col_list]
    else :
        return list_result


def animation(data_norm, raw_data, angle_name, name):
    fig2, ax2 = plt.subplots(4, len(angle_name), figsize=(20, 20))
    fig2.suptitle(name)

    modulo = 1
    print(raw_data.columns)
    data_norm['index_c'] = data_norm.index
    raw_data['index_c'] = raw_data.index

    data_norm = data_norm[data_norm["index_c"] % modulo == 0]
    data_modif = raw_data[raw_data['index_c'] % modulo == 0]

    # print(raw_data)

    def update(k):
        data_m = data_modif.head(k+1)
        data_m_norm = data_norm.head(k+1)
        # print('data ?')
        # print(data_m)

        for i in range(len(angle_name)):
            ax2[0, i].cla()
            sns.scatterplot(
                data=data_m_norm, y=angle_name[i], x="index_c",
                hue="index_c", ax=ax2[0, i], legend=False)
            ax2[0, i].set_ylim(-1, 1)
            ax2[0, i].set_xlim(0, len(data_norm))
            ax2[0, i].set_title(angle_name[i])
            ax2[0, i].set_ylabel(angle_name[i])
            ax2[0, i].set_xlabel("Time")

            ax2[1, i].cla()
            sns.scatterplot(
                data=data_m_norm, y=angle_name[i], x="index_c",
                hue="index_c", ax=ax2[1, i], legend=False)
            # ax[1, i].set_ylim(data_modif["vel_"+self.n_columns_ang[i]].min()-0.00001,
            #                  data_modif["vel_"+self.n_columns_ang[i]].max()+0.00001)
            ax2[1, i].set_ylim(-1, 1)
            ax2[1, i].set_xlim(0, len(data_m_norm))
            ax2[1, i].set_ylabel("velocity "+angle_name[i])
            ax2[1, i].set_xlabel("Time")

            ax2[2, i].cla()
            sns.scatterplot(
                data=data_m_norm, x=angle_name[i],
                hue="index_c", ax=ax2[2, i], legend=False)
            ax2[2, i].set_ylim(-1, 1)
            ax2[2, i].set_xlim(-1, 1)
            ax2[2, i].set_ylabel("velocity "+angle_name[i])
            ax2[2, i].set_xlabel(angle_name[i])

            ax2[3, i].cla()
            sns.scatterplot(
                data=data_m, y="phase_"+angle_name[i], x="index_c",
                hue="index_c", ax=ax2[3, i], legend=False)
            ax2[3, i].set_ylim(data_modif["phase_"+angle_name[i]].min()-2,
                               data_modif["phase_"+angle_name[i]].max()+2)
            ax2[3, i].set_xlim(0, len(data_m))
            ax2[3, i].set_ylabel("crp " + angle_name[i])
            ax2[3, i].set_xlabel("Time")

        return fig2

    ani = FuncAnimation(fig2, update, frames=len(
        data_modif), interval=0)
    writervideo = FFMpegWriter(fps=1)
    ani.save('crp.mp4', writer=writervideo)

    plt.show()

# def cross_corr_crp_sliding_window(sig1, sig2, win_len):
#     lw = int(np.floor(win_len/2))
#     if lw==0:
#         lw=1
#
#     corr = np.zeros(len(sig1))
#     for k,d in enumerate(sig1):
#         corr_sig = sig.correlate(sig1.iloc[k:k+lw], sig2.iloc[k:k+lw])
#         corr[k] = corr_sig[int(np.floor((len(corr_sig) - 1)/2))]
#         if lw<=win_len:
#             lw=lw+1
#         if k*lw >=len(sig1):
#             lw = lw-1
#
#     return corr
# def cross_corr_crp(sig1, sig2):
#     corr = sig.correlate(sig1, sig2)
#
#     return corr

def crp_difference(base1, base2, n_angles, data_name_1, data_name_2):
        results_matrix = pd.DataFrame(index=n_angles, columns=n_angles)
        results_matrix_area = pd.DataFrame(index=n_angles, columns=n_angles)

        col = base1.columns

        fig, ax = plt.subplots(1, len(col), figsize=(15, 6))
        if len(col)==1:
            ax=[ax]

        list_crp_diff = []
        for i, c in enumerate(col):

            # Plot base
            ax[i].plot((base1[c].groupby('time').mean()), c='b', label=data_name_1)
            mean_low = base1[c].groupby('time').mean() - base1[c].groupby(
                'time').std()  # /np.sqrt(len(coord_metric.get_all_data_ang()))
            mean_high = base1[c].groupby('time').mean() + base1[c].groupby(
                'time').std()  # /np.sqrt(len(coord_metric.get_all_data_ang()))
            ax[i].fill_between(mean_low.index, mean_low, mean_high, alpha=0.2, color='b')
            # ax[i].plot(mean_low c='b')
            # ax[i].plot(), c='b')
            ax[i].set_title(c)

            # plot new data
            ax[i].plot((base2[c].groupby('time').mean()), c='m', label=data_name_2)
            mean_low = base2[c].groupby('time').mean() - base2[c].groupby(
                'time').std()  # /np.sqrt(len(coord_metric.get_all_data_ang()))
            mean_high = base2[c].groupby('time').mean() + base2[c].groupby(
                'time').std()  # /np.sqrt(len(coord_metric.get_all_data_ang()))
            ax[i].fill_between(mean_low.index, mean_low, mean_high, alpha=0.2, color='m')
            ax[i].legend()

            # cross_corr_w = cross_corr_crp_sliding_window(base1[c].groupby('time').mean(),
            #                                                  base2[c].groupby('time').mean(), 100)
            # plt.figure()
            # plt.plot(cross_corr_w)
            # plt.suptitle('Sliding window ' + data_name_2 + ' and ' + data_name_1 + ' \n' + c)
            #
            # cross_corr = cross_corr_crp(base1[c].groupby('time').mean(), base2[c].groupby('time').mean())
            # plt.figure()
            # plt.plot(cross_corr)
            # plt.suptitle(
            #     'Cross Correlation ' + data_name_2 + ' and ' + data_name_1 + ' \n' + c)

            polygon_points = []  # creates a empty list where we will append the points to create the polygon

            for xyvalue in pd.DataFrame(base1[c].groupby('time').mean()).itertuples():
                polygon_points.append([xyvalue[0], xyvalue[1]])  # append all xy points for curve 1

            for xyvalue in pd.DataFrame(base2[c].groupby('time').mean()).iloc[::-1].itertuples():
                polygon_points.append([xyvalue[0], xyvalue[
                    1]])  # append all xy points for curve 2 in the reverse order (from last point to first point)

            for xyvalue in pd.DataFrame(base1[c].groupby('time').mean()).iloc[0:1].itertuples():
                polygon_points.append(
                    [xyvalue[0], xyvalue[1]])  # append the first point in curve 1 again, to it "closes" the polygon

            polygon = Polygon(polygon_points)
            area = polygon.area
            x, y = polygon.exterior.xy
            ax[i].fill(x, y, alpha=0.2)
            ax[i].set_ylim([-200, 200])

            mean2 = base2[c].groupby('time').mean()
          
            str_col = c.split('_')

            #results_matrix.at[str_col[-2], str_col[-1]] = under.values.sum() + over.values.sum()
            results_matrix_area.at[str_col[-2], str_col[-1]] = area

            list_crp_diff.append(
                pd.Series(np.array([area]), index=['Area'],
                          name=c).transpose())

        fig.suptitle(data_name_1 + ' as base and ' + data_name_2)
        max_y = min_y = 0
        for axs in ax:
            lim_x, lim_y = max(axs.get_xlim()), max(axs.get_ylim())
            if lim_y > max_y:
                max_y = lim_y
            lim_x, lim_y = min(axs.get_xlim()), min(axs.get_ylim())
            if lim_y < min_y:
                min_y = lim_y
        plt.setp(ax, ylim=(min_y - 10, max_y + 10))
        mask = np.tril(np.ones_like(results_matrix_area))
        #sns.heatmap(results_matrix.fillna(0), annot=True, mask=mask, vmin=0, vmax=1000, fmt=".1f", cmap=cm.gray_r)
        #plt.suptitle('Sum Over + Under')

        fig2 = plt.figure()
        sns.heatmap(results_matrix_area.fillna(0), annot=True, mask=mask, vmin=1000, vmax=12000, fmt=".1f",cmap="Reds")
        plt.suptitle('Area')
        result_list = pd.concat(list_crp_diff, axis=1)
        result_list.columns = col

        return result_list, results_matrix_area, fig, fig2
