#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:12:14 2022.

@author: dubois
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
from matplotlib.pyplot import cm
import utils as ut
import seaborn as sns
import random
from sklearn.decomposition import PCA


matplotlib.rcParams.update({'font.size': 10})


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def compute_pca(coord_metric, plot=True, position=True, spca=False, verbose_plot=False, last=False):

    # get columns and data
    if position:
        columns = coord_metric.get_n_columns_ang()
        df_mean_movement_norm = pd.DataFrame(np.rad2deg(
            coord_metric.get_mean_data()), columns=coord_metric.get_mean_data().columns)

    else:
        columns = coord_metric.get_n_columns_vel()
        df_mean_movement_norm = pd.DataFrame(np.rad2deg(coord_metric.get_mean_data_vel()), columns=coord_metric.get_mean_data_vel()
                                             .columns)

    df_data = coord_metric.get_concatenated_data(position=position)
    list_of_data = [df_data]

    pca_list = []
    for data in list_of_data:

        # drop nans and get number of components and data
        data = data.dropna()
        n = len(data)
        p = coord_metric.get_n_dof()

        # print(data)
        # if spca : not working for now
       
   
        pca = PCA(n_components=p)

        # center data
        # if not all same unit, normalize, if same unit, center only
        # df_norm = (data - data.mean())/data.std()
        # df_norm = df_norm.dropna()
        # pca_data = pca.fit_transform(df_norm)

        # data_norm = StandardScaler().fit_transform(data)
        # data_norm = pd.DataFrame(data_norm, columns=data.columns)

        data_norm = data - data.mean()
        pca_data = pca.fit(data_norm)

       
        explained_var = pca.explained_variance_ratio_

        eigval = (n - 1) / n * explained_var
        cum_sum_variance = np.cumsum(explained_var)

        n_pca = len(cum_sum_variance[cum_sum_variance < 0.95] < 0.95) + 1

        # pc reconstruction
        pc = pd.DataFrame(pca.components_,
                          columns=columns)
        pc_reconstruct = pd.DataFrame(
            np.zeros((len(df_mean_movement_norm), len(pc))), index=df_mean_movement_norm.index, columns=columns)

        for i, n in enumerate(columns):
            pc_reconstruct[n] = (pc.loc[i]*df_mean_movement_norm).sum(axis=1)

        if verbose_plot:
            fig = plt.figure()
            ax = fig.add_subplot(223)
            ax.plot(np.arange(1, p + 1), eigval)
            ax.set_title("Scree plot")
            ax.set_ylabel("Eigen values")
            ax.set_xlabel("Factor number")

            ax = fig.add_subplot(224)
            ax.plot(np.arange(1, p + 1), np.cumsum(explained_var))
            ax.set_title("Explained variance vs. # of factors")
            ax.set_ylabel("Cumsum explained variance ratio")
            ax.set_xlabel("Factor number")
            plt.show()

            ax = fig.add_subplot(211)
            data['t'] = data.index
            color = iter(cm.rainbow(np.linspace(0, 1, len(data.columns))))
            for n in columns:
                c = next(color)
                data.plot.scatter(x='t', y=n, ax=ax, c=c)
            # ax.scatter(data.index, data.values)
            ax.legend(coord_metric.get_columns_name())

        if plot:
            fig = plt.figure(figsize=(8, 10))
            ax = fig.add_subplot(311)
            # biplot(pca_data[:, -2:], np.transpose(pca.components_[-2:, :]),
            #        explained_var,  ax, position=position, last=last)

            biplot(pca_data.explained_variance_, np.transpose(pca.components_),
                   explained_var,  ax, labels=columns, last=False)

            ax = fig.add_subplot(312)
            pc_reconstruct['t'] = pc_reconstruct.index
            color = iter(cm.rainbow(np.linspace(
                0, 1, len(pc_reconstruct.columns))))
            for n in columns:
                c = next(color)
                pc_reconstruct[1:].plot(x='t', y=n, ax=ax, c=c)
                # ylim=[-180, 180], c=c)
            ax.legend(np.arange(coord_metric.get_n_dof())+1)
            ax.set_ylabel('PC reconstruction (deg)')

            bar_plot(pca, coord_metric.get_n_columns_ang(), fig)
            fig.suptitle(' PCA \n Data : ' + coord_metric.get_data_name() +
                         '\n Position : ' + str(position))

            fig = plt.figure(figsize=(15, 10))
            y_pos = np.arange(len(pca.components_[0, :]))

            data = pd.DataFrame(
                (pca.components_.transpose()*explained_var).transpose(), columns=columns)

            pc_list = pc_columns(pca.components_.shape[0])
            ax = fig.add_subplot(121)
            ax = data.plot.bar(rot=0, ax=ax, width=0.2)
            ax.set_xticklabels(pc_list)
            ax.set_ylabel('PC weight')

            ax = fig.add_subplot(122)
            sns.heatmap(pca.components_,
                        cmap='YlGnBu',
                        yticklabels=["PC_"+str(x-1)
                                     for x in range(1, pca.n_components_+1)],
                        xticklabels=list(data.columns),
                        cbar_kws={"orientation": "horizontal"},
                        ax=ax)
            ax.set_aspect("equal")

        pca_list.append(pca)
    #     print(pca.components_)
    # print('list')
    # print(len(pca_list))
    return pca_list, fig


def biplot(score, coeff,  explained_var, ax,  labels=None, last=False):
    """
    PCA biplot.

    Plot along 2 main PCs and original axis projection inside the PC's plan'

    Parameters
    ----------
    score : TYPE
        DESCRIPTION.
    coeff : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # if last:
    #     xs = score[:, -2]
    #     ys = score[:, -1]
    # else:
    #     xs = score[:, 0]  # projection on PC1
    #     ys = score[:, 1]  # projection on PC2

    n = coeff.shape[0]  # number of variables
   # ax.scatter(xs, ys)  # color based on group

    for i in range(n):
        # plot as arrows the variable scores (each variable has a
        # score for PC1 and one for PC2)
        ax.arrow(
            0,
            0,
            coeff[i, 0],
            coeff[i, 1],
            color="grey",
            alpha=0.6,
            linestyle="-",
            linewidth=1,
            overhang=0.2,
        )
        ax.text(
            coeff[i, 0] * 1.15,
            coeff[i, 1] * 1.15,
            labels[i],
            color="grey",
            ha="center",
            va="center",
            fontsize=10,
        )

    if last:
        ax.set_xlabel(
            "PC_3 : " + str(round(explained_var[-2] * 100, 2)),
            size=14,
        )
        ax.set_ylabel(
            "PC_2 : " + str(round(explained_var[-1] * 100, 2)),
            size=14,
        )
        string = 'PC_0 = ' + \
            str(round(explained_var[0]*100, 2)) + \
            '\n PC_1 = ' + str(round(explained_var[1]*100, 2))

    else:
        ax.set_xlabel(
            "PC_0 : " + str(round(explained_var[0] * 100, 2)),
            size=14,
        )
        ax.set_ylabel(
            "PC_1 : " + str(round(explained_var[1] * 100, 2)),
            size=14,
        )
        if len(explained_var) > 2:
            string = 'PC_2 = ' + \
                str(round(explained_var[2]*100, 2))
        else:
            string = ''

    trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
    ax.annotate(string, xy=(0.9, 1.05), xycoords=trans)
    limx = 2  # int(xs.max()) + 1
    limy = 2  # int(ys.max()) + 1
    ax.set_xlim([-limx, limx])
    ax.set_ylim([-limy, limy])
    ax.grid(visible=True)
    ax.tick_params(axis="both", which="both", labelsize=14)


def bar_plot(pca, columns, fig):
    """
    Plot the joints in each PC.

    Returns
    -------
    None.

    """
    y_pos = np.arange(len(pca.components_[0, :]))
    data = pd.DataFrame(pca.components_, columns=columns)

    pc_list = pc_columns(pca.components_.shape[0])
    ax = fig.add_subplot(313)
    ax = data.plot.bar(rot=0, ax=ax, width=0.2)
    ax.set_xticklabels(pc_list)
    ax.set_ylabel('PC weight')


def pc_columns(n_pc):
    pc_list = []
    for n in range(n_pc):
        pc_list.append('PC_'+str(n))
    return pc_list

# %% Bockemuhl


def compute_pca_distance(coord_metric, coord_metric_list, position=True, plot=True):
    pca_data = coord_metric.compute_pca(
        plot=False, position=position, spca=False, verbose_plot=False)

    subspace = get_pca_subspace(coord_metric)
    # print('ref')
    # print(coord_metric.get_data_name())

    subspace_data_list = []
    dist_list = []
    ang_list = []
    for i, d in enumerate(coord_metric_list):
        res_pca = d.compute_pca(plot=False, position=position,
                                spca=False, verbose_plot=False)
        for s in subspace:
            # print(d.get_data_name())
            # print('debug')
            # print(dist_sous_espace(s[:, -1],
            #                        get_pca_subspace(d)[0][:, -1]))
            # print(get_pca_subspace(d))
            subspace_data_list.append(
                (d.get_data_name(), get_pca_subspace(d)[0]))
            dist_list.append((d.get_data_name(), d.get_data_name(), dist_sous_espace(
                s[:, -1], subspace_data_list[-1][1][:, -1])[0]))
            ang_list.append((d.get_data_name(), d.get_data_name(), math.asin(dist_sous_espace(
                s[:, -1], subspace_data_list[-1][1][:, -1])[0])*180/math.pi))

    if plot:
        labels, _, ys = zip(*dist_list)

        result = pd.DataFrame()
        result['condition'] = labels
        result['distance'] = ys

        # ys = [elem[0] for elem in ys]

        # xs = np.arange(len(labels))
        # width = 0.6

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.barplot(x='condition', y='distance', data=result)
        # ax.bar(xs, ys, width, align='center')
        # ax.set_ylim([0, 1])
        ax.set_title('PCA distance \n Position : ' + str(position) +
                     '\n Ref : ' + str(coord_metric.get_data_name()))
        plt.xticks(rotation=70)
        plt.tight_layout()
        # Replace default x-ticks with xs, then replace xs with labels
        # plt.xticks(xs, labels)
        plt.yticks(ys)
        ax.set_ylim(0, 1)

        # plot angles
        labels, _, ys = zip(*ang_list)

        result = pd.DataFrame()
        result['condition'] = labels
        result['distance'] = ys

        # ys = [elem[0] for elem in ys]

        # xs = np.arange(len(labels))
        # width = 0.6

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.barplot(x='condition', y='distance', data=result)
        # ax.bar(xs, ys, width, align='center')
        # ax.set_ylim([0, 1])
        ax.set_title('PCA distance \n Position : ' + str(position) +
                     ' \n Ref : ' + str(coord_metric.get_data_name()))
        ax.set_ylim(0, 90)
        plt.xticks(rotation=70)
        plt.tight_layout()
        # Replace default x-ticks with xs, then replace xs with labels
        # plt.xticks(xs, labels)
        plt.yticks(ys)

    return dist_list


def compute_pca_distance_baseline(coord_metric, position=True, plot=True, n=2):
    pca_data = coord_metric.compute_pca(
        plot=False, position=position, spca=False, verbose_plot=False)
    subspace = []
    dist_list = []
    for d in pca_data:
        subspace.append(d.components_)
    # print(subspace)
    # print(len(subspace))

    for n in range(len(subspace)-1):

        dist_list.append(dist_sous_espace(
            subspace[n][:, -1], subspace[n+1][:, -1])[0])

    return dist_list


def compute_pca_reprojection(coord_metric, list_coord_metrics, position=False):
    pca_base = coord_metric.get_pca(position)[0]

    if position:
        columns = coord_metric.get_n_columns_ang()
        df_mean_movement_norm = pd.DataFrame(np.rad2deg(
            coord_metric.get_mean_data()), columns=coord_metric.get_mean_data().columns)

    else:
        columns = coord_metric.get_n_columns_vel()
        df_mean_movement_norm = pd.DataFrame(np.rad2deg(coord_metric.get_mean_data_vel()), columns=coord_metric.get_mean_data_vel()
                                             .columns)

    pc = pd.DataFrame(pca_base.components_,
                      columns=columns)
    pc_reconstruct = pd.DataFrame(
        np.zeros((len(df_mean_movement_norm), len(pc))), index=df_mean_movement_norm.index, columns=columns)
    for i, n in enumerate(columns):
        pc_reconstruct[n] = (pc.loc[i]*df_mean_movement_norm).sum(axis=1)

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    fig.suptitle('PCA reconstruction')
    pc_reconstruct['t'] = pc_reconstruct.index
    color = iter(cm.rainbow(np.linspace(0, 1, len(pc_reconstruct.columns))))
    for n in columns:
        c = next(color)
        pc_reconstruct[1:].plot(x='t', y=n, ax=ax, c=c, style='--')
        # ylim=[-180, 180], c=c)
        ax.legend(np.arange(coord_metric.get_n_dof())+1)
        ax.set_ylabel('PC reconstruction')

    for l in list_coord_metrics:
        df_data_l = l.get_concatenated_data(position)
        df_data_base = coord_metric.get_concatenated_data(position)

        if position:
            df_mean_movement_norm_l = pd.DataFrame(np.rad2deg(
                l.get_mean_data()), columns=l.get_mean_data().columns)
        else:
            df_mean_movement_norm_l = pd.DataFrame(np.rad2deg(
                l.get_mean_data_vel()), columns=l.get_mean_data_vel().columns)

        pc_reconstruct_l = pd.DataFrame(
            np.zeros((len(df_mean_movement_norm_l), len(pc))), index=df_mean_movement_norm_l.index, columns=columns)
        for i, n in enumerate(columns):
            pc_reconstruct_l[n] = (
                pc.loc[i]*df_mean_movement_norm_l).sum(axis=1)
        pc_reconstruct_l['t'] = pc_reconstruct_l.index
        color = iter(cm.rainbow(np.linspace(
            0, 1, len(pc_reconstruct.columns))))
        for n in columns:
            c = next(color)
            pc_reconstruct_l[1:].plot(x='t', y=n, ax=ax, c=c)

        # reprojection in the data base
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(111)
        ax.scatter(df_data_l[columns[0]], df_data_l[columns[1]], c='b')
        ax.scatter(df_data_base[columns[0]], df_data_base[columns[1]], c='r')

        df_data_l = np.matmul(df_data_l.to_numpy(), pca_base.components_)
        df_data_l = pd.DataFrame(df_data_l, columns=columns)

        df_data_base = np.matmul(df_data_base.to_numpy(), pca_base.components_)
        df_data_base = pd.DataFrame(df_data_base, columns=columns)

        ax.scatter(df_data_l[columns[0]],
                   df_data_l[columns[1]], c='b', alpha=0.5)
        ax.scatter(df_data_base[columns[0]],
                   df_data_base[columns[1]], c='r', alpha=0.5)

    return


def compute_fpca(coord_metric, position=False, time_wrapping=False):
    if position:
        columns = [coord_metric.get_n_columns_ang()[0]]

    else:
        columns = [coord_metric.get_n_columns_vel()[0]]
    data = coord_metric.get_concatenated_data(position, normalized=True)

    # mettre de 0 à 100% et creation de la fdgrid
    data_grid = []
    for n in columns:
        data_var = []
        for d in coord_metric.get_all_data_ang():
            if time_wrapping:
                data = ut.time_wrap_data(d, coord_metric.get_mean_data()[n])
            else:
                data = ut.normalize_data_in_time(d)
            data[columns] = data[columns].fillna(0)
            d = data[n].to_numpy()
            if not list(data_var):
                data_var = d
            else:
                data_var = np.vstack((data_var, d))
        data_grid.append(np.array(data_var))
    data_grid = np.array(data_grid)
    data_grid = np.moveaxis(data_grid, [0, 1, 2], [2, 0, 1])


    # fpca
    fpca_discretized = FPCA(n_components=2)
    grid_points = data.index.to_numpy()
    data[columns].plot(style='.')
    data = skfda.representation.grid.FDataGrid(
        data_grid, grid_points=grid_points)
    # data.scatter()
    
    basis = skfda.representation.basis.BSplineBasis(n_basis=7)
    basis_fd = data.to_basis(basis)
    fpca_discretized.fit(data)
    # plot
    fpca_discretized.components_.plot()
    fig = plt.gcf()
    fig.suptitle('PC1 and PC2 \n position = ' + str(position))
    FPCAPlot(
        data.mean(),
        fpca_discretized.components_,
        factor=0.005,
        fig=plt.figure(figsize=(6, 2 * 4)),
        n_rows=2,
    ).plot()
    fig = plt.gcf()
    fig.suptitle('Position = ' + str(position))

    return fpca_discretized


# def compute_nmf(coord_metric, position=False):
#     if position:
#         columns = coord_metric.get_n_columns_ang()

#     else:
#         columns = coord_metric.get_n_columns_vel()

#     data = coord_metric.get_concatenated_data(
#         position=position)[columns].dropna()

#     # remove negative values
#     # print(data.min().min())
#     if data.min().min():
#         data = data-data.min().min()
#     else:
#         data = data+data.min().min()




def compute_pca_reprojection_noise(coord_metric, position=True, n=2):

    if position:
        columns = coord_metric.get_n_columns_ang()

    else:
        columns = coord_metric.get_n_columns_vel()

    data = coord_metric.get_concatenated_data(
        position=position)[columns].dropna()

    # split data in n subdatasets
    list_df = np.array_split(data, n)


    res_list = []
    for d1, d2 in itertools.combinations(list_df, 2):
        # 1) PCA sur data before exp
        _, pca1, var1 = get_pca_frame(d1.dropna().to_numpy())

        # 2) Reprojeter data2 dans les PCA de data1

        data2_transformed = np.matmul(pca1, d2.dropna().to_numpy().transpose())


        # 3) Faire une pca sur cet ensemble de données reprojeter

        _ , pca2, var2 = get_pca_frame(data2_transformed.transpose())

        res = np.matmul(pca2, pca1)

        sub = res-pca1
        res_list.append((sub, var1))

    return res_list

def compute_pca_reprojection2(coord_metric, coord_metric_2, position=True, absolute=True, plot=False, emg=False, nPCA=2):
    # test methode
    if emg:
        columns = coord_metric.n_columns_emg
        data = coord_metric.get_concatenated_data(position=False, emg=True)[columns].dropna()
        _, pca1, var1 = get_pca_frame(data.to_numpy(), nPCA)
        data2 = coord_metric_2.get_concatenated_data(position=False, emg=True)[columns].dropna()
    else:
        if position:
            columns = coord_metric.get_n_columns_ang()

        else:
            columns = coord_metric.get_n_columns_vel()

        # 1) PCA sur data before exp
        data = coord_metric.get_concatenated_data(position=position)[columns].dropna()
        # fig, ax = plt.subplots(figsize=(8, 6))
        # fig.suptitle('Dataset 1')
        # ax.scatter(data.to_numpy()[:, 2], data.to_numpy()[:, 3], alpha=0.7, c='orange')
        _, pca1, var1 = get_pca_frame(data.to_numpy(), nPCA)
        data2 = coord_metric_2.get_concatenated_data(position=position)[columns].dropna()
        
    # 2) Reprojeter data after exp dans les PCA de data before exp
    data2_transformed = np.matmul( data2.to_numpy(), pca1.transpose())
    data1_transformed = np.matmul( data.to_numpy(), pca1.transpose())

    # 3) Faire une pca sur cet ensemble de données reprojetées
    _, pca2, var2 = get_pca_frame(data2_transformed, nPCA)
    # print(data2_transformed)
    # print('PCA2 : ', pca2)
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig.suptitle('Dataset 2 reprojected')
    # ax.scatter(data2_transformed[:, 0], data2_transformed[:, 1], alpha=0.7, c='orange')
    # ax.scatter(data1_transformed[:, 0], data2_transformed[:, 1], alpha=0.05, c='blue')
    # print(pca2)
   

    # for i, (comp, var) in enumerate(zip(pca2, var2)):
    #     arrow_start = np.mean(data2_transformed, axis=0)  # Mean of dataset 1
    #     arrow_end = comp * var * 15 # Scale for visualization
        
    #     ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1],
    #             head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    #     plt.text(arrow_end[0] * 1.1, arrow_end[1] * 1.1, f"PC{i+1}", color='red', fontsize=12)
    # plt.axis('equal')
    # ax.set_xlabel("Principal Component 1")
    # ax.set_ylabel("Principal Component 2")
    # ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    # ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)

    # plt.grid(alpha=0.3)
    # plt.axis('equal')

    # plt.show()

    fig = plt.figure(figsize=(10,14))
    fig, ax = plt.subplots(3, nPCA, figsize=(15, 10))
    
    if absolute :
        plt.suptitle('Absolute values Reprojection \n Position = ' + str(position))
    else:
        plt.suptitle('Position = ' + str(position))
    for i in range(nPCA):
        ax[0, i].set_title(f"PC {i+1} {var1[i]:.2f}")
        ax[0, i].set_ylim([-0.5, 1.1])
        ax[0,i].bar(np.arange(len(pca1[i, :])), pca1[i, :])
    ax[0,0].set_ylabel(coord_metric.get_data_name())
    ax[1,0].set_ylabel(coord_metric_2.get_data_name())


    # 4) Remonter aux joints

    res = np.matmul(pca2, pca1)
    
    for i in range(nPCA):
        ax[1, i].set_title(f"PC {i+1} {var2[i]:.2f}")
        ax[1, i].set_ylim([-0.5, 1.1])
        ax[1, i].bar(np.arange(len(res[i, :])), res[i, :])

    if absolute :
        sub = np.absolute(res) - np.absolute(pca1)
        #sub = np.absolute(res-pca1)
    else:
        sub = res-pca1

    # print(res)
    for i in range(nPCA):
        ax[2, i].set_title(f"PC {i+1}")
        ax[2, i].bar(np.arange(len(sub[i, :])), sub[i, :], color='orange')
    ax[2,0].set_ylabel('Diff ' + coord_metric_2.get_data_name() +
                  '- ' + coord_metric.get_data_name())


    res_prop = np.array([sub[0, :] * var1[0], sub[1, :] * var1[1]]).flatten()
    if coord_metric.n_dof == 4 or emg==True:
        bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC1_4', 'PC2_1', 'PC2_2', 'PC2_3', 'PC2_4']
    elif coord_metric.n_dof == 3: 
        bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC2_1', 'PC2_2', 'PC2_3']
    elif coord_metric.n_dof == 2: 
        bins = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']
    elif coord_metric.n_dof == 5: 
        bins = ['PC1_1', 'PC1_2', 'PC1_3','PC1_4','PC1_5', 'PC2_1', 'PC2_2', 'PC2_3', 'PC2_4','PC2_5']
    # ax6.bar(bins, res_prop)
    # ax6.set_title('Difference reported to the explained variance')
    # ax4.set_ylim([-0.5, 1.1])
    # ax5.set_ylim([-0.5, 1.1])
    # ax6.set_ylim([-0.5, 1.1])

    max_x =max_y= min_x= min_y = 0
    for axs_row in ax:
        for axs in axs_row:
            lim_x, lim_y = max(axs.get_xlim()), max(axs.get_ylim())
            if lim_y > max_y:
                max_y = lim_y
            lim_x, lim_y = min(axs.get_xlim()), min(axs.get_ylim())
            if lim_y < min_y:
                min_y = lim_y
    plt.setp(fig.get_axes(), #xlim=(min_x - 0.2, max_x + 0.2),
      ylim=(min_y - 0.2, max_y + 0.2))
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter plot of PCA-transformed data
    # ax.scatter(dataA[:, 0], dataA[:, 1], alpha=0.7, c='blue', edgecolors='k')
    # ax.scatter(data22[:, 0], data22[:, 1], alpha=0.7, c='orange', edgecolors='k')

    # # Add arrows for principal components
    # for i, (comp, var) in enumerate(zip(pca1.components_, var1)):
    #     arrow_start = [0, 0]
    #     arrow_end = comp * var * 5  # Scale for visualization
    #     ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1],
    #              head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    #     plt.text(arrow_end[0] * 1.1, arrow_end[1] * 1.1, f"PC{i+1}", color='red', fontsize=12)
    
    # # Add labels and title
    # ax.set_xlabel("Principal Component 1")
    # ax.set_ylabel("Principal Component 2")
    # ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    # ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    # ax.set_title("PCA Plot with Arrows Showing Principal Components")
    
    # plt.grid(alpha=0.3)
    # plt.show()
    # pca_plot(coord_metric.get_concatenated_data(position=position)[coord_metric.get_n_columns_ang()], 
    #          coord_metric_2.get_concatenated_data(position=position)[coord_metric_2.get_n_columns_ang()], 
    #          pca1, sub, var1, var2, labels=coord_metric.get_n_columns_ang(), title = 'Reprojection')
    return pca1, res, sub, fig

def compute_pca_diff_no_reprojection(coord_metric, coord_metric_2, position=True, absolute=True, plot=False, nPCA=2):
    # test methode

  
    if position:
        columns = coord_metric.get_n_columns_ang()

    else:
        columns = coord_metric.get_n_columns_vel()

    # 1) PCA sur data before exp
    data = coord_metric.get_concatenated_data(position=position)[columns].dropna()
    data_centered1 , pca1, var1 = get_pca_frame(data.to_numpy(), nPCA)

        # print('PCA')
        # print(pca1)
        # print( coord_metric.get_concatenated_data(position=position)[columns])

    # 1) PCA sur data après exp (sans reprojections)
    data2 = coord_metric_2.get_concatenated_data(position=position)[columns].dropna()
    data_centered2, pca2, var2 = get_pca_frame(data2.to_numpy(), nPCA)

    if plot:
        fig, ax = plt.subplots()
        ax.set_rasterization_zorder(0)

        data = data -data.mean()        
        data22 = data2 -data2.mean()

        # ax.plot(data[columns[-1:]].values[:,0], data[columns[-2:]].values[:,0], '.', alpha=0.2, rasterized=True)
        # ax.plot(data22[columns[-1:]].values[:,0], data22[columns[-2:]].values[:,0], '.', c='g', alpha=0.2, rasterized=True)

        # ax.arrow(0,0, pca1[0,2], pca1[0,3])
        # ax.arrow(0,0, pca1[1,2], pca1[1,3])
        # ax.set_title('PC 1 : ' + str(pca1[0,:2]) + '\n PC 2 : ' + str(pca1[1,:2]))
        # ax.set_ylim([-0.5, 1.1])
        
    
    # 4) Remonter aux joints
    fig = plt.figure(figsize=(10,14))
    fig, ax = plt.subplots(3, nPCA, figsize=(15, 10))
    

    if absolute :
        plt.suptitle('Absolute values No Reprojection \n Position = ' + str(position))
    else:
        plt.suptitle('Position = ' + str(position))
    for i in range(nPCA):
        ax[0, i].set_title(f"PC {i+1} {var1[i]:.2f}")
        ax[0, i].set_ylim([-0.5, 1.1])
        ax[0,i].bar(np.arange(len(pca1[i, :])), pca1[i, :])
    ax[0,0].set_ylabel(coord_metric.get_data_name())
    ax[1,0].set_ylabel(coord_metric_2.get_data_name())

    for i in range(nPCA):
        ax[1, i].set_title(f"PC {i+1} {var2[i]:.2f}")
        ax[1, i].set_ylim([-0.5, 1.1])
        ax[1, i].bar(np.arange(len(pca2[i, :])), pca2[i, :])

    if absolute :
        sub = np.absolute(pca2) - np.absolute(pca1)
        sub = np.absolute(pca2-pca1)
    else:
        sub = pca2-pca1

    for i in range(nPCA):
        ax[2, i].set_title(f"PC {i+1}")
        ax[2, i].bar(np.arange(len(sub[i, :])), sub[i, :], color='orange')
    ax[2,0].set_ylabel('Diff ' + coord_metric_2.get_data_name() +
                  '- ' + coord_metric.get_data_name())

    res_prop = np.array([sub[0, :] * var1[0], sub[1, :] * var1[1]]).flatten()
    if coord_metric.n_dof == 4:
        bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC1_4', 'PC2_1', 'PC2_2', 'PC2_3', 'PC2_4']
    elif coord_metric.n_dof == 3: 
        bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC2_1', 'PC2_2', 'PC2_3']
    elif coord_metric.n_dof == 2: 
        bins = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']
    elif coord_metric.n_dof == 5: 
        bins = ['PC1_1', 'PC1_2', 'PC1_3','PC1_4','PC1_5', 'PC2_1', 'PC2_2', 'PC2_3', 'PC2_4','PC2_5']
    # ax6.bar(bins, res_prop)
    # ax6.set_title('Difference reported to the explained variance')
    # ax4.set_ylim([-0.5, 1.1])
    # ax5.set_ylim([-0.5, 1.1])
    # ax6.set_ylim([-0.5, 1.1])

    max_x =max_y= min_x= min_y = 0
    for axs_row in ax:
        for axs in axs_row:
            lim_x, lim_y = max(axs.get_xlim()), max(axs.get_ylim())
            # if lim_x > max_x:
            #     max_x = lim_x
            if lim_y > max_y:
                max_y = lim_y
            lim_x, lim_y = min(axs.get_xlim()), min(axs.get_ylim())
            # if lim_x < min_x:
            #     min_x = lim_x
            if lim_y < min_y:
                min_y = lim_y
    plt.setp(fig.get_axes(), #xlim=(min_x - 0.2, max_x + 0.2),
      ylim=(min_y - 0.2, max_y + 0.2))
    
    # pca_plot(coord_metric.get_concatenated_data(position=position)[coord_metric.get_n_columns_ang()], 
    #          coord_metric_2.get_concatenated_data(position=position)[coord_metric_2.get_n_columns_ang()], 
    #          pca1, pca2, var1, var2, labels=coord_metric.get_n_columns_ang(), title = 'No reprojection')

    return pca1, pca2, sub


# def compute_pca_reprojection_emg(coord_metric, coord_metric_2, absolute=True, plot=False):

#     # 1) Concatenate all EMG data
#     data1 = coord_metric.get_emg_data()
    
#     data2 = coord_metric2.get_emg_data()
    
#     df_data1 = pd.DataFrame()
#     for d in data1 :
#         df_data1 = pd.concat([df_data1, d[1]])
#     df_data1.dropna()
        
#     df_data2 = pd.DataFrame()
#     for d in data2 :
#         df_data2 = pd.concat([df_data2, d[1]])
#     df_data2.dropna()
        
        
    
#     pca1, var1 = get_pca_frame(data1.to_numpy())
#     # print('PCA')
#     # print(pca1)

#     # 2) Transform Data 2 into the reference frame of pca1
    
#     if plot:
#         fig, ax = plt.subplots()
#         ax.set_rasterization_zorder(0)

#         data = data -data.mean()        
#         data22 = data2 -data2.mean()
     
#     data2_transformed = np.matmul( data2.to_numpy(), pca1.transpose())
  
#     # 3) Compute a PCA on the Transformed data

#     pca2, var2 = get_pca_frame(data2_transformed)


#     # 4) Compute joints weight within this PC
    
#     fig = plt.figure(figsize=(10,14))

#     ax0 = fig.add_subplot(421)
#     ax1 = fig.add_subplot(422)
#     ax2 = fig.add_subplot(423)
#     ax3 = fig.add_subplot(424)
#     ax4 = fig.add_subplot(425)
#     ax5 = fig.add_subplot(426)
#     ax6 = fig.add_subplot(414)

#     ax=[ax0,ax1,ax2,ax3,ax4,ax5,ax6]

#     if absolute :
#         plt.suptitle('Absolute values \n Position = ' + str(position))
#     else:
#         plt.suptitle('Position = ' + str(position))
#     ax0.set_title(" PC1 %.2f" % var1[0])
#     ax1.set_title(" PC2 %.2f" % var1[1])
#     ax0.set_ylim([-0.5, 1.1])
#     ax1.set_ylim([-0.5, 1.1])

#     ax0.set_ylabel(coord_metric.get_data_name())
#     if absolute:
#         ax0.bar(np.arange(len(pca1[0, :])), np.absolute(pca1[0, :]))
#         ax1.bar(np.arange(len(pca1[1, :])), np.absolute(pca1[1, :]))
#     else :
#         ax0.bar(np.arange(len(pca1[0, :])), pca1[0, :])
#         ax1.bar(np.arange(len(pca1[1, :])), pca1[1, :])



#     if absolute :
#         res = np.absolute(np.matmul(pca2, pca1))

#     else :
#         res = np.matmul(pca2, pca1)

#     ax2.set_title('PC1')
#     ax2.set_ylabel(coord_metric_2.get_data_name())
#     ax2.bar(np.arange(len(res[0, :])), res[0, :])
#     ax3.set_title('PC2')
#     ax3.bar(np.arange(len(res[1, :])), res[1, :])
#     ax2.set_ylim([-0.5, 1.1])
#     ax3.set_ylim([-0.5, 1.1])

#     if absolute :
#         sub = res - np.absolute(pca1)
#     else:
#         sub = res-pca1

#     # print(res)
#     ax4.set_title('PC1')
#     ax4.set_ylabel('Diff ' + coord_metric_2.get_data_name() +
#                   '- ' + coord_metric.get_data_name())

#     ax4.bar(np.arange(len(sub[0, :])), sub[0, :], color='orange')
#     ax5.set_title('PC2')
#     ax5.bar(np.arange(len(sub[1, :])), sub[1, :], color='orange')

#     res_prop = np.array([sub[0, :] * var1[0], sub[1, :] * var1[1]]).flatten()
#     if coord_metric.n_dof == 4:
#         bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC1_4', 'PC2_1', 'PC2_2', 'PC2_3', 'PC2_4']
#     if coord_metric.n_dof == 3: 
#         bins = ['PC1_1', 'PC1_2', 'PC1_3', 'PC2_1', 'PC2_2', 'PC2_3']
#     if coord_metric.n_dof == 2: 
#         bins = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']
#     ax6.bar(bins, res_prop)
#     ax6.set_title('Difference reported to the explained variance')
#     ax4.set_ylim([-0.5, 1.1])
#     ax5.set_ylim([-0.5, 1.1])
#     ax6.set_ylim([-0.5, 1.1])

#     max_x =max_y= min_x= min_y = 0
#     for axs in ax:
#         lim_x, lim_y = max(axs.get_xlim()), max(axs.get_ylim())
       
#         if lim_y > max_y:
#             max_y = lim_y
#         lim_x, lim_y = min(axs.get_xlim()), min(axs.get_ylim())
       
#         if lim_y < min_y:
#             min_y = lim_y
#     plt.setp(fig.get_axes(),
#       ylim=(min_y - 0.2, max_y + 0.2))

#     return pca1, res, sub



def get_pca_frame(data, nPCA =2):
    # data_norm = StandardScaler().fit_transform(data)  # normalizing the features
    #center data
    data_norm = data - data.mean()
    plt.figure()
    print("Normalized data : ", data_norm)

    pca = PCA(n_components=nPCA)
    pca_data = pca.fit_transform(data_norm)

    return pca_data, pca.components_, pca.explained_variance_ratio_


def dist_sous_espace(u, v):
    """
    Compute Bockemühle distance between PCA subspace.

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    # print('dist')
    # print(np.transpose(u), v)
    # print(np.dot(u, np.transpose(v)))
    u, s, v = np.linalg.svd(np.dot(u, np.transpose(v)).reshape(1, 1))
    # print(u, s, v)

    s_min = np.min(s)
    # print(s_min)
    if s_min > 1:
        s_min = 1
    d = np.real(np.sqrt(1 - (s_min * s_min)))
    angle = math.asin(d)*180/math.pi
    # print("d Computation")
    # print(d)
    return d, angle


def get_pca_subspace(coord_metric):
    """
    Return the subspace defined by the eigenvectors of the covariance matrix.

    Returns
    -------
    eigenvectors : TYPE
        DESCRIPTION.

    """
    # cov = self.pca.get_covariance()
    # cov = coord_metric.get_pca().components_.transpose()
    data = coord_metric.compute_pca(plot=False)
    cov_list = []
    for d in data:
        cov_list.append(d.components_)
    # print(cov_list)
    # cov = cov[:, -1]
    # print('last cov')
    # print(cov)
    # cov = self.pca.get_covariance()

    # cov = cov / np.linalg.norm(cov, axis=0)
    return cov_list


def pca_plot(score1, score2 ,pca1, pca2, var1, var2, labels, title):
  # Scatter plot of the two datasets
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title) 
    X1 = (score1 - score1.mean()) 
    X2 = (score2 - score2.mean())
    ax[0].scatter(X1[labels[0]], X1[labels[1]], color='blue', alpha=0.5, label='Healthy')
    ax[0].scatter(X2[labels[0]], X2[labels[1]], color='orange', alpha=0.5, label='Pathological')
    ax[1].scatter(X1[labels[2]], X1[labels[3]], color='blue', alpha=0.5, label='Healthy')
    ax[1].scatter(X2[labels[2]], X2[labels[3]], color='orange', alpha=0.5, label='Pathological')
    # Plot PCA components for first dataset
    origin = np.mean(X1, axis=0)  # Mean of dataset 1
    for i, (length, vector) in enumerate(zip(var1, pca1)):
        ax[0].arrow(origin[0], origin[1], vector[0] * length *100, vector[1] * length *100, 
                color='cyan', width=0.02, head_width=0.1, alpha=0.8)
        ax[0].text(origin[0] + vector[0] * length*100, origin[1] + vector[1] * length*100, 
             f'PC{i+1}', color='cyan', fontsize=12, weight='bold')
    for i, (length, vector) in enumerate(zip(var2, pca2)):
        ax[0].arrow(origin[0], origin[1], vector[0] * length *100, vector[1] * length *100, 
                color='gold', width=0.02, head_width=0.1, alpha=0.8)
        ax[0].text(origin[0] + vector[0] * length*100, origin[1] + vector[1] * length*100, 
            f'PC{i+1}', color='gold', fontsize=12, weight='bold')
        
    for i, (length, vector) in enumerate(zip(var1, pca1)):
        ax[1].arrow(origin[2], origin[3], vector[2] * length *100, vector[3] * length *100, 
                color='cyan', width=0.02, head_width=0.1, alpha=0.8)
        ax[1].text(origin[2] + vector[2] * length*100, origin[3] + vector[3] * length*100, 
             f'PC{i+1}', color='cyan', fontsize=12, weight='bold')
    for i, (length, vector) in enumerate(zip(var2, pca2)):
        ax[1].arrow(origin[2], origin[3], vector[2] * length *100, vector[3] * length *100, 
                color='gold', width=0.02, head_width=0.1, alpha=0.8)
        ax[1].text(origin[2] + vector[2] * length*100, origin[3] + vector[3] * length*100, 
            f'PC{i+1}', color='gold', fontsize=12, weight='bold')


    # Labels and legend
    ax[0].set_xlabel('Theta '+str(labels[0]))
    ax[0].set_ylabel('Theta '+str(labels[1]))
    ax[0].legend()
    ax[0].grid()
    
    # Labels and legend
    ax[1].set_xlabel('Theta ' + str(labels[2]))
    ax[1].set_ylabel('Theta '+ str(labels[3]))
    ax[1].legend()
    ax[1].grid()
    plt.show()