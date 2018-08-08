"""
code to plot maps of snow covered area for individual years from summary lists generated by catchment_evalutation.py
"""

from __future__ import division

import numpy as np
import pickle
import matplotlib.pylab as plt

average_scd = True # boolean specifying if all years are to be averaged together - now plots difference between
which_model = 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
clark2009run = True  # boolean specifying if the run already exists
dsc_snow_opt = 'python'  # string identifying which version of the dsc snow model to use output from 'python' or 'fortran'
catchment = 'Clutha'
output_dem = 'nztm250m'  # identifier for output dem
hydro_years_to_take = range(2001, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered
model_output_folder = 'D:/Projects-DSC-Snow/nz_snow_runs/baseline_clutha1'
plot_folder = 'P:/Projects/DSC-Snow/nz_snow_runs/baseline_clutha2'

ann = pickle.load(open(model_output_folder + '/summary_{}_{}_thres{}_swe{}.pkl'.format(catchment, output_dem, modis_sc_threshold,model_swe_sc_threshold), 'rb'))
# indexes 0-3 modis, 4-8 model 1 and 9-13 model 2
# ann = [ann_ts_av_sca_m, ann_hydro_days_m, ann_dt_m, ann_scd_m,
# ann_ts_av_sca, ann_ts_av_swe, ann_hydro_days, ann_dt, ann_scd,
# ann_ts_av_sca2, ann_ts_av_swe2, ann_hydro_days2, ann_dt2, ann_scd2]

ann_scd_m = np.asarray(ann[3],dtype=np.double)
ann_scd = np.asarray(ann[8],dtype=np.double)
ann_scd2 = np.asarray(ann[13],dtype=np.double)
# two years of data are bad

ann_scd_m[5] = np.nan
ann_scd_m[10] = np.nan


fig1 = plt.figure(figsize=[10,4])

# fig, axes = plt.subplots(nrows=1, ncols=3,figsize=[8,3])
# for ax in axes.flat:

if average_scd ==True:
    modis_scd = np.nanmean(ann_scd_m, axis=0)
    mod1_scd = np.nanmean(ann_scd, axis=0)
    mod1_scd[(mod1_scd==-999)]= np.nan

    if which_model == 'all':
        mod2_scd = np.mean(ann_scd2, axis=0)

    plt.subplot(1, 3, 1)
    plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
    plt.colorbar()
    plt.title('modis duration fsca > {}'.format(modis_sc_threshold))

    plt.subplot(1, 3, 2)
    plt2 = mod1_scd - modis_scd
    plt2[np.logical_or(modis_scd==-1,mod1_scd==-999)]=0
    plt2[(plt2 > 100)] = 0# some bad model points
    plt.imshow(plt2, origin=0, interpolation='none',vmin=-100, vmax=100,cmap=plt.cm.RdBu)#, vmin=0, vmax=365, cmap='viridis')
    # plt.imshow(plt2, origin=0, interpolation='none',vmin=-1 * np.nanmax(np.abs(plt2)), vmax=np.nanmax(np.abs(plt2)),cmap=plt.cm.RdBu)#, vmin=0, vmax=365, cmap='viridis')
    plt.colorbar()
    plt.title('difference (days)')
    plt.subplot(1, 3, 3)
    plt.imshow(mod1_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
    plt.colorbar()
    plt.title('dsc_snow duration')

    # if which_model != 'all':
    #     plt.title('{} duration'.format(which_model))
    # if which_model == 'all':
    #     #plt.title('clark2009 duration')
    #     plt.title('difference (days)')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(mod1_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
    #     plt.colorbar()
    #     plt.title('dsc_snow duration')
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCD hy{} to hy{}.png'.format(hydro_years_to_take[0],hydro_years_to_take[-1]), dpi=300)
else:
    for i, hydro_year_to_take in enumerate(hydro_years_to_take):
        modis_scd = ann_scd_m[i]
        mod1_scd = ann_scd[i]
        if which_model == 'all':
            mod2_scd = ann_scd2[i]

        plt.subplot(1, 3, 1)
        h = plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
        plt.title('modis fsca >= {}'.format(modis_sc_threshold))
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.imshow(mod1_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
        if which_model != 'all':
            plt.title('{} duration'.format(which_model))
            plt.tight_layout()

            plt.subplots_adjust(right=0.8)
            cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
            plt.colorbar(h, cax=cbar_ax)

        if which_model == 'all':
            plt.title('clark2009')
            plt.tight_layout()

            plt.subplot(1, 3, 3)
            plt.imshow(mod2_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
            #plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.title('dsc_snow')
            plt.tight_layout()

        plt.subplots_adjust(right=0.8)
        cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(h, cax=cbar_ax)

        # plt.subplot(1, 4, 4,frameon=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar(h)
        # plt.tight_layout()
        plt.savefig(plot_folder + '/SCD hy{}_{}_{}_swe{}.png'.format(hydro_year_to_take,catchment, output_dem,model_swe_sc_threshold), dpi=300)
        plt.clf()
