"""
code to evaluate snow models on grid, pixel by pixel
options to call a series of models then compute summary statistics
reads in a computes statistics on MODIS data to evaluate against
requires that dsc_snow model has been pre run using Fortran version

Jono Conway
"""
from __future__ import division

import matplotlib.pylab as plt
from nz_snow_tools.eval.catchment_evaluation import *
from nz_snow_tools.eval.catchment_evaluation_annual import load_dsc_snow_output_annual, load_subset_modis_annual
from nz_snow_tools.util.utils import resample_to_fsca, nash_sut, mean_bias, rmsd, mean_absolute_error

if __name__ == '__main__':

    rl = 4  # resample length (i.e. how many grid cells in each direction to resample.
    smooth_period = 10  # number of days to smooth model data
    origin = 'topleft'
    catchment = 'SouthIsland'  # string identifying catchment modelled
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = range(2001, 2005 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
    output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/from Ruschle/eval'
    plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/from Ruschle/eval/plots_jono'

    lists = [[] for _ in range(6)]
    a_obs, a_mod, a_ns, a_bias, a_rmse, a_mae = lists

    for tempchange in [0, -1, -2]: # NOTE THE PLOTTING OF TBIAS IS VERY DEPENDENT on the ORDER used here
        for precipchange in [0]:

            s_ns = []
            s_bias = []
            s_rmse = []
            s_mae = []
            s_obs = []
            s_mod = []

            run_id = 'norton_5_t{}_p{}_topleft'.format(tempchange, precipchange)  # string identifying fortran dsc_snow run. everything after the year

            for year_to_take in years_to_take:
                ann = pickle.load(open(
                    output_folder + '/resample_fit_{}_swe{}_{}_rs{}_smooth{}_{}.pkl'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period,
                                                                                            year_to_take), 'rb'))  # ,encoding='latin1'

                #
                s_obs.append(ann[0][0])
                s_mod.append(ann[1][0])
                s_ns.append(ann[2][0])
                s_bias.append(ann[3][0])
                s_rmse.append(ann[4][0])
                s_mae.append(ann[5][0])

            # s_obs, s_mod, s_ns, s_bias, s_rmse, s_mae = ann

            a_obs.append(np.mean(np.asarray(s_obs), axis=0))
            a_mod.append(np.mean(np.asarray(s_mod), axis=0))
            a_ns.append(np.mean(np.asarray(s_ns), axis=0))
            a_bias.append(np.mean(np.asarray(s_bias), axis=0))
            a_rmse.append(np.mean(np.asarray(s_rmse), axis=0))
            a_mae.append(np.mean(np.asarray(s_mae), axis=0))

            # plot mean bias in fsca for each option
            cmap = plt.get_cmap('RdBu')
            cmap.set_bad('0.5')  # set bad values to grey colour
            plt.figure(figsize=(8, 8))
            plot_var = np.mean(np.asarray(s_mod) - np.asarray(s_obs), axis=0)  # set to number of simulation
            plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
            # # plot_var[a_obs[0] < 20] = np.nan
            # plot_var[np.logical_and(a_obs[0] < 10,
            #                         ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
            plt.imshow(plot_var, origin=0, vmax=60, vmin=-60, cmap=cmap, interpolation='none')
            plt.colorbar(label='fsca bias (%fsca)')
            plt.yticks([]), plt.xticks([]), plt.title('model-obs mean FSCA t{} p{}'.format(tempchange, precipchange))
            plt.tight_layout()
            plt.savefig(
                plot_folder + '/mean_fsca_bias_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

            cmap = plt.get_cmap('cubehelix')
            cmap.set_bad('0.5')  # set bad values to grey colour
            plt.figure(figsize=(8, 8))
            plot_var = np.mean(np.asarray(s_mod), axis=0)  # set to number of simulation
            plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
            # # plot_var[a_obs[0] < 20] = np.nan
            # plot_var[np.logical_and(a_obs[0] < 10,
            #                         ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
            plt.imshow(plot_var, origin=0, vmax=100, vmin=0, cmap=cmap, interpolation='none')
            plt.colorbar(label='fsca %')
            plt.yticks([]), plt.xticks([]), plt.title('model mean FSCA t{} p{}'.format(tempchange, precipchange))
            plt.tight_layout()
            plt.savefig(
                plot_folder + '/mean_model_fsca_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

            # plot modis mean
            cmap = plt.get_cmap('cubehelix')
            cmap.set_bad('0.5')  # set bad values to grey colour
            plt.figure(figsize=(8, 8))
            plot_var = np.mean(np.asarray(s_obs), axis=0)  # set to number of simulation
            plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
            # # plot_var[a_obs[0] < 20] = np.nan
            # plot_var[np.logical_and(a_obs[0] < 10,
            #                         ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
            plt.imshow(plot_var, origin=0, vmax=100, vmin=0, cmap=cmap, interpolation='none')
            plt.colorbar(label='fsca %')
            plt.yticks([]), plt.xticks([]), plt.title('modis mean FSCA t{} p{}'.format(tempchange, precipchange))
            plt.tight_layout()
            plt.savefig(
                plot_folder + '/mean_modis_fsca_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

    # plot parameter set that gives lowest average absolute bias in FSCA
    cmap = plt.get_cmap('cubehelix', 3)
    cmap.set_bad('0.5')
    plt.figure(figsize=(8, 8))
    plot_var = np.argmin(np.abs(np.asarray(a_bias)), axis=0).astype(np.float32) * -1  # set to number of simulation
    plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    # exclude areas with very low modis fsca(< 5) and no modelled fsca(> 5) in any year
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan
    plt.imshow(plot_var, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.colorbar(label='T bias (C)')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for lowest average absolute FSCA bias')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_bias_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

    # plot parameter set that gives highest NS in FSCA
    cmap = plt.get_cmap('cubehelix', 3)
    cmap.set_bad('0.5')
    plt.figure(figsize=(8, 8))
    plot_var = np.argmax(np.asarray(a_ns), axis=0).astype(np.float32) * -1
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    # exclude areas with very low modis fsca(< 5) and no modelled fsca(> 5) in any year
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan
    plt.imshow(plot_var, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

    # plot lowest absolute bias in FSCA
    cmap = plt.get_cmap('viridis', 11)
    cmap.set_bad('0.5')
    plt.figure(figsize=(8, 8))
    plot_var = np.min(np.abs(np.asarray(a_bias)), axis=0)
    plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    # exclude areas with very low modis fsca(< 5) and no modelled fsca(> 5) in any year
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan
    plt.imshow(plot_var, origin=0, cmap=cmap, interpolation='none')  # vmax=1.5, vmin=-5.5,
    plt.yticks([]), plt.xticks([]), plt.title('lowest average absolute FSCA bias')
    plt.colorbar(label='absolute FSCA bias (%)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_bias_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

    # plot highest NS in FSCA
    cmap = plt.get_cmap('RdBu', 7)
    cmap.set_bad('0.5')
    # cmap.set_over('0')
    # cmap.set_under('0')
    plt.figure(figsize=(8, 8))
    plot_var = np.max(np.asarray(a_ns), axis=0)
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    # exclude areas with very low modis fsca(< 5) and no modelled fsca(> 5) in any year
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan
    plt.imshow(plot_var, origin=0, vmax=0.7, vmin=-0.7, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('highest average FSCA NS')
    cbar = plt.colorbar(label='FSCA NS')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_NS_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period), dpi=300)

    cmap = plt.get_cmap('cubehelix', 3)
    cmap.set_bad('0.5')

    # generate a 250m grid of bias and save. set areas previously nan to 0 bias
    plot_var = np.argmax(np.asarray(a_ns), axis=0).astype(np.float32) * -1
    plot_var[plot_var == -0] = 0  # _all
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # _all_valid
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # trimmed
    final_grid = np.zeros((2800, 2540))
    for i in range(plot_var.shape[0]):
        for j in range(plot_var.shape[1]):
            final_grid[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = plot_var[i, j]
    final_grid2 = np.zeros((2800, 2560))
    final_grid2[:, 20:] = final_grid

    plt.figure(figsize=(8, 8))
    plt.imshow(final_grid2, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed_fullres.png'.format(catchment, model_swe_sc_threshold, run_id, rl,
                                                                                                smooth_period), dpi=300)

    # reset nan to 0 and dump
    final_grid2[np.isnan(final_grid2)] = 0  # _all
    pickle.dump(final_grid2, open(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed_fullres.pkl'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period),
        'wb'), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(final_grid2, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed_fullres_nonan.png'.format(catchment, model_swe_sc_threshold, run_id, rl,
                                                                                                      smooth_period), dpi=300)

    # generate a 250m grid of bias and save. set areas previously nan to 0 bias - trim to only areas with NS>0
    plot_var = np.argmax(np.asarray(a_ns), axis=0).astype(np.float32) * -1
    plot_var[plot_var == -0] = 0  # _all
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # _all_valid
    plot_var[np.max(np.asarray(a_ns), axis=0) <= 0] = np.nan  # only areas with ok model performance
    plot_var[np.logical_and(a_obs[0] < 10, ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # trimmed
    final_grid = np.zeros((2800, 2540))
    for i in range(plot_var.shape[0]):
        for j in range(plot_var.shape[1]):
            final_grid[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = plot_var[i, j]
    final_grid2 = np.zeros((2800, 2560))
    final_grid2[:, 20:] = final_grid

    # plot
    plt.figure(figsize=(8, 8))
    plt.imshow(final_grid2, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed2_fullres.png'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period),
        dpi=300)

    # reset nan to 0 and dump
    final_grid2[np.isnan(final_grid2)] = 0  # _all
    pickle.dump(final_grid2, open(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed2_fullres.pkl'.format(catchment, model_swe_sc_threshold, run_id, rl, smooth_period),
        'wb'), -1)

    # plot
    plt.figure(figsize=(8, 8))
    plt.imshow(final_grid2, origin=0, vmax=0.5, vmin=-2.5, cmap=cmap, interpolation='none')
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}_trimmed2_fullres_nonan.png'.format(catchment, model_swe_sc_threshold, run_id, rl,
                                                                                                       smooth_period), dpi=300)

