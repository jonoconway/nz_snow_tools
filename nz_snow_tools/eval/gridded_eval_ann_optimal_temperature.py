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
    catchment = 'Clutha'  # string identifying catchment modelled
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = range(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    # modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
    dsc_snow_output_folder = 'D:/DSC-Snow/runs/output/clutha_nztm250m_erebus'
    mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Temp/DSC-Snow/Masks'
    catchment_shp_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Hydrology/Catchments'
    modis_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/MODIS_snow/NSDI_SI_cloudfilled'
    dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/'
    modis_dem = 'modis_si_dem_250m'
    met_inp_folder = 'D:/DSC-Snow/input_data_hourly'
    dsc_snow_dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/Projects-DSC-Snow/runs/input_DEM'
    output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/Projects-DSC-Snow/runs/output/clutha_nztm250m_erebus'
    plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/Projects-DSC-Snow/runs/output/clutha_nztm250m_erebus/plots'

    lists = [[] for _ in range(6)]
    a_obs, a_mod, a_ns, a_bias, a_rmse, a_mae = lists

    for tempchange in range(-5, 2, 1):
        run_id = 'norton_5_{}_topleft'.format(tempchange)  # string identifying fortran dsc_snow run. everything after the year

        ann = pickle.load(open(
            output_folder + '/resample_fit_{}_swe{}_{}_rs{}_smooth{}.pkl'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period), 'rb'))

        s_obs, s_mod, s_ns, s_bias, s_rmse, s_mae = ann

        a_obs.append(np.mean(np.asarray(s_obs), axis=0))
        a_mod.append(np.mean(np.asarray(s_mod), axis=0))
        a_ns.append(np.mean(np.asarray(s_ns), axis=0))
        a_bias.append(np.mean(np.asarray(s_bias), axis=0))
        a_rmse.append(np.mean(np.asarray(s_rmse), axis=0))
        a_mae.append(np.mean(np.asarray(s_mae), axis=0))


    # plot parameter set that gives lowest average absolute bias in FSCA
    cmap = plt.get_cmap('cubehelix',7)
    cmap.set_bad('0.5')
    plt.figure(figsize=(6,8))
    plot_var = np.argmin(np.abs(np.asarray(a_bias)), axis=0).astype(np.float32) - 5
    plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    plot_var[np.logical_and(a_obs[0] < 10,
                            ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    plt.imshow(plot_var, origin=0, vmax=1.5, vmin=-5.5, cmap=cmap)
    plt.yticks([]), plt.xticks([]), plt.title('T bias for lowest average absolute FSCA bias')
    plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_bias_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period), dpi=300)

    # plot parameter set that gives highest NS in FSCA
    cmap = plt.get_cmap('cubehelix',7)
    cmap.set_bad('0.5')
    plt.figure(figsize=(6,8))
    plot_var = np.argmax(np.asarray(a_ns), axis=0).astype(np.float32) - 5
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    plot_var[np.logical_and(a_obs[0] < 10,
                            ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    plt.imshow(plot_var, origin=0, vmax=1.5, vmin=-5.5, cmap=cmap)
    plt.yticks([]), plt.xticks([]), plt.title('T bias for highest average FSCA NS')
    cbar = plt.colorbar(label='T bias (C)')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_t_NS_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period), dpi=300)

    # plot lowest absolute bias in FSCA
    cmap = plt.get_cmap('viridis', 11)
    cmap.set_bad('0.5')
    plt.figure(figsize=(6,8))
    plot_var = np.min(np.abs(np.asarray(a_bias)), axis=0)
    plot_var[np.isnan(np.min(np.asarray(a_bias), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    plot_var[np.logical_and(a_obs[0] < 10,
                            ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    plt.imshow(plot_var, origin=0, cmap=cmap)  # vmax=1.5, vmin=-5.5,
    plt.yticks([]), plt.xticks([]), plt.title('lowest average FSCA bias')
    plt.colorbar(label='FSCA bias')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_bias_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period), dpi=300)

    # plot highest NS in FSCA
    cmap = plt.get_cmap('RdBu', 7)
    cmap.set_bad('0.5')
    # cmap.set_over('0')
    # cmap.set_under('0')
    plt.figure(figsize=(6,8))
    plot_var = np.max(np.asarray(a_ns), axis=0)
    plot_var[np.isnan(np.max(np.asarray(a_ns), axis=0))] = np.nan  # set values outside domain etc to nan
    # plot_var[a_obs[0] < 20] = np.nan
    plot_var[np.logical_and(a_obs[0] < 10,
                            ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    plt.imshow(plot_var, origin=0, vmax = 0.7, vmin=-0.7,cmap=cmap)
    plt.yticks([]), plt.xticks([]), plt.title('highest average FSCA NS')
    cbar = plt.colorbar(label='FSCA NS')
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/t_bias_optim_NS_{}_swe{}_{}_rs{}_smooth{}.png'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period), dpi=300)
    plt.close('all')
    # plt.figure()
    # plot_var = np.argmin(np.asarray(a_rmse), axis=0).astype(np.float32) - 5
    # plot_var[np.isnan(np.min(np.asarray(a_rmse), axis=0))] = np.nan  # set values outside domain etc to nan
    # # plot_var[a_obs[0] < 20] = np.nan
    # plot_var[np.logical_and(a_obs[0] < 10,
    #                         ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    # plt.imshow(plot_var, origin=0, vmax=1.5, vmin=-5.5, cmap=plt.cm.get_cmap('viridis', 7))
    # plt.yticks([]), plt.xticks([]), plt.title('RMSE')
    # plt.colorbar(label='T bias')
    #
    # plt.figure()
    # plot_var = np.argmin(np.asarray(a_mae), axis=0).astype(np.float32) - 5
    # plot_var[np.isnan(np.min(np.asarray(a_mae), axis=0))] = np.nan  # set values outside domain etc to nan
    # # plot_var[a_obs[0] < 20] = np.nan
    # plot_var[np.logical_and(a_obs[0] < 10,
    #                         ~(np.any(np.asarray(a_mod) > 10, axis=0)))] = np.nan  # exclude areas with very low modis fsca(< 10) and no modelled fsca(> 10)
    # plt.imshow(plot_var, origin=0, vmax=1.5, vmin=-5.5, cmap=plt.cm.get_cmap('viridis', 7))
    # plt.yticks([]), plt.xticks([]), plt.title('MAE')
    # plt.colorbar(label='T bias')
