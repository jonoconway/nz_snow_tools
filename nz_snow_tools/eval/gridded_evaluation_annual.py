"""
code to evaluate snow models on grid, pixel by pixel
options to call a series of models then compute summary statistics
reads in a computes statistics on MODIS data to evaluate against

requires that dsc_snow model has been pre run either using Fortran version or using run_snow_model
the Clark2009 model can be run on-the-fly or prerun

Jono Conway
"""
from __future__ import division

import matplotlib.pylab as plt
from nz_snow_tools.eval.catchment_evaluation import *
from nz_snow_tools.eval.catchment_evaluation_annual import load_dsc_snow_output_annual, load_subset_modis_annual
from nz_snow_tools.util.utils import resample_to_fsca, nash_sut, mean_bias, rmsd, mean_absolute_error

def plot_point(i,j,name,year):
    plt.figure()
    plt.plot(np.convolve(model_fsca_rs[:, i, j], np.ones((smooth_period,)) / smooth_period, mode='same'))
    plt.plot(np.convolve(modis_fsca_rs[:, i, j], np.ones((smooth_period,)) / smooth_period, mode='same'))
    plt.ylabel('fsca (%)')
    plt.xlabel('day of year')
    plt.title(name)
    plt.savefig('P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus/plots/timeseries/timeseries_{}_{}_{}_{}.png'.format(name,year,smooth_period,run_id))
    plt.close()

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


    for year_to_take in years_to_take:

        print('loading modis data {}'.format(year_to_take))

        # load modis data for evaluation
        modis_fsca, modis_dt, modis_mask = load_subset_modis_annual(catchment, output_dem, year_to_take, modis_folder, dem_folder, modis_dem, mask_folder,
                                                                    catchment_shp_folder)

        # set up output array
        nt = modis_fsca.shape[0]
        ny = modis_fsca.shape[1]
        nx = modis_fsca.shape[2]
        ny_out = ny // rl  # integer divide to ensure fits
        nx_out = nx // rl
        modis_fsca_rs = np.zeros((nt, ny_out, nx_out))

        for i in range(nt):
            modis_sub = modis_fsca[i,]
            fsca_rs = resample_to_fsca(modis_sub, rl=rl)
            modis_fsca_rs[i] = fsca_rs

        # load model data
        for tempchange in [-2, 0]:
            for precipchange in [20, 50, 100]:

                # if tempchange == 0 and precipchange in [50,100]:
                #     pass
                # else:
                s_ns = []
                s_bias = []
                s_rmse = []
                s_mae = []
                s_obs = []
                s_mod = []

                run_id = 'norton_5_t{}_p{}_topleft'.format(tempchange,precipchange)  # string identifying fortran dsc_snow run. everything after the year
                # recipie
                # read in modis and model data for one year
                # average to large spatial scale
                # compare timeseries of fsca at each point
                # store statistics - for each point for each year dims = [year,y,x]

                print('loading dsc_snow model data {}'.format(year_to_take))
                # load previously run simulations from netCDF
                st_swe, _, _, out_dt, mask = load_dsc_snow_output_annual(catchment, output_dem, year_to_take, dsc_snow_output_folder,
                                                                                    dsc_snow_dem_folder, run_id, origin=origin)

                if year_to_take == 2000:
                    # cut so that first day corresponds to first MODIS obs on 24th Feb i.e. 2000-02-25 00:00:00
                    st_swe = st_swe[54:, ]
                    # st_melt = st_melt[54:, ]
                    # st_acc = st_acc[54:, ]
                    out_dt = out_dt[54:]

                st_sc = np.zeros(st_swe.shape, dtype=np.float32)
                st_sc[st_swe > model_swe_sc_threshold] = 100
                st_sc[:, mask == False] = np.nan
                model_fsca_rs = np.zeros((nt, ny_out, nx_out))

                for i in range(nt):
                    model_sub = st_sc[i,]
                    fsca_rs = resample_to_fsca(model_sub, rl=rl)
                    model_fsca_rs[i] = fsca_rs

                # plt.plot(np.nanmean(model_fsca_rs, axis=(1, 2)))
                #
                # plt.figure()
                # plt.imshow(np.mean(modis_fsca_rs, axis=0),origin=0)
                # plt.figure()
                # plt.imshow(np.mean(model_fsca_rs, axis=0),origin=0)

                ns_array = np.zeros((ny_out, nx_out))
                mbd_array = np.zeros((ny_out, nx_out))
                rmsd_array = np.zeros((ny_out, nx_out))
                mae_array = np.zeros((ny_out, nx_out))


                for i in range(ny_out):
                    for j in range(nx_out):
                        obs = np.convolve(modis_fsca_rs[:, i, j], np.ones((smooth_period,)) / smooth_period, mode='same')
                        mod = np.convolve(model_fsca_rs[:, i, j], np.ones((smooth_period,)) / smooth_period, mode='same')
                        ns_array[i, j] = nash_sut(mod, obs)
                        mbd_array[i, j] = mean_bias(mod, obs)
                        rmsd_array[i, j] = rmsd(mod, obs)
                        mae_array[i, j] = mean_absolute_error(mod, obs)

                modis_mean = np.mean(modis_fsca_rs, axis=0)
                model_mean = np.mean(model_fsca_rs, axis=0)

                s_ns.append(ns_array)
                s_bias.append(mbd_array)
                s_rmse.append(rmsd_array)
                s_mae.append(mae_array)
                s_obs.append(modis_mean)
                s_mod.append(model_mean)

                for i,j,name in zip([161,147,127,107,186,125],[83,102,59,88,21,34],['Pisa','Dunstan','Hector','Old Man','Earnslaw','Lochy']):
                    plot_point(i,j,name,year_to_take)

                ann = [s_obs, s_mod, s_ns, s_bias, s_rmse, s_mae]
                pickle.dump(ann, open(
                    output_folder + '/resample_fit_{}_swe{}_{}_rs{}_smooth{}_{}.pkl'.format(catchment, model_swe_sc_threshold, run_id, rl,smooth_period,year_to_take),
                    'wb'), -1)

                st_swe, st_sc = None, None

        # reset variables to save space
        modis_fsca =  None