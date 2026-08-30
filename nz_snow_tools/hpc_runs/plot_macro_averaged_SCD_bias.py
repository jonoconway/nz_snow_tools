import numpy as np
import pickle
import copy
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import datetime as dt
from nz_snow_tools.util.utils import convert_datetime_julian_day
from nz_snow_tools.util.utils import setup_nztm_dem, trim_data_to_mask, trim_lat_lon_bounds

#TODO # todos indicate which parameters need to change to switch between VCSN and NZCSM
hydro_years_to_take = np.arange(2018, 2020 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/NZ/august2021' #TODO
# plot_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz'
# model_analysis_area = 145378  # sq km.
catchment = 'NZ'  # string identifying catchment modelled #TODO
mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'
modis_dem = 'modis_nz_dem_250m' #TODO

if modis_dem == 'modis_si_dem_250m':

    si_dem_file = dem_folder + '/si_dem_250m' + '.tif'
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(si_dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250)
    nztm_dem = nztm_dem[:, 20:]
    x_centres = x_centres[20:]
    lat_array = lat_array[:, 20:]
    lon_array = lon_array[:, 20:]
    modis_output_dem = 'si_dem_250m'
    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, modis_dem))

elif modis_dem == 'modis_nz_dem_250m':
    si_dem_file = dem_folder + '/nz_dem_250m' + '.tif'
    _, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                   resolution=250, origin='bottomleft')
    nztm_dem = np.load(dem_folder + '/{}.npy'.format(modis_dem))
    modis_output_dem = 'modis_nz_dem_250m'
    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment,
                                                     modis_dem))  # just load the mask the chooses land points from the dem. snow data has modis hy2018_2020 landpoints mask applied in NZ_evaluation_otf
    # mask = np.load("C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/modis_mask_hy2018_2020_landpoints.npy")

lat_array, lon_array, nztm_dem, y_centres, x_centres = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)

# # modis options
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
modis_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
# modis_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, modis_output_dem,
                                                                          modis_sc_threshold), 'rb'))
# model options

run_id = 'cl09_default_ros'  ## 'cl09_tmelt275'#'cl09_default' #'cl09_tmelt275_ros' ##TODO
which_model = 'clark2009'  #TODO
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO
output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)

h2d = plt.hist2d(modis_scd.ravel()[~np.isnan(modis_scd.ravel())], nztm_dem.ravel()[~np.isnan(modis_scd.ravel())],
                 bins=(np.arange(0, 367, 0.1), np.arange(0, 4000, 200)))
scd_elev_modis = np.sum(h2d[0]*h2d[1][1:,np.newaxis],axis=0)/np.sum(h2d[0],axis=0)

h2d_mod = plt.hist2d(model_scd.ravel()[~np.isnan(model_scd.ravel())], nztm_dem.ravel()[~np.isnan(model_scd.ravel())],
                 bins=(np.arange(0, 367, 0.1), np.arange(0, 4000, 200)))
scd_elev_mod = np.sum(h2d_mod[0]*h2d_mod[1][1:,np.newaxis],axis=0)/np.sum(h2d_mod[0],axis=0)

run_id = 'cl09_default_ros'  ## 'cl09_tmelt275'#'cl09_default' #'cl09_tmelt275_ros' ##TODO
which_model = 'clark2009'  #TODO
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO
output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)

h2d_mod_30 = plt.hist2d(model_scd.ravel()[~np.isnan(model_scd.ravel())], nztm_dem.ravel()[~np.isnan(model_scd.ravel())],
                 bins=(np.arange(0, 367, 0.1), np.arange(0, 4000, 200)))
scd_elev_mod_30 = np.sum(h2d_mod_30[0]*h2d_mod_30[1][1:,np.newaxis],axis=0)/np.sum(h2d_mod_30[0],axis=0)

run_id = 'cl09_AAA_ros'  ## 'cl09_tmelt275'#'cl09_default' #'cl09_tmelt275_ros' ##TODO
which_model = 'clark2009'  #TODO
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO
output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)

h2d_mod_30_AAA= plt.hist2d(model_scd.ravel()[~np.isnan(model_scd.ravel())], nztm_dem.ravel()[~np.isnan(model_scd.ravel())],
                 bins=(np.arange(0, 367, 0.1), np.arange(0, 4000, 200)))
scd_elev_mod_30_AAA = np.sum(h2d_mod_30_AAA[0]*h2d_mod_30_AAA[1][1:,np.newaxis],axis=0)/np.sum(h2d_mod_30_AAA[0],axis=0)

run_id = 'cl09_AAA_ros'  ## 'cl09_tmelt275'#'cl09_default' #'cl09_tmelt275_ros' ##TODO
which_model = 'clark2009'  #TODO
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO
output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)

h2d_mod_5_AAA= plt.hist2d(model_scd.ravel()[~np.isnan(model_scd.ravel())], nztm_dem.ravel()[~np.isnan(model_scd.ravel())],
                 bins=(np.arange(0, 367, 0.1), np.arange(0, 4000, 200)))
scd_elev_mod_5_AAA = np.sum(h2d_mod_5_AAA[0]*h2d_mod_5_AAA[1][1:,np.newaxis],axis=0)/np.sum(h2d_mod_5_AAA[0],axis=0)

np.mean(scd_elev_mod)
np.mean(scd_elev_modis)
np.mean(scd_elev_mod_30)
np.mean(scd_elev_mod_30_AAA)
np.mean(scd_elev_mod_5_AAA)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize': 6})
fig1 = plt.figure(figsize=[4, 4])

plt.plot(scd_elev_modis,np.arange(100,3800,200),'k',label="MODIS")
plt.plot(scd_elev_mod_30,np.arange(100,3800,200),'--',label="Default run, 30 mm")
plt.plot(scd_elev_mod,np.arange(100,3800,200),label="Default run, 5 mm")
plt.plot(scd_elev_mod_30_AAA,np.arange(100,3800,200),'--',label="Sensitivity run, 30 mm")
plt.plot(scd_elev_mod_5_AAA,np.arange(100,3800,200),label="Sensitivity run, 5 mm")
plt.legend()

plt.ylabel('Elevation (m)')
plt.xlabel('SCD (days)')
plt.yticks(np.arange(0,3800,200))
plt.xticks(np.arange(0,390,30))
plt.tight_layout()
plt.title('(d) Mean SCD by elevation',fontweight='bold',fontsize=10)
plt.tight_layout()
fig1.savefig(
    plot_folder + '/line elevation fit HY{}to{} thres{} {}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                                       run_id),
    dpi=300)
fig1.savefig(
    plot_folder + '/line elevation fit HY{}to{} thres{} {}.pdf'.format(hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                                       run_id),
    dpi=300)