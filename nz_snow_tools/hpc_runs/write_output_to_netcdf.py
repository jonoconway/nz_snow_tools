import xarray as xr
import numpy as np
import pickle
import datetime as dt

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
# run_id = 'dsc_default'  #'dsc_mueller_TF2p4_tmelt278_ros'  #
# which_model = 'dsc_snow'  # 'clark2009'  # 'dsc_snow'#
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO

output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
# model_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/nzcsm'


[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
# cut down model data to trimmed modis SI domain.

if modis_dem == 'modis_si_dem_250m':
    ann_scd = [trim_data_to_mask(a, mask) for a in ann_scd]
    ann_max_swe = [trim_data_to_mask(a, mask) for a in ann_max_swe]
    ann_av_swe = [trim_data_to_mask(a, mask) for a in ann_av_swe]
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)


def write_fields_to_netcdf(outfile, t_scd, t_lat, t_lon, t_northing, t_easting, t_dem):
    da_scd = xr.DataArray(
        data=t_scd.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'long_name': 'annual snow cover duration',
               'units': 'days per year', })
    da_lat = xr.DataArray(
        data=t_lat.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'latitude',
               'long_name': 'latitude of grid centre',
               'units': 'degrees north', })
    da_lon = xr.DataArray(
        data=t_lon.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'longitude',
               'long_name': 'longitude of grid centre',
               'units': 'degrees east', })
    da_north = xr.DataArray(
        data=t_northing.astype('f4'),
        dims=['northing'],
        attrs={'long_name': 'NZTM northing of grid centre',
               'units': 'm', })
    da_east = xr.DataArray(
        data=t_easting.astype('f4'),
        dims=['easting'],
        attrs={'long_name': 'NZTM easting of grid centre',
               'units': 'm', })
    da_dem = xr.DataArray(
        data=t_dem.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'surface_altitude',
               'long_name': 'height of surface above sea level',
               'units': 'm', })

    ds = xr.Dataset({'scd': da_scd,
                     'latitude': da_lat,
                     'longitude': da_lon,
                     'easting': da_east,
                     'northing': da_north,
                     'dem': da_dem,
                     },
                    attrs={
                        'title': 'This file contains snow cover duration data from the paper Conway, J., Carey-Smith, T., Cattoën, C., Moore, S., Sirguey, P., & Zammit, C. (2021). Simulations of seasonal snowpack duration and water storage across New Zealand. Weather and Climate, 41(1), 72-89. ',
                        'institution': 'NIWA',
                        'author': 'Jono Conway',
                        'email': 'jono.conway@niwa.co.nz',
                        'source': 'https://github.com/jonoconway/nz_snow_tools/blob/master/nz_snow_tools/hpc_runs/write_output_to_netcdf.py',
                        'file_creation_time': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

    ds.to_netcdf(outfile)



def write_fields_to_netcdf_small(outfile, t_scd, t_northing, t_easting, t_dem):
    da_scd = xr.DataArray(
        data=t_scd.astype('int'),
        dims=['northing', 'easting'],
        attrs={'long_name': 'annual snow cover duration',
               'units': 'days', })

    da_north = xr.DataArray(
        data=t_northing.astype('int'),
        dims=['northing'],
        attrs={'long_name': 'NZTM northing of grid centre',
               'units': 'm', })
    da_east = xr.DataArray(
        data=t_easting.astype('int'),
        dims=['easting'],
        attrs={'long_name': 'NZTM easting of grid centre',
               'units': 'm', })
    da_dem = xr.DataArray(
        data=t_dem.astype('int'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'surface_altitude',
               'long_name': 'height of surface above sea level',
               'units': 'm', })

    ds = xr.Dataset({'scd': da_scd,
                     'easting': da_east,
                     'northing': da_north,
                     'dem': da_dem,
                     },
                    attrs={
                        'title': 'This file contains snow cover duration data from the paper Conway, J., Carey-Smith, T., Cattoën, C., Moore, S., Sirguey, P., & Zammit, C. (2021). Simulations of seasonal snowpack duration and water storage across New Zealand. Weather and Climate, 41(1), 72-89. ',
                        'institution': 'NIWA',
                        'author': 'Jono Conway',
                        'email': 'jono.conway@niwa.co.nz',
                        'source': 'https://github.com/jonoconway/nz_snow_tools/blob/master/nz_snow_tools/hpc_runs/write_output_to_netcdf.py',
                        'file_creation_time': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

    ds.to_netcdf(outfile)
