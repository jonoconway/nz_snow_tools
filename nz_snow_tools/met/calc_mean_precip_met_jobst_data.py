"""
Generate hourly temperature and precipitation data for use in the snow model

The daily gridded fields from Andi Jobst at 250m for each calander year
plus vcsn radiation feilds
are downscaled to hourly using the same methods given in "Simulations of seasonal snow for the South Island, New Zealand"
Clark et al, 2009

"""

from __future__ import division

import datetime as dt
import netCDF4 as nc
import numpy as np
import pickle
import matplotlib.pylab as plt

import os
os.environ['PROJ_LIB']=r'C:\miniconda\envs\nz_snow27\Library\share'

import mpl_toolkits.basemap as basemap

from nz_snow_tools.util.utils import process_precip, process_temp, create_mask_from_shpfile, make_regular_timeseries, calc_toa, trim_lat_lon_bounds, \
    setup_nztm_dem

from nz_snow_tools.util.write_fsca_to_netcdf import write_nztm_grids_to_netcdf, setup_nztm_grid_netcdf

from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import load_new_vscn, interpolate_met, daily_to_hourly_temp_grids, daily_to_hourly_swin_grids




def load_jobst(variable, dts_to_take, nc_file_in, mask_dem,origin):
    nc_file = nc.Dataset(nc_file_in + '01-Jan-{}to31-dec-{}.nc'.format(dts_to_take[0].year, dts_to_take[0].year),'r')
    # nc_datetimes = nc.num2date(nc_file.variables['time'][:], nc_file.variables['time'].units)
    data = nc_file.variables[variable][:]
    # the data is in a funny grid - need to swap last two axes, then flip to align with vcsn grid

    # plt.imshow(np.flipud(np.transpose(hi_res_max_temp,(0,2,1))[0][mask]),origin=0)
    if mask_dem == True:
        hi_res_precip_trimmed = []
        for precip in data:
            if origin == 'topleft':
                _, _, trimmed_precip, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, np.transpose(precip), y_centres, x_centres)
            elif origin == 'bottomleft':
                _, _, trimmed_precip, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, np.flipud(np.transpose(precip)), y_centres, x_centres)
            hi_res_precip_trimmed.append(trimmed_precip)
        data = np.asarray(hi_res_precip_trimmed)

    if variable in ['tmin', 'tmax']:
        data = data + 273.15

    return data


if __name__ == '__main__':

    # dem control
    origin = 'topleft'  # orientation of output dem options are 'topleft' or 'bottomleft'. assume that input Dem is 'topleft'
    output_dem = 'clutha_dem_250m'  # identifier for output dem
    dem_file = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/clutha_dem_250m.tif'
    # mask control
    mask_dem = True  # boolean to set whether or not to mask the output dem - assume mask is bottom left
    catchment = 'Clutha'
    mask_created = True  # boolean to set whether or not the mask has already been created - assume mask is bottom left
    mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Temp/Masks'  # location of numpy catchment mask. must be writeable if mask_created == False
    mask_shpfile = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Hydrology/Catchments/{}.shp'.format(
        catchment)  # shapefile containing polyline or polygon of catchment in WGS84. Not needed if mask_created==True
    # time control
    years_to_take = range(2000, 2009 + 1)  # range(2001, 2013 + 1)
    ta_version = 'jobst_ucc'
    # input met data
    nc_file_rain = 'C:/Users/conwayjp/Documents/CLUTHA_PRECIP_250m_'  # provide only partial filename up to 01-Jan-2000to31-dec-2000
    # nc_file_tmax = 'E:/Data/JOBST_met_data/Tmax/CLUTHA_Tmax_250m_'
    # nc_file_tmin = 'E:/Data/JOBST_met_data/Tmin/CLUTHA_Tmin_250m_'
    # nc_file_srad = 'E:/Data/newVCSN/srad_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'

    # output met data
    met_out_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Temp/DSC-Snow/input_data_hourly'

    ####

    # set up input and output DEM for processing
    # output DEM
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, origin=origin)
    data_id = '{}_{}'.format(catchment, output_dem)  # name to identify the output data
    if mask_dem == True:
        # Get the masks for the individual regions of interest
        if mask_created == True:  # load precalculated mask
            mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, output_dem))
            if origin == 'topleft':
                mask = np.flipud(mask)
        else:  # create mask and save to npy file
            # masks = get_masks() #TODO set up for multiple masks
            mask = create_mask_from_shpfile(lat_array, lon_array, mask_shpfile)
            if origin == 'topleft':  # flip to save as bottom left
                mask = np.flipud(mask)
            np.save(mask_folder + '/{}_{}.npy'.format(catchment, output_dem), mask)
            if origin == 'topleft':  # flip back to topleft
                mask = np.flipud(mask)
        # Trim down the number of latitudes requested so it all stays in memory
        lats, lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)
        _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)
    else:
        mask = None
        lats = lat_array
        lons = lon_array
        elev = nztm_dem
        northings = y_centres
        eastings = x_centres

    ann_mean_precip = []

    for year_to_take in years_to_take:
        # load data
        # create timestamp to get - this is in NZST
        dts_to_take = np.asarray(make_regular_timeseries(dt.datetime(year_to_take, 1, 1), dt.datetime(year_to_take, 12, 31), 86400))
        # pull only data needed.
        # this loads data for 00h NZST that corresponds to the day to come in i.e. min@ 8am, max @ 2pm , total sw and total rain for 1/1/2000 at 2000-01-01 00:00:00
        # precip_daily = load_new_vscn('rain', dts_to_take, nc_file_rain)
        # max_temp_daily = load_new_vscn('tmax', dts_to_take, nc_file_tmax)
        # # min_temp_daily = load_new_vscn('tmin', dts_to_take, nc_file_tmin)
        # sw_rad_daily = load_new_vscn('srad', dts_to_take, nc_file_srad)
        # # load grid (assume same for all input data)
        # ds = nc.Dataset(nc_file_srad)
        # vcsn_elev = np.flipud(ds.variables['elevation'][:])
        # vcsn_lats = ds.variables['latitude'][::-1]
        # vcsn_lons = ds.variables['longitude'][:]
        # hy_index = np.ones(dts_to_take.shape, dtype='int')

        # check dimensions and projection of the new data
        # nztm_elev_check = interpolate_met(np.asarray([vcsn_elev]), Precip, vcsn_lons, vcsn_lats, vcsn_elev, lons,
        #                                     lats, elev)
        # plt.imshow(nztm_elev_check[0], origin=0)
        # plt.imshow(elev, origin=0)
        start_dt = dts_to_take[0]
        finish_dt = dts_to_take[-1]

        # interpolate data to fine grid
        hi_res_precip = load_jobst('precipitation', dts_to_take, nc_file_rain, mask_dem,origin=origin)
        # hi_res_max_temp = load_jobst('tmax', dts_to_take, nc_file_tmax, mask_dem,origin=origin)
        # hi_res_min_temp = load_jobst('tmin', dts_to_take, nc_file_tmin, mask_dem,origin=origin)
        # hi_res_sw_rad = interpolate_met(sw_rad_daily, 'srad', vcsn_lons, vcsn_lats, np.ma.fix_invalid(vcsn_elev).data, lons, lats, elev)
        p = np.sum(hi_res_precip,axis=0)
        ann_mean_precip.append(p)

    long_term_mean_precip = np.mean(np.asarray(ann_mean_precip),axis=0)

    plt.imshow(long_term_mean_precip)
    plt.colorbar()
    plt.show()

    for sce in ['RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5']:  #
        for v in ['P']:  #
            for year in ['2045', '2095']:  #
                diff = np.load(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\{}diff_{}_{}_mean_downscaled_v3.npy'.format(v, sce, year))

                percent_diff = diff / long_term_mean_precip
                plt.imshow(percent_diff)
                plt.colorbar()
                plt.show()
                np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\{}diff_{}_{}_mean_downscaled_v4.npy'.format(v, sce, year), percent_diff)
    #     #


        # if mask_dem:
        #     hi_res_precip[:, trimmed_mask == False] = np.nan
        #     hi_res_max_temp[:, trimmed_mask == False] = np.nan
        #     hi_res_min_temp[:, trimmed_mask == False] = np.nan
        #     hi_res_sw_rad[:, trimmed_mask == False] = np.nan
        #
        # # process and write
        # hourly_dt = np.asarray(make_regular_timeseries(start_dt + dt.timedelta(hours=1), finish_dt + dt.timedelta(days=1), 3600))
        # out_nc_file = setup_nztm_grid_netcdf(met_out_folder + '/met_inp_{}_{}_{}_{}.nc'.format(data_id, year_to_take, ta_version, origin),
        #                                      None, ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
        #                                      hourly_dt, northings, eastings, lats, lons, elev)
        # day_weightings = []
        # num_days = hi_res_precip.shape[0]
        # for i in range(num_days):
        #     # Do the temporal downsampling for one day
        #     # precip is random cascade for each day. NOTE original VCSN data has almost correct timestamp - ie. total from 9am.
        #     hourly_precip, day_weightings_1 = process_precip(hi_res_precip[i],
        #                                                      one_day=True)  # TODO: align to 9am-9am - currently counts pretends it is midnight-midnight
        #     # air temperature is three part sinusoidal between min at 8am and max at 2pm. NOTE original VCSN data has correct timestamp - ie. minimum to 9am, maximum from 9am.
        #     # hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i], hi_res_min_temp[i], single_dt=True)  #
        #     if i == 0:
        #         hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i:i + 2], hi_res_min_temp[i:i + 2])
        #         hourly_temp = hourly_temp[:24]
        #     elif i == num_days:
        #         hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i - 1:], hi_res_min_temp[i - 1:])
        #         hourly_temp = hourly_temp[-24:]
        #     else:
        #         hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i - 1:i + 2], hi_res_min_temp[i - 1:i + 2])
        #         hourly_temp = hourly_temp[24:48]
        #     #
        #     hourly_swin = daily_to_hourly_swin_grids(hi_res_sw_rad[i], lats, lons, hourly_dt[i * 24: (i + 1) * 24], single_dt=True)
        #
        #     for var, data in zip(['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
        #                          [hourly_temp, hourly_precip, hourly_swin]):
        #         out_nc_file.variables[var][i * 24: (i + 1) * 24, :, :] = data
        #     day_weightings.extend(day_weightings_1)
        # out_nc_file.close()
        #
        # pickle.dump(day_weightings, open(met_out_folder + '/met_inp_{}_{}_{}_{}_daywts.pkl'.format(data_id, year_to_take, ta_version, origin), 'wb'), -1)
