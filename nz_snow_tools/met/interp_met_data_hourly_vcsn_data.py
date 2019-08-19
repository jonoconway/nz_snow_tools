"""
Generate hourly temperature and precipitation data for use in the snow model

The daily gridded fields (e.g. from VCSN) are downscaled using the same methods given in "Simulations of seasonal snow for the South Island, New Zealand"
Clark et al, 2009

"""

from __future__ import division

import datetime as dt
import netCDF4 as nc
import numpy as np
import pickle

import os  # need to import this as basemap is a bit broken https://github.com/conda-forge/basemap-feedstock/issues/30

os.environ['PROJ_LIB'] = r'C:\miniconda\envs\nz_snow27\Library\share'
import mpl_toolkits.basemap as basemap
from scipy import interpolate
from nz_snow_tools.util.utils import process_precip, process_temp, create_mask_from_shpfile, make_regular_timeseries, calc_toa, trim_lat_lon_bounds, \
    setup_nztm_dem

from nz_snow_tools.util.write_fsca_to_netcdf import write_nztm_grids_to_netcdf, setup_nztm_grid_netcdf


# def get_elevation_data(ravel_idxs):
#     ds = nc.Dataset(env.data('/RCMData/Version6/RCP2.6/BCC-CSM1.1/MaxTempCorr_VCSN_BCC-CSM1.1_2006-2010_RCP2.6.nc'))
#     return ds.variables['elevation'][:].ravel()[ravel_idxs]
#
#
# def get_vscn_elevation_data(ravel_idxs):
#     ds = nc.Dataset(env.data('/RCMData/Version6/RCP2.6/BCC-CSM1.1/MaxTempCorr_VCSN_BCC-CSM1.1_2006-2010_RCP2.6.nc'))
#     return np.flipud(ds.variables['elevation'][:]).ravel()[ravel_idxs]

#
# def get_ravel_idxs(mask):
#     existing_landpoints = np.load(get_subset_landpoints_fname())
#     masked_landpoints = ravel.calc_ravel_idxs(mask, False)
#
#     # Only include the points in the subsetted data
#     masked_landpoints = np.array(filter(lambda x: x in existing_landpoints, masked_landpoints))
#
#     return masked_landpoints
#
#
# def get_masks():
#     shp_file = shapefile.Reader(mask_shpfile)
#     res = []
#     # first field is id and the second is the name
#     for idx, sr in enumerate(shp_file.shapeRecords()):
#         mask = create_mask_from_shpfile(VCSN_file.latitudes(), VCSN_file.longitudes(), mask_shpfile, idx)
#         res.append((sr.record[1], mask))
#     return res
#
#
# def subset_mask(mask, data):
#     """
#     Obtain the subset defined by the mask
#
#     Since the data array is already ravelled, it cannot be masked easily. Instead the overlapping ravelled indexs must be calculated and
#     then extracted
#     :param mask:
#     :param data:
#     :return: The subsetted data
#     """
#     existing_landpoints = np.load(get_subset_landpoints_fname())
#     masked_landpoints = get_ravel_idxs(mask)
#
#     # Find the overlapping points
#     overlap = np.array([idx in masked_landpoints for idx in existing_landpoints])
#
#     return data[:, overlap]


def interpolate_met(in_dat, var, in_lons, in_lats, in_elev, out_lons, out_lats, out_elev, lapse=-0.005, single_dt=False):
    """
    interpolate met data for one timestep from coarse (vcsn) grid onto higher-resolution grid using bilinear interpolation.

    Air temperatures are first lapsed to sea level using default lapse rate of 0.005 K per m, interpolated, then lapsed to elevation of new grid

    when the input is a masked array, nearest neighbour interpolation is used to fill the gaps iaround the edge of the input data mask and the output data mask.
     If no mask is set on the elevation data, a mask at sea-level will be created.

    :param in_dat: 3D array with data to be interpolated. has matrix [i,j] coordinates i.e. dimensions [time, in_lats, in_lons]
    :param var: name of variable to be interpolated. if 't_max', or 't_min'  will lapse to sea level before interpolation
    :param in_lons: 1D or 2D array containing longitudes of input data
    :param in_lats: 1D or 2Darray containing latitudes of input data
    :param in_elev: 2D array containing elevation of input data, dimesions [in_lats, in_lons] or same as in_lons
    :param out_lons: 1D array containing longitudes of output data
    :param out_lats: 1D array containing latitudes of output data
    :param out_elev: 2D array containing elevation of output data, dimension [out_lats, out_lons]
    :param lapse: lapse rate used to reduce data to sea level before interpolation
    :return: out_dat: 3D array with interpolated data has dimensions [time, out_lats, out_lons]
    """

    # X, Y = np.meshgrid(vcsn_lons, vcsn_lats)
    if out_lons.ndim == 1 and out_lats.ndim == 1:
        # y_array, x_array = np.meshgrid(y_centres, x_centres, indexing='ij')
        YI, XI, = np.meshgrid(out_lats, out_lons,
                              indexing='ij')  # the input grid must have i,j ordering with y(lats) being the first dimension.
        num_out_lats = len(out_lats)
        num_out_lons = len(out_lons)
    else:
        num_out_lats = out_lons.shape[0]
        num_out_lons = out_lons.shape[1]
        XI = out_lons
        YI = out_lats

    if single_dt == False:
        out_dat = np.empty([in_dat.shape[0], num_out_lats, num_out_lons], dtype=np.float32) * np.nan

        for i in range(in_dat.shape[0]):

            in_dat1 = in_dat[i, :, :] * 1.0

            if type(in_dat) == np.ma.core.MaskedArray:
                in_dat1.data[in_dat1.mask] = np.nan

            if var in ['tmax', 'tmin']:  # lapse to sea level
                in_t_offset = in_elev * lapse
                in_dat1 = in_dat1 - in_t_offset

            out_dat1 = basemap.interp(in_dat1, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=1)  # bilinear grid - will miss edges
            if type(in_dat) == np.ma.core.MaskedArray:
                if type(out_elev) == np.ma.core.MaskedArray:
                    elev_mask = out_elev.mask
                else:
                    elev_mask = out_elev == 0  # mask out areas less than 1 m above sea level

                # first extrapolate around the edges of the grid
                array = np.ma.masked_invalid(in_dat1)
                xx, yy = np.meshgrid(in_lons, in_lats)
                # get only the valid values
                x1 = xx[~array.mask]
                y1 = yy[~array.mask]
                newarr = array[~array.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')
                GD1 = np.ma.masked_invalid(GD1)
                out_dat0 = basemap.interp(GD1, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=0)  # nearest neighbour grid to fill edges
                ind = np.logical_and(out_dat1.mask, elev_mask == False)  # select areas without data in bilinear grid and above sea level
                out_dat1[ind] = out_dat0[ind]  # use nearest neighbour grid to fill grid with elev mask to set limits
                out_dat1[elev_mask] = np.nan
                # out_dat0 = basemap.interp(in_dat1, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=0)  # nearest neighbour grid to fill edges
                # out_dat1[np.where(out_dat1.mask)] = out_dat0[np.where(out_dat1.mask)]  # replace the masked elements in bilinear grid with the nn grid

            if var in ['tmax', 'tmin']:  # lapse back to new elevations
                out_t_offset = out_elev * lapse
                out_dat1 = out_dat1 + out_t_offset

            out_dat[i, :, :] = out_dat1

    elif single_dt == True:

        # out_dat = np.empty([num_out_lats, num_out_lons], dtype=np.float32) * np.nan
        # in_dat1 = in_dat * 1.0

        if type(in_dat) == np.ma.core.MaskedArray:
            in_dat.data[in_dat.mask] = np.nan

        if var in ['tmax', 'tmin']:  # lapse to sea level
            in_t_offset = in_elev * lapse
            in_dat = in_dat - in_t_offset

        out_dat = basemap.interp(in_dat, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=1)
        if type(in_dat) == np.ma.core.MaskedArray:
            out_dat0 = basemap.interp(in_dat, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=0)  # nearest neighbour grid to fill edges
            out_dat[np.where(out_dat.mask)] = out_dat0[np.where(out_dat.mask)]
        # mask data at sea level
        # out_dat1[out_elev.data < 1.0] = np.nan # no longer send in a masked array

        if var in ['tmax', 'tmin']:  # lapse back to new elevations
            out_t_offset = out_elev * lapse
            out_dat = out_dat + out_t_offset

    return out_dat.astype(np.float32)


def daily_to_hourly_temp_grids(max_temp_grid, min_temp_grid, single_dt=False):
    """
    run through and process daily data into hourly, one slice at a time.
    :param max_temp_grid: input data with dimension [time,y,x]
    :param min_temp_grid: input data with dimension [time,y,x]
    :return: hourly data with dimension [time*24,y,x]
    """
    if single_dt == True:  # assume is 2d and add a time dimension on the start
        max_temp_grid = max_temp_grid.reshape([1, max_temp_grid.shape[0], max_temp_grid.shape[1]])
        min_temp_grid = min_temp_grid.reshape([1, min_temp_grid.shape[0], min_temp_grid.shape[1]])
    hourly_grid = np.empty([max_temp_grid.shape[0] * 24, max_temp_grid.shape[1], max_temp_grid.shape[2]], dtype=np.float32) * np.nan
    for i in range(max_temp_grid.shape[1]):
        hourly_grid[:, i, :] = process_temp(max_temp_grid[:, i, :], min_temp_grid[:, i, :])
    return hourly_grid


def daily_to_hourly_swin_grids(swin_grid, lats, lons, hourly_dt, single_dt=False):
    """
    converts daily mean SW into hourly using TOA rad, applying

    :param hi_res_sw_rad: daily sw in data with dimension [time,y,x]
    :return:
    """
    if single_dt == True:  # assume is 2d and add a time dimension on the start
        swin_grid = swin_grid.reshape([1, swin_grid.shape[0], swin_grid.shape[1]])

    num_steps_in_day = int(86400. / (hourly_dt[1] - hourly_dt[0]).total_seconds())
    hourly_grid = np.ones([swin_grid.shape[0] * num_steps_in_day, swin_grid.shape[1], swin_grid.shape[2]])

    lon_ref = np.mean(lons)
    lat_ref = np.mean(lats)
    # compute hourly TOA for reference in middle of domain #TODO explicit calculation for each grid point?
    toa_ref = calc_toa(lat_ref, lon_ref, hourly_dt)
    # compute daily average TOA and atmospheric transmissivity
    daily_av_toa = []
    for i in range(0, len(toa_ref), num_steps_in_day):
        daily_av_toa.append(np.mean(toa_ref[i:i + num_steps_in_day]))
    daily_trans = swin_grid / np.asarray(daily_av_toa)[:, np.newaxis, np.newaxis]
    # calculate hourly sw from daily average transmisivity and hourly TOA
    for ii, i in enumerate(range(0, len(toa_ref), num_steps_in_day)):
        hourly_grid[i:i + num_steps_in_day] = hourly_grid[i:i + num_steps_in_day] * toa_ref[i:i + num_steps_in_day, np.newaxis, np.newaxis] * daily_trans[ii]

    return hourly_grid


def load_new_vscn(variable, dt_out, nc_file_in, point=None, nc_opt=False, single_dt=False, nc_datetimes=None):
    """
    load vcsn data from file for specified datetimes. transforms spatial dimensions so that latitude and longitude are increasing
    :param variable: string describing the field to take. options for newVCSN data are 'rain', 'tmax', 'tmin', 'srad'
    :param dt_out: array of datetimes requested
    :param nc_file_in: string describing full path to netCDF file with VCSN data
    :param point[y,x] : point to extract data at, where y and x refer to the array positions of point required
    :param nc_opt: set to True if nc_file_in is a netCDF instance rather than a string
    :return: array containing VCSN data with dimensions [time, lat, lon]
    """
    if nc_opt:
        nc_file = nc_file_in
    else:
        nc_file = nc.Dataset(nc_file_in)

    if nc_datetimes is None:
        nc_datetimes = nc.num2date(nc_file.variables['time'][:], nc_file.variables['time'].units)

    if single_dt == False:
        # nc dts are in UTC, and recorded at 9am. To get record relevant to NZST day (at 00:00), need to subtract 3 hours (12 hour offset, plus 9 hours)
        index = np.where(np.logical_and(nc_datetimes >= (dt_out[0] - dt.timedelta(hours=3)),
                                        nc_datetimes <= (dt_out[-1] - dt.timedelta(hours=3))))
    else:
        index = np.where(nc_datetimes == (dt_out - dt.timedelta(hours=3)))

    start_idx = index[0][0]
    end_idx = index[0][-1]
    if variable == 'tmax' or variable == 'rain' or variable == 'srad':  # take measurement (max or sum) to 9am next day
        start_idx = start_idx + 1
        end_idx = end_idx + 1
    if point is None:
        data = np.fliplr(nc_file.variables[variable][start_idx:end_idx + 1, :, :])
        # flip so latitude and longitude is increasing. i.e. origin at bottom left.    # fliplr flips second dimension
        if single_dt:
            data = np.squeeze(data)
    else:
        data = nc_file.variables[variable][start_idx:end_idx + 1, point[0], point[1]]

    return data


if __name__ == '__main__':

    # dem control
    origin = 'topleft'  # orientation of output dem options are 'topleft' or 'bottomleft'. assume that input Dem is 'topleft'
    output_dem = 'nztm250m'  # identifier for output dem
    dem_file = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/si_dem_250m.tif'
    # mask control
    mask_dem = False  # boolean to set whether or not to mask the output dem - assume mask is bottom left
    catchment = 'SI_DEM'
    mask_created = True  # boolean to set whether or not the mask has already been created - assume mask is bottom left
    mask_folder = 'T:/DSC-Snow/Masks'  # location of numpy catchment mask. must be writeable if mask_created == False
    mask_shpfile = 'Z:/GIS_DATA/Hydrology/Catchments/{}.shp'.format(
        catchment)  # shapefile containing polyline or polygon of catchment in WGS84. Not needed if mask_created==True
    # time control
    hydro_years = False  # use hydrolgical years (april 1 to March 31)?
    hydro_years_to_take = range(2000, 2000 + 1)  # range(2001, 2013 + 1)
    save_by_timestep = False  # save one timestep per file? Needed for Fortran version of dsc_snow, only works with compute_by_day==False
    compute_by_day = True  # only compute hourly values one day at a time? Useful for large grids, as not enough memory to compute for whole grid at once.
    ta_version = 'norton'  # options for temperature vcsn and norton
    # input met data
    nc_file_rain = 'C:/Users/conwayjp/OneDrive - NIWA/Data/newVCSN/rain_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
    if ta_version == 'norton':
        nc_file_tmax = 'C:/Users/conwayjp/OneDrive - NIWA/Data/newVCSN/tmax_N2_1980010100_2017073100_south-island_p05_daily.nc'
        nc_file_tmin = 'C:/Users/conwayjp/OneDrive - NIWA/Data/newVCSN/tmin_N2_1980010100_2017073100_south-island_p05_daily.nc'
    elif ta_version == 'vcsn':
        nc_file_tmax = 'Z:/newVCSN/tmax_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
        nc_file_tmin = 'Z:/newVCSN/tmin_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
    nc_file_srad = 'C:/Users/conwayjp/OneDrive - NIWA/Data/newVCSN/srad_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
    # output met data
    met_out_folder = 'C:/Users/conwayjp/Documents/test3'

    ####

    # set up input and output DEM for processing
    # output DEM
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, origin=origin, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6,
                                                                          extent_s=4.82e6, resolution=250)
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

    for hydro_year_to_take in hydro_years_to_take:
        # load data
        # create timestamp to get - this is in NZST
        if hydro_years == True:
            dts_to_take = np.asarray(make_regular_timeseries(dt.datetime(hydro_year_to_take - 1, 4, 1), dt.datetime(hydro_year_to_take, 3, 31), 86400))
        else:
            dts_to_take = np.asarray(make_regular_timeseries(dt.datetime(hydro_year_to_take, 1, 1), dt.datetime(hydro_year_to_take, 12, 31), 86400))

        # pull only data needed.
        # this loads data for 00h NZST that corresponds to the day to come in i.e. min@ 8am, max @ 2pm , total sw and total rain for 1/1/2000 at 2000-01-01 00:00:00
        precip_daily = load_new_vscn('rain', dts_to_take, nc_file_rain)
        max_temp_daily = load_new_vscn('tmax', dts_to_take, nc_file_tmax)
        min_temp_daily = load_new_vscn('tmin', dts_to_take, nc_file_tmin)
        sw_rad_daily = load_new_vscn('srad', dts_to_take, nc_file_srad)
        # load grid (assume same for all input data)
        ds = nc.Dataset(nc_file_rain)
        vcsn_elev = np.flipud(ds.variables['elevation'][:])
        vcsn_lats = np.flip(ds.variables['latitude'])
        vcsn_lons = ds.variables['longitude'][:]
        hy_index = np.ones(dts_to_take.shape, dtype='int')

        # check dimensions and projection of the new data
        # nztm_elev_check = interpolate_met(np.asarray([vcsn_elev]), Precip, vcsn_lons, vcsn_lats, vcsn_elev, lons,
        #                                     lats, elev)
        # plt.imshow(nztm_elev_check[0], origin=0)
        # plt.imshow(elev, origin=0)
        start_dt = dts_to_take[0]
        finish_dt = dts_to_take[-1]

        # interpolate data to fine grid
        hi_res_precip = interpolate_met(precip_daily, 'rain', vcsn_lons, vcsn_lats, np.ma.fix_invalid(vcsn_elev).data, lons, lats, elev)
        hi_res_max_temp = interpolate_met(max_temp_daily, 'tmax', vcsn_lons, vcsn_lats, np.ma.fix_invalid(vcsn_elev).data, lons, lats, elev)
        hi_res_min_temp = interpolate_met(min_temp_daily, 'tmin', vcsn_lons, vcsn_lats, np.ma.fix_invalid(vcsn_elev).data, lons, lats, elev)
        hi_res_sw_rad = interpolate_met(sw_rad_daily, 'srad', vcsn_lons, vcsn_lats, np.ma.fix_invalid(vcsn_elev).data, lons, lats, elev)

        # make all the data outside catchment nan to save space
        if mask_dem:
            hi_res_precip[:, trimmed_mask == False] = np.nan
            hi_res_max_temp[:, trimmed_mask == False] = np.nan
            hi_res_min_temp[:, trimmed_mask == False] = np.nan
            hi_res_sw_rad[:, trimmed_mask == False] = np.nan

        # process and write
        if compute_by_day == True:  # process and write one day at a time.
            hourly_dt = np.asarray(make_regular_timeseries(start_dt + dt.timedelta(hours=1), finish_dt + dt.timedelta(days=1), 3600))
            if hydro_years == True:
                outfile = met_out_folder + '/met_inp_{}_hy{}.nc'.format(data_id, hydro_year_to_take)
            else:
                outfile = met_out_folder + '/met_inp_{}_{}_{}_{}.nc'.format(data_id, hydro_year_to_take, ta_version, origin)

            out_nc_file = setup_nztm_grid_netcdf(outfile, None, ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                                 hourly_dt, northings, eastings, lats, lons, elev)
            day_weightings = []
            num_days = hi_res_precip.shape[0]
            for i in range(num_days):
                # Do the temporal downsampling for one day
                # precip is random cascade for each day. NOTE original VCSN data has almost correct timestamp - ie. total from 9am.
                hourly_precip, day_weightings_1 = process_precip(hi_res_precip[i],
                                                                 one_day=True)  # TODO: align to 9am-9am - currently counts pretends it is midnight-midnight
                # air temperature is three part sinusoidal between min at 8am and max at 2pm. NOTE original VCSN data has correct timestamp - ie. minimum to 9am, maximum from 9am.
                # hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i], hi_res_min_temp[i], single_dt=True)  #
                if i == 0:
                    hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i:i + 2], hi_res_min_temp[i:i + 2])
                    hourly_temp = hourly_temp[:24]
                elif i == num_days:
                    hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i - 1:], hi_res_min_temp[i - 1:])
                    hourly_temp = hourly_temp[-24:]
                else:
                    hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp[i - 1:i + 2], hi_res_min_temp[i - 1:i + 2])
                    hourly_temp = hourly_temp[24:48]
                #
                hourly_swin = daily_to_hourly_swin_grids(hi_res_sw_rad[i], lats, lons, hourly_dt[i * 24: (i + 1) * 24], single_dt=True)

                for var, data in zip(['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                     [hourly_temp, hourly_precip, hourly_swin]):
                    out_nc_file.variables[var][i * 24: (i + 1) * 24, :, :] = data
                day_weightings.extend(day_weightings_1)
            out_nc_file.close()
            if hydro_years == True:
                pickle.dump(day_weightings, open(met_out_folder + '/met_inp_{}_hy{}_daywts.pkl'.format(data_id, hydro_year_to_take), 'wb'), -1)
            else:
                pickle.dump(day_weightings,
                            open(met_out_folder + '/met_inp_{}_{}_{}_{}_daywts.pkl'.format(data_id, hydro_year_to_take, ta_version, origin), 'wb'), -1)

        else:  # compute all the values then write (takes too much memory for large grids)
            # Do the temporal downsampling
            # precip is random cascade for each day. # NOTE original VCSN data has almost correct timestamp - ie. total from 9am.
            hourly_precip, day_weightings = process_precip(hi_res_precip)  # TODO: align to 9am-9am
            # air temperature is three part sinusoidal between min at 8am and max at 2pm. # NOTE original VCSN data has correct timestamp - ie. minimum to 9am, maximum from 9am.
            hourly_temp = daily_to_hourly_temp_grids(hi_res_max_temp, hi_res_min_temp)
            hourly_dt = np.asarray(make_regular_timeseries(start_dt, finish_dt + dt.timedelta(days=1), 3600))
            hourly_swin = daily_to_hourly_swin_grids(hi_res_sw_rad, lats, lons, hourly_dt)

            # Save out the met data and weightings used for precip
            if save_by_timestep == True:
                for i in range(len(hourly_dt)):
                    dt_save = hourly_dt[i]
                    write_nztm_grids_to_netcdf(met_out_folder + '/{}/met_inp_{}_{}.nc'.format(catchment, data_id, dt_save.strftime('%Y%m%d%H')),
                                               [np.squeeze(hourly_temp[i, :, :]), np.squeeze(hourly_precip[i, :, :]), np.squeeze(hourly_swin[i, :, :])],
                                               ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                               dt_save, northings, eastings, lats, lons, elev, no_time=True)
                pickle.dump(day_weightings, open(met_out_folder + '/{}/met_inp_{}_hy{}_daywts.pkl'.format(catchment, data_id, hydro_year_to_take), 'wb'), -1)
            else:
                write_nztm_grids_to_netcdf(met_out_folder + '/met_inp_{}_hy{}.nc'.format(data_id, hydro_year_to_take),
                                           [hourly_temp, hourly_precip, hourly_swin],
                                           ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                           hourly_dt, northings, eastings, lats, lons, elev)
                pickle.dump(day_weightings, open(met_out_folder + '/met_inp_{}_hy{}_daywts.pkl'.format(data_id, hydro_year_to_take), 'wb'), -1)
