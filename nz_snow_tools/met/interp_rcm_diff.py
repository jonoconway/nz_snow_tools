"""
Generate hourly temperature and precipitation data for use in the snow model

The daily gridded fields (e.g. from VCSN) are downscaled using the same methods given in "Simulations of seasonal snow for the South Island, New Zealand"
Clark et al, 2009
"""
from __future__ import division
import os
#os.environ['PROJ_LIB'] = '/home/jared/anaconda/envs/nz_snow_tools/share/proj'



import datetime as dt
import netCDF4 as nc
import numpy as np
import pickle
#from bs.core import source, env
# from nz_snow_tools.util import convert_projection
import matplotlib.pylab as plt

import os
os.environ['PROJ_LIB']=r'C:\miniconda\envs\nz_snow27\Library\share'

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
                out_dat0 = basemap.interp(in_dat1, in_lons, in_lats, XI, YI, checkbounds=False, masked=False, order=0)  # nearest neighbour grid to fill edges
                out_dat1[np.where(out_dat1.mask)] = out_dat0[np.where(out_dat1.mask)]  # replace the masked elements in bilinear grid with the nn grid
            # mask data at sea level
            # out_dat1[out_elev.data < 1.0] = np.nan # no longer send in a masked array

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
    # code to get elevation and lat/lon - assumes VCSN and RCM use same lat/lon
    # nc_file_rain = nc.Dataset('/mnt/data/RCMData/Version6/Annual/Ann_WS10_VCSN_xairj_1971-2005c0_RCPpast.nc', 'r')
    in_lats = np.genfromtxt(r"C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\lats.dat")#nc_file_rain.variables['latitude'][::-1]  # source.RCM_file.latitudes()
    in_lons = np.genfromtxt(r"C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\lons.dat")#nc_file_rain.variables['longitude'][:]  # source.RCM_file.longitudes()
    # vcsn_elev = np.flipud(nc_file_rain.variables['elevation'][:])
    # vcsn_elev_interp = np.ma.fix_invalid(vcsn_elev).data
    # in_elev = vcsn_elev_interp
    in_elev = np.ones((260,243))



    # code to get output lat/lons
    # dem_folder = '/mnt/data/GIS_DATA/Topography/DEM_NZSOS/'
    #catchment = 'Clutha'
    dem = 'clutha_dem_250m'
    #subcatchment = 'qldc_ta_area'
    catchment = 'Clutha'

    mask_folder = r'C:\Users\conwayjp\OneDrive - NIWA\Temp\Masks'

    # dem_file = dem_folder + dem + '.tif'
    # elev, easting, northing, lat_array, lon_array = setup_nztm_dem(dem_file)

    # create new outlons and out_lats using trim for the qldc mask here
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file=None)
    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, dem))

    # qldc_mask = np.load(mask_folder + '/{}_{}.npy'.format(subcatchment, dem))

    out_lats, out_lons, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres,
                                                   x_centres)
    # returns: lats, lons, elev, northings, eastings
    mask = trimmed_mask

    #out_lons = lon_array
    out_lats = np.flipud(out_lats)  #(lat_array)
    # northing = np.flipud(northing)
    # out_elev = elev
    out_elev = trimmed_mask*0.0

    # # Clip to the same extent as the met data
    # northing_clip = (4861875, 5127625)
    # easting_clip = (1214375, 1370375)
    # northing_mask = (northing >= northing_clip[0]) & (northing <= northing_clip[1])
    # easting_mask = (easting >= easting_clip[0]) & (easting <= easting_clip[1])
    # out_lats = out_lats[northing_mask][:, easting_mask]
    # out_lons = out_lons[northing_mask][:, easting_mask]

    for sce in ['RCP2.6', 'RCP4.5', 'RCP6.0','RCP8.5']:#
        for v in ['T']:#
            for year in ['2045','2095']:# '
                # note: don't lapse the temperature e.g., call it 'rain' variable rather than temperature or use 0 lapse rate
                dat_file = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\NEW_{}diff_{}_{}_mean.npy'.format(v, sce, year)
                # dat_file = 'D:/NEW_{}diff_{}_{}_percent.npy'.format(v, sce, year)
                inp_dat = np.load(dat_file)
                inp_dat = np.flipud(inp_dat)
                in_dat = np.ma.masked_invalid(inp_dat)
                array = in_dat#np.ma.masked_invalid(in_dat)
                xx, yy = np.meshgrid(in_lons, in_lats)
                # get only the valid values
                x1 = xx[~array.mask]
                y1 = yy[~array.mask]
                newarr = array[~array.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                           (xx, yy),
                                           method='nearest')
                GD1 = np.ma.masked_invalid(GD1)
                var = 'rain'

                lapse = 0.000
                single_dt = True

                hi_res_temp_2095 = interpolate_met(GD1, var, in_lons, in_lats, in_elev, out_lons, out_lats, out_elev, lapse, single_dt)

                # with open('/mnt/temp/CODC/metric_plots/SavedData-change/{}diff_{}_{}_mean_downscaled_v2.npy'.format(v, sce, year), 'w') as fh:

                    # np.save(fh, hi_res_temp_2095)
                # plt.imshow(hi_res_temp_2095.data)
                # plt.colorbar()
                # plt.show()
                np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\RCM_differences\diff\{}diff_{}_{}_mean_downscaled_v5.npy'.format(v, sce, year), hi_res_temp_2095.data)
                # np.save('D:/{}diff_{}_{}_mean_downscaled_v3.npy'.format(v, sce, year), hi_res_temp_2095.data)


