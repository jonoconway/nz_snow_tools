from __future__ import division

import datetime as dt
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
import pickle

from scipy.stats import linregress
from nz_snow_tools.util.utils import process_precip, process_temp, create_mask_from_shpfile, make_regular_timeseries, calc_toa, trim_lat_lon_bounds, \
    setup_nztm_dem
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import load_new_vscn
from nz_snow_tools.util.write_fsca_to_netcdf import write_nztm_grids_to_netcdf, setup_nztm_grid_netcdf


def find_vcsn_point(lat, lon, nc_file_in):
    nc_file = nc.Dataset(nc_file_in)
    lats = nc_file.variables['latitude'][:]
    lons = nc_file.variables['longitude'][:]

    lat_idx = (np.abs(lats - lat)).argmin()
    lon_idx = (np.abs(lons - lon)).argmin()
    print('latitude = {}'.format((nc_file.variables['latitude'][lat_idx])))
    print('longitude = {}'.format((nc_file.variables['longitude'][lon_idx])))
    print('elevation = {}m'.format((nc_file.variables['elevation'][lat_idx, lon_idx])))

    return [lat_idx, lon_idx]


if __name__ == '__main__':


    calc_grid = True  # calculate for whole grid?

    if calc_grid == False:
        lat_to_get = -44.075
        lon_to_get = 169.425

        nc_file_rain = 'T:/newVCSN/rain_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
        nc_file_tmax = 'T:/newVCSN/tmax_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
        nc_file_tmin = 'T:/newVCSN/tmin_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'
        nc_file_srad = 'T:/newVCSN/srad_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'

        point_to_get = find_vcsn_point(lat_to_get, lon_to_get, nc_file_rain)

        dts_to_take = np.asarray(make_regular_timeseries(dt.datetime(2001 - 1, 4, 1), dt.datetime(2016, 3, 31), 86400))
        # pull only data needed.
        # this loads data for 00h NZST that corresponds to the day to come in i.e. min@ 8am, max @ 2pm , total sw and total rain for 1/1/2000 at 2000-01-01 00:00:00
        precip_daily = load_new_vscn('rain', dts_to_take, nc_file_rain, point=point_to_get)
        max_temp_daily = load_new_vscn('tmax', dts_to_take, nc_file_tmax, point=point_to_get)
        min_temp_daily = load_new_vscn('tmin', dts_to_take, nc_file_tmin, point=point_to_get)
        sw_rad_daily = load_new_vscn('srad', dts_to_take, nc_file_srad, point=point_to_get)

        n = len(max_temp_daily)
        x = np.arange(n)
        plt.figure()

        plt.subplot(4, 1, 1)
        slope, intercept, r_value, p_value, std_err = linregress(x, precip_daily)
        y = np.arange(n) * slope + intercept
        plt.plot(dts_to_take, precip_daily)
        plt.plot(dts_to_take, y)
        plt.title('precip. slope = {} yr^-1, p = {}'.format(slope * 365, p_value))

        plt.subplot(4, 1, 2)
        slope, intercept, r_value, p_value, std_err = linregress(x, max_temp_daily)
        y = np.arange(n) * slope + intercept
        plt.plot(dts_to_take, max_temp_daily)
        plt.plot(dts_to_take, y)
        plt.title('tmax. slope = {} yr^-1, p = {}'.format(slope * 365, p_value))

        plt.subplot(4, 1, 3)
        slope, intercept, r_value, p_value, std_err = linregress(x, min_temp_daily)
        y = np.arange(n) * slope + intercept
        plt.plot(dts_to_take, min_temp_daily)
        plt.plot(dts_to_take, y)
        plt.title('tmin. slope = {} yr^-1, p = {}'.format(slope * 365, p_value))

        plt.subplot(4, 1, 4)
        slope, intercept, r_value, p_value, std_err = linregress(x, sw_rad_daily)
        y = np.arange(n) * slope + intercept
        plt.plot(dts_to_take, sw_rad_daily)
        plt.plot(dts_to_take, y)
        plt.title('sw rad. slope = {} yr^-1, p = {}'.format(slope * 365, p_value))

        plt.tight_layout()
        plt.show()

    if calc_grid:

        dts_to_take = np.asarray(make_regular_timeseries(dt.datetime(2000, 1, 1), dt.datetime(2017, 1, 1), 86400))

        metrics = ['rain','tmax','tmin','srad']
        for var in metrics:
            trend_data = load_new_vscn(var, dts_to_take, 'T:/newVCSN/{}_vclim_clidb_1972010100_2017102000_south-island_p05_daily.nc'.format(var))

            slopes = np.empty(trend_data.shape[1:],dtype='float64') * np.nan
            p_values = np.empty(trend_data.shape[1:],dtype='float64') * np.nan

            n = trend_data.shape[0]
            x = np.arange(n)

            for l in range(trend_data.shape[1]):
                for m in range(trend_data.shape[2]):
                    t_data = np.squeeze(trend_data[:, l, m])
                    slope, intercept, r_value, p_value, std_err = linregress(x, t_data)
                    slopes[l, m] = slope
                    p_values[l, m] = p_value


            plot_slopes = slopes * 365
            plot_slopes[(p_values > 0.05)] = np.nan
            
            plt.figure()
            plt.imshow(plot_slopes, cmap=plt.cm.RdBu, origin='lower', interpolation='none', aspect='auto',vmin=-1 * np.nanmax(np.abs(plot_slopes)), vmax=np.nanmax(np.abs(plot_slopes))) #, vmin=vmin, vmax=vmax
            plt.colorbar()
            plt.title('trend in {} per year 2000-2016'.format(var))
            plt.tight_layout()
            plt.savefig(r'D:\Snow project\VCSN trends\{}.png'.format(var))
        #plt.show()
    # plt.figure()
    # plt.plot(dts_to_take, plt.cumsum(precip_daily[(min_temp_daily<273.15)]))
    # plt.plot(dts_to_take, plt.cumsum(precip_daily[(max_temp_daily<275.15)]))
    # plt.plot(dts_to_take, plt.cumsum(precip_daily))
    #plt.show()
