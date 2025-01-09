"""
interpolate input data to model grid adjusting temperature-dependent fields for elevation changes and output to netCDF

assumes input air temp in K for calculation of LW rad

if interpolating lw or calculating rain vs snow then need to ensure air temp and humidity come before these variables in the config file.

requires
- dem
- mask for the same DEM (created using generate_mask)
- met input files with elevation variable
"""

import yaml
import os
import sys
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import datetime as dt
from dateutil import parser
import matplotlib.pylab as plt
import pickle
from scipy import interpolate

from nz_snow_tools.util.utils import make_regular_timeseries, u_v_from_ws_wd, ws_wd_from_u_v
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import interpolate_met, setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds

from pvlib.location import Location, solarposition
from cloudglacier.obj1.calc_cloud_metrics import *


def interp_met_nzcsm(config_file):
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    print('processing input orogrpahy')  # load to get coordinate reference system for interpolation

    if config['input_grid'][
        'dem_file'] == 'none':  # if no specific model orography file then open air pressure input variable file and extract coordinate system out of it
        nc_file_orog = nc.Dataset(config['variables']['air_pres']['input_file'], 'r')
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog.variables['rotated_pole']
            rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print('currently only set up for rotated pole')
    else:  # config['input_grid']['dem_file'] != 'none': # assume all files have same coordinates and load coordiantes from separte file
        nc_file_orog = nc.Dataset(config['input_grid']['dem_file'], 'r')
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog.variables['rotated_pole']
            rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print('currently only set up for rotated pole')
        input_elev = nc_file_orog.variables[config['input_grid']['dem_var_name']][:]  # needed for pressure
        inp_elev_interp = input_elev.copy()
        # inp_elev_interp = np.ma.fix_invalid(input_elev).data
        inp_lats = nc_file_orog.variables[config['input_grid']['y_coord_name']][:]
        inp_lons = nc_file_orog.variables[config['input_grid']['x_coord_name']][:]

    # create dem of model output grid:
    print('processing output orogrpahy')
    if config['output_grid']['dem_name'] == 'si_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.08e6, extent_e=1.72e6,
                                                                              extent_n=5.52e6, extent_s=4.82e6, resolution=250, origin='bottomleft')
    elif config['output_grid']['dem_name'] == 'nz_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.05e6, extent_e=2.10e6,
                                                                              extent_n=6.275e6, extent_s=4.70e6, resolution=250, origin='bottomleft')
    elif config['output_grid']['dem_name'] == 'modis_nz_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.085e6, extent_e=2.10e6,
                                                                              extent_n=6.20e6, extent_s=4.70e6, resolution=250, origin='bottomleft')
    else:
        print('incorrect dem name specified')

    if config['output_grid']['catchment_mask'] == "elev":  # just set mask to all land points in DEM domain (does not trim to catchment mask first)
        wgs84_lats = lat_array
        wgs84_lons = lon_array
        elev = nztm_dem
        northings = y_centres
        eastings = x_centres
        mask = elev > 0
        trimmed_mask = mask
    else:  # Get the mask for the region of interest
        mask = np.load(config['output_grid']['catchment_mask'])
        # Trim down the number of latitudes requested so it all stays in memory
        wgs84_lats, wgs84_lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)
        _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)

    # calculate rotated grid lat/lon of output grid
    yy, xx = np.meshgrid(northings, eastings, indexing='ij')
    out_rotated_coords = rot_pole_crs.transform_points(ccrs.epsg(2193), xx, yy)
    out_rlats = out_rotated_coords[:, :, 1]
    out_rlons = out_rotated_coords[:, :, 0]
    out_rlons[out_rlons < 0] = out_rlons[out_rlons < 0] + 360

    # precompute slope and aspect from elevation grid
    if 'slope_grid' in config['output_grid'].keys() and 'aspect_grid' in config['output_grid'].keys():
        # read slope and aspect grids # TODO add read
        gridslo = 0
        gridasp = 0
        # assume a square, axis-oriented grid
        gx, gy = np.gradient(elev, 250.0)
        gridslo = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
        # if origin == 'topleft':
        #     data = - np.pi / 2. - np.arctan2(-gx, gy)
        # elif origin == 'bottomleft':
        # assume grid has origin in SW corner ('bottomleft'
        data = - np.pi / 2. - np.arctan2(gx, gy)
        data = np.where(data < -np.pi, data + 2 * np.pi, data)
        gridasp = np.mod(np.degrees(data), 360)

    # set up output times
    first_time = parser.parse(config['output_file']['first_timestamp'])
    last_time = parser.parse(config['output_file']['last_timestamp'])
    out_dt = np.asarray(make_regular_timeseries(first_time, last_time, config['output_file']['timestep']))
    print('time output from {} to {}'.format(first_time.strftime('%Y-%m-%d %H:%M'), last_time.strftime('%Y-%m-%d %H:%M')))

    # set up output netCDF without variables
    if not os.path.exists(config['output_file']['output_folder']):
        os.makedirs(config['output_file']['output_folder'])
    output_file = config['output_file']['file_name_template'].format(first_time.strftime('%Y%m%d%H%M'), last_time.strftime('%Y%m%d%H%M'))
    out_nc_file = setup_nztm_grid_netcdf(config['output_file']['output_folder'] + output_file, None, [], out_dt, northings, eastings, wgs84_lats, wgs84_lons,
                                         elev)

    # run through each variable
    for var in config['variables'].keys():
        print('processing {}'.format(var))
        # set up variable in output file
        t = out_nc_file.createVariable(config['variables'][var]['output_name'], 'f4', ('time', 'northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
        t.setncatts(config['variables'][var]['output_meta'])

        # open input met including model orography (so we can use input on different grids (as long as they keep the same coordinate system)
        inp_nc_file = nc.Dataset(config['variables'][var]['input_file'], 'r')

        if config['input_grid']['dem_file'] == 'none': # load coordinates of each file
            inp_lats = inp_nc_file.variables[config['input_grid']['y_coord_name']][:]
            inp_lons = inp_nc_file.variables[config['input_grid']['x_coord_name']][:]
            if 'rotated' in config['input_grid']['coord_system']:
                rot_pole = inp_nc_file.variables['rotated_pole']
                assert rot_pole_crs == ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude,
                                                        rot_pole.north_pole_grid_longitude)
            else:
                print('only set up for rotated pole coordinates')
            if var in ['air_temp', 'air_pres']:
                input_elev = inp_nc_file.variables[config['input_grid']['dem_var_name']][:]  # needed for pressure adjustment
                inp_elev_interp = input_elev.copy()  # needed for air temp
            else:
                inp_elev_interp = None

        inp_dt = nc.num2date(inp_nc_file.variables[config['variables'][var]['input_time_var']][:],
                            inp_nc_file.variables[config['variables'][var]['input_time_var']].units,
                            only_use_cftime_datetimes=False)  # only_use_python_datetimes=True
        if 'round_time' in config['variables'][var].keys():
            if config['variables'][var]['round_time']:
                inp_hours = nc.date2num(inp_dt, 'hours since 1900-01-01 00:00')
                inp_dt = nc.num2date(np.round(inp_hours, 0), 'hours since 1900-01-01 00:00')
        # load variables relevant for interpolation
        if var == 'lw_rad':
            inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]
            # load air temperature
            inp_nc_file_t = nc.Dataset(config['variables']['air_temp']['input_file'], 'r')
            inp_dt_t = nc.num2date(inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']][:],
                                   inp_nc_file_t.variables[config['variables']['air_temp']['input_time_var']].units, only_use_cftime_datetimes=False)
            inp_nc_var_t = inp_nc_file_t.variables[config['variables']['air_temp']['input_var_name']]
        elif var == 'wind_speed' or var == 'wind_direction':
            if 'convert_uv' in config['variables'][var].keys():
                if config['variables'][var]['convert_uv']:
                    inp_nc_var_u = inp_nc_file.variables[config['variables'][var]['input_var_name_u']]
                    inp_nc_var_v = inp_nc_file.variables[config['variables'][var]['input_var_name_v']]
            else:
                inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]
                if var == 'wind_direction':  # load wind speed to enable conversion to u/v for interpolation
                    inp_nc_var_ws = inp_nc_file.variables[config['variables']['wind_speed']['input_var_name']]
        else:
            inp_nc_var = inp_nc_file.variables[config['variables'][var]['input_var_name']]

        if var == 'total_precip':  # load temp (and optionally rh) to calculate rain/snow rate if needed
            if 'calc_rain_snow_rate' in config['variables']['total_precip'].keys():
                if config['variables']['total_precip']['calc_rain_snow_rate']:
                    # set up additional outputs
                    sfr = out_nc_file.createVariable('snowfall_rate', 'f4', ('time', 'northing', 'easting',), zlib=True)
                    sfr.setncatts(config['variables']['total_precip']['snow_rate_output_meta'])
                    rfr = out_nc_file.createVariable('rainfall_rate', 'f4', ('time', 'northing', 'easting',), zlib=True)
                    rfr.setncatts(config['variables']['total_precip']['rain_rate_output_meta'])
                    if config['variables']['total_precip']['rain_snow_method'] == 'harder':
                        # load rh #TODO arange variable keys so that t and rh are calculated before rain/snow rate and lw_rad
                        dict_hm = pickle.load(open(config['variables']['total_precip']['harder_interp_file'], 'rb'))
                        th_interp = interpolate.RegularGridInterpolator((dict_hm['tc'], dict_hm['rh']), dict_hm['th'], method='linear', bounds_error=False,
                                                                        fill_value=None)

        # run through each timestep output interpolate data to fine grid
        for ii, dt_t in enumerate(out_dt):
            ind_dt = int(np.where(inp_dt == dt_t)[0][0])

            if var == 'lw_rad':
                # calculate effective emissivity using lwin and air temp, interpolate that, then recreate lw rad with lapsed air temperature (and/or air temp adjusted for cliamte change offset)
                input_hourly = inp_nc_var[ind_dt, :, :]
                input_hourly = input_hourly / (5.67e-8 * inp_nc_var_t[int(np.where(inp_dt_t == dt_t)[0][0]), :, :] ** 4)
                hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)
                # loads new air temp adjusted for elevation and optionally climate change scenario
                hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
                hi_res_out = hi_res_out * (5.67e-8 * hi_res_tk ** 4)

            elif var == 'air_pres':  # assumes input data is in Pa. reduce to sea-level (if needed) - interpolate, then raise to new grid.
                if 'input_mslp' in config['variables']['air_pres'].keys():
                    if config['variables']['air_pres']['input_mslp'] == True:
                        input_hourly = inp_nc_var[ind_dt, :, :]
                    elif config['variables']['air_pres']['input_mslp'] == False:
                        # reduce to sea-level
                        input_hourly = inp_nc_var[ind_dt, :, :]
                        input_hourly = input_hourly + 101325 * (1 - (1 - input_elev / 44307.69231) ** 5.253283)
                else: # default to input data being at model level
                    # reduce to sea-level
                    input_hourly = inp_nc_var[ind_dt, :, :]
                    input_hourly = input_hourly + 101325 * (1 - (1 - input_elev / 44307.69231) ** 5.253283)
                    # taken from campbell logger program from Athabasca Glacier
                    # comes from https://s.campbellsci.com/documents/au/manuals/cs106.pdf
                    # U. S. Standard Atmosphere and dry
                    # air were assumed when Equation 3 was derived (Wallace, J. M. and P. V.
                    # Hobbes, 1977: Atmospheric Science: An Introductory Survey, Academic Press,
                    # pp. 59-61).
                hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)
                hi_res_out = hi_res_out - 101325 * (1 - (1 - elev / 44307.69231) ** 5.253283)
                if config['variables']['air_pres']['output_meta']['units'] == 'hPa':
                    hi_res_out /= 100.

            elif var == 'rh':  # assumes input data is as a fraction
                # need to ignore mask for rh data and has incorrect limit of 1.
                input_hourly = inp_nc_var[ind_dt, :, :].data
                input_hourly = input_hourly * 100  # convert to %
                hi_res_out = interpolate_met(input_hourly, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)

            elif var == 'wind_speed':
                if 'convert_uv' in config['variables'][var].keys():
                    if config['variables'][var]['convert_uv']:
                        input_hourly = np.sqrt(inp_nc_var_u[ind_dt, :, :] ** 2 +
                                               inp_nc_var_v[ind_dt, :, :] ** 2)
                else:
                    input_hourly = inp_nc_var[ind_dt, :, :]
                hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)

            elif var == 'wind_direction':
                if 'convert_uv' in config['variables'][var].keys():
                    if config['variables'][var]['convert_uv']:
                        input_hourly_u = inp_nc_var_u[ind_dt, :, :]
                        input_hourly_v = inp_nc_var_v[ind_dt, :, :]
                else:
                    input_hourly_wd = inp_nc_var[ind_dt, :, :]
                    input_hourly_ws = inp_nc_var_ws[ind_dt, :, :]
                    input_hourly_u, input_hourly_v = u_v_from_ws_wd(input_hourly_ws, input_hourly_wd)

                hi_res_out_u = interpolate_met(input_hourly_u.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev,
                                               single_dt=True)
                hi_res_out_v = interpolate_met(input_hourly_v.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev,
                                               single_dt=True)
                hi_res_out = np.rad2deg(np.arctan2(-hi_res_out_u, -hi_res_out_v))
            elif var == 'solar_rad' and 'slope_grid' in config['output_grid'].keys() and 'aspect_grid' in config['output_grid'].keys():
                input_hourly = inp_nc_var[ind_dt, :, :]
                # compute solar geometry for centre of domain (to check if need to compute)
                aws_loc = Location(wgs84_lats.mean(), wgs84_lons.mean(), tz='UTC', altitude=elev.mean())
                # get solar geometry for correction calculations
                if 'mean' in config['variables']['solar_rad']['input_var_name']:
                    dt_solar = pd.to_datetime(dt_t).tz_localize('UTC') - np.timedelta64(30, 'm')  # move back to centre of previous hour
                else:
                    dt_solar = pd.to_datetime(dt_t).tz_localize('UTC')
                solp = aws_loc.get_solarposition(dt_solar)
                if solp['elevation'].values > 0.5: # only compute if sun is more than 0.5 degrees above horizon
                    # split into diffuse and direct components (so could be compatible with reading these directly)
                    # compute clear-sky in middle of output domain
                    cs = aws_loc.get_clearsky(dt_solar, model='simplified_solis', aod700=0.10) # TODO - use precipitable water and/or Iqbal to get better estimate
                    sw_cs_ghi = cs.ghi.values
                    trc = input_hourly.filled(np.nan) / sw_cs_ghi
                    k = 0.65 # TODO use Conway, et al., 2016 formulae for k     k = 0.1715 + 0.07182 * vp ;   k[k > 0.95] = 0.95
                    neff = (1 - trc) / k
                    neff[neff < 0] = 0
                    neff[neff > 1] = 1
                    fdiff = 0.1 + 0.9 * neff
                    input_hourly_diff = input_hourly * fdiff
                    input_hourly_dir = input_hourly * (1 - fdiff)
                    #TODO read in diffuse and direct components

                    # interpolate the diffuse component (accounting for svf of self-shading slope (no terrain shading))
                    hi_res_diff = interpolate_met(input_hourly_diff.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev,
                                                  single_dt=True)
                    hi_res_diff_svf = hi_res_diff * (1 + np.cos(np.radians(gridslo))) / 2  # Liu and Jordan
                    # interpolate the direct component accounting for slope, aspect and self-shading
                    # ratio of normal irradiance to horizontal (sin_h)
                    sin_h = np.maximum(0.01, np.sin(np.radians(solp['elevation'].values)))
                    # calculate geometry between solar beam and slope (cos_zetap)
                    lat = aws_loc.latitude  # latitude in degrees north
                    longitude = aws_loc.longitude  # longitude in degrees east
                    doy = [t.timetuple().tm_yday for t in [dt_solar]]  # day of year
                    soldec = np.asarray([pvlib.solarposition.declination_spencer71(d) for d in doy])  # solar declination in radians
                    eqt = np.asarray([pvlib.solarposition.equation_of_time_spencer71(d) for d in doy])  # equation of time in minutes
                    # solar hour angle (in degrees) 0 = local midday, -90 = 9 am local time, 90 = 3pm local time
                    hour_angle = pvlib.solarposition.hour_angle(pd.DatetimeIndex([dt_solar, dt_solar]), longitude, eqt)  # hack time as DatetimeIndex for pvlib
                    hour_angle = hour_angle[0]
                    cos_zetap1 = (np.cos(np.deg2rad(gridslo)) * np.sin(np.deg2rad(lat)) - np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(180 - gridasp)) * np.sin(
                        np.deg2rad(gridslo))) * np.sin(soldec)
                    cos_zetap2 = (np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(180 - gridasp)) * np.sin(np.deg2rad(gridslo)) + np.cos(
                        np.deg2rad(gridslo)) * np.cos(
                        np.deg2rad(lat))) * np.cos(soldec) * np.cos(-1 * hour_angle * np.pi / 180)
                    cos_zetap3 = np.sin(np.deg2rad(180 - gridasp)) * np.sin(np.deg2rad(gridslo)) * np.cos(soldec) * np.sin(
                        -1 * hour_angle * np.pi / 180)  # hour_angle from pvlib is opposite to that in Thomas's model calculation
                    cos_zetap = cos_zetap1 + cos_zetap2 + cos_zetap3
                    hi_res_dir_hor = interpolate_met(input_hourly_dir.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev,
                                                     single_dt=True)
                    hor_to_slope_multiplier = cos_zetap / sin_h  # ratio of direct radiation on slope and horizontal planes
                    hor_to_slope_multiplier[cos_zetap < 0] = 0  # no direct beam if self shaded.
                    hor_to_slope_multiplier[hor_to_slope_multiplier < 0] = 0
                    hor_to_slope_multiplier[hor_to_slope_multiplier > 10] = 10  # limit to enhancement given that could be issues with timing etc
                    hi_res_dir_slope = hi_res_dir_hor * hor_to_slope_multiplier
                    # recombine into surface radiation
                    hi_res_out = hi_res_diff_svf + hi_res_dir_slope
                else:  # sun below horizon so solar rad = 0
                    hi_res_out = elev * 0

            else:  # all other variables. NOTE air temperature is lapsed to sea level before interpolation within interpolate_met
                input_hourly = inp_nc_var[ind_dt, :, :]
                hi_res_out = interpolate_met(input_hourly.filled(np.nan), var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)

            hi_res_out[trimmed_mask == 0] = np.nan

            # plt.figure()
            # plt.imshow(hi_res_out, origin='lower')
            # plt.colorbar()
            # plt.show()
            # add climate chnage offsets (temperature change will also affect the longwave radiation and rain/snow partioning, but not RH, or SW rad)
            if 'climate_change_offsets' in config.keys():
                if var in config['climate_change_offsets'].keys():
                    if 'percentage_change' in config['climate_change_offsets'][var].keys():
                        hi_res_out = hi_res_out * (100. + config['climate_change_offsets'][var]['percentage_change']) / 100.
                    elif 'absolute_change' in config['climate_change_offsets'][var].keys():
                        hi_res_out = hi_res_out + config['climate_change_offsets'][var]['absolute_change']

            # save timestep to netCDF
            t[ii, :, :] = hi_res_out

            if var == 'total_precip':  # load temp (and optionally rh) to calculate rain/snow rate if needed
                if 'calc_rain_snow_rate' in config['variables'][var].keys():
                    if config['variables'][var]['calc_rain_snow_rate']:
                        if config['variables'][var]['rain_snow_method'] == 'harder':
                            hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
                            hi_res_rh = out_nc_file[config['variables']['rh']['output_name']][ii, :, :]
                            hi_res_tc = hi_res_tk - 273.15
                            th = np.asarray([th_interp([t, r]) for r, t in zip(hi_res_rh.ravel(), hi_res_tc.ravel())]).squeeze().reshape(hi_res_rh.shape)
                            b = 2.6300006
                            c = 0.09336
                            hi_res_frs = 1 - (1. / (1 + b * c ** th))  # fraction of snowfall
                            hi_res_frs[hi_res_frs < 0.01] = 0
                            hi_res_frs[hi_res_frs > 0.99] = 1

                        else:
                            hi_res_tk = out_nc_file[config['variables']['air_temp']['output_name']][ii, :, :]
                            hi_res_frs = (hi_res_tk < config['variables'][var]['rain_snow_method']).astype('float')
                        hi_res_rain_rate = hi_res_out * (1 - hi_res_frs) / config['output_file']['timestep']
                        hi_res_snow_rate = hi_res_out * hi_res_frs / config['output_file']['timestep']
                        rfr[ii, :, :] = hi_res_rain_rate.filled(np.nan)
                        sfr[ii, :, :] = hi_res_snow_rate.filled(np.nan)
                        # plt.figure()
                        # plt.imshow(hi_res_rain_rate, origin='lower')
                        # plt.colorbar()
                        # plt.figure()
                        # plt.imshow(hi_res_snow_rate, origin='lower')
                        # plt.colorbar()
                        # plt.show()

    out_nc_file.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        print('reading configuration file')
    else:
        config_file = r'C:\Users\conwayjp\code\github\nz_snow_tools_jc\nz_snow_tools\hpc_runs\nzcsm_local.yaml'
        print('incorrect number of commandline inputs')

    interp_met_nzcsm(config_file)
