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
import numpy as np
import cartopy.crs as ccrs
from dateutil import parser
import datetime
import gc
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

from nz_snow_tools.util.utils import make_regular_timeseries, u_v_from_ws_wd, ws_wd_from_u_v
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import interpolate_met, setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds


def process_output_orogrpahy(config, first_time, last_time, rot_pole_crs):
    # create dem of model output grid:
    print(f"{datetime.datetime.now()}: processing output orogrpahy")
    if config['output_grid']['dem_name'] == 'si_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6,
                                                                            extent_s=4.82e6, resolution=250, origin='bottomleft')
    elif config['output_grid']['dem_name'] == 'nz_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6,
                                                                            extent_s=4.70e6, resolution=250, origin='bottomleft')
    elif config['output_grid']['dem_name'] == 'modis_nz_dem_250m':
        nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(config['output_grid']['dem_file'], extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6,
                                                                            extent_s=4.70e6, resolution=250, origin='bottomleft')
    else:
        print(' incorrect dem name specified')

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

    # set up output times
    out_dt = np.asarray(make_regular_timeseries(first_time, last_time, config['output_file']['timestep']))
    print(' time output from {} to {}'.format(first_time.strftime('%Y-%m-%d %H:%M'), last_time.strftime('%Y-%m-%d %H:%M')))

    # set up output netCDF without variables
    if not os.path.exists(config['output_file']['output_folder']):
        os.makedirs(config['output_file']['output_folder'])
    output_file = config['output_file']['file_name_template'].format(first_time.strftime('%Y%m%d%H%M'), last_time.strftime('%Y%m%d%H%M'))
    out_nc_file = setup_nztm_grid_netcdf(config['output_file']['output_folder'] + output_file, None, [], out_dt, northings, eastings, wgs84_lats, wgs84_lons, elev)

    output_grid_dict = {}
    output_grid_dict['out_rlons'] = out_rlons
    output_grid_dict['out_rlats'] = out_rlats
    output_grid_dict['elev'] = elev
    output_grid_dict['trimmed_mask'] = trimmed_mask
    return out_nc_file, output_grid_dict

def get_dataset_dict(config, first_time, last_time):
    dataset_dict = {}
    for var in config['variables'].keys():
        with xr.open_dataset(config['variables'][var]['input_file'], decode_times=True) as ds:
            ds = ds.rename_dims({config['variables'][var]['input_time_var']: 'time'}).rename_vars({config['variables'][var]['input_time_var']: 'time'})
            dataset_dict[var] = ds.sel(time=slice(first_time, last_time))
    # dataset_dict['lw_rad'][config['variables']['lw_rad']['input_var_name']] = dataset_dict['lw_rad'][config['variables']['lw_rad']['input_var_name']] / (5.67e-8 * dataset_dict['air_temp'][config['variables']['air_temp']['input_var_name']] ** 4)

    return dataset_dict

def process_time_step(config, dataset_dict_vars, i_time, var, intput_dict, output_grid_dict):
    # read the original data
    dataset_dict_timestep = {}
    for i_vars in dataset_dict_vars.keys():
        dataset_dict_timestep[i_vars] = dataset_dict_vars[i_vars].sel(time=i_time).load()
    
    try:
        if var == 'rh':
            input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values * 100  # convert to %
        elif var == 'lw_rad':
            input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values
            air_temp_hourly = dataset_dict_timestep['air_temp'][config['variables']['air_temp']['input_var_name']].values
            input_hourly = input_hourly / (5.67e-8 * air_temp_hourly ** 4)
        elif var == 'air_pres':
            if 'input_mslp' in config['variables']['air_pres'].keys():
                if config['variables']['air_pres']['input_mslp'] == True:
                    input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values
                elif config['variables']['air_pres']['input_mslp'] == False:
                    # reduce to sea-level
                    input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values
                    input_hourly = input_hourly + 101325 * (1 - (1 - intput_dict['input_elev'] / 44307.69231) ** 5.253283)
            else: # default to input data being at model level
                # reduce to sea-level
                input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values
                input_hourly = input_hourly + 101325 * (1 - (1 - intput_dict['input_elev'] / 44307.69231) ** 5.253283)
                # taken from campbell logger program from Athabasca Glacier
                # comes from https://s.campbellsci.com/documents/au/manuals/cs106.pdf
                # U. S. Standard Atmosphere and dry
                # air were assumed when Equation 3 was derived (Wallace, J. M. and P. V.
                # Hobbes, 1977: Atmospheric Science: An Introductory Survey, Academic Press,
                # pp. 59-61).
        elif var == 'wind_speed' or var == 'wind_direction':
            pass
        elif var == 'total_precip':
            pass
        else:
            input_hourly = dataset_dict_timestep[var][config['variables'][var]['input_var_name']].values
    except Exception as e:
        print(f"Error process_time_step {i_time}: {e}")
        return i_time, None

    out_rlons = output_grid_dict['out_rlons']
    out_rlats = output_grid_dict['out_rlats']
    elev = output_grid_dict['elev']
    inp_lons = intput_dict['inp_lons']
    inp_lats = intput_dict['inp_lats']
    inp_elev_interp = intput_dict['inp_elev_interp']

    hi_res_out = interpolate_met(input_hourly, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)
    
    # post processing
    if var == 'air_pres':
        hi_res_out = hi_res_out - 101325 * (1 - (1 - elev / 44307.69231) ** 5.253283)
        if config['variables']['air_pres']['output_meta']['units'] == 'hPa':
            hi_res_out /= 100.
    
    hi_res_out[output_grid_dict['trimmed_mask'] == 0] = np.nan
    return i_time, hi_res_out

def process_input_orogrpahy(config):
    print(f"{datetime.datetime.now()}: processing input orogrpahy") # load to get coordinate reference system for interpolation
    
    intput_dict = {}

    if config['input_grid']['dem_file'] == 'none': # if no specific model orography file then open air pressure input variable file and extract coordinate system out of it
        nc_file_orog = xr.open_dataset(config['variables']['air_pres']['input_file'])
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog['rotated_pole']
            intput_dict['rot_pole_crs'] = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print(' currently only set up for rotated pole')
    else: # config['input_grid']['dem_file'] != 'none': # assume all files have same coordinates and load coordiantes from separte file
        nc_file_orog = xr.open_dataset(config['input_grid']['dem_file'])
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog['rotated_pole']
            intput_dict['rot_pole_crs'] = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print(' currently only set up for rotated pole')
            
        input_elev = nc_file_orog[config['input_grid']['dem_var_name']].values # needed for pressure
        intput_dict['input_elev'] = input_elev
        intput_dict['inp_elev_interp'] = input_elev.copy()
        # inp_elev_interp = np.ma.fix_invalid(input_elev).data
        intput_dict['inp_lats'] = nc_file_orog[config['input_grid']['y_coord_name']].values
        intput_dict['inp_lons'] = nc_file_orog[config['input_grid']['x_coord_name']].values

    return intput_dict

def process_input_orogrpahy_no_dem_file(config, var, inp_nc_file, intput_dict):
    inp_lats = inp_nc_file[config['input_grid']['y_coord_name']].values
    inp_lons = inp_nc_file[config['input_grid']['x_coord_name']].values
    if 'rotated' in config['input_grid']['coord_system']:
        rot_pole = inp_nc_file['rotated_pole']
        assert intput_dict['rot_pole_crs'] == ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
    else:
        print('only set up for rotated pole coordinates')
    if var in ['air_temp','air_pres']:
        input_elev = inp_nc_file[config['input_grid']['dem_var_name']].values  # needed for pressure adjustment
        inp_elev_interp = input_elev.copy() # needed for air temp
    else:
        inp_elev_interp = None

    intput_dict['inp_lats'] = inp_lats
    intput_dict['inp_lons'] = inp_lons
    intput_dict['input_elev'] = input_elev
    intput_dict['inp_elev_interp'] = inp_elev_interp

def post_processing_lw_rad(out_nc_file, config, var, i_time_index):
    if config['variables']['air_temp']['output_name'] in out_nc_file.variables:
        air_temp = out_nc_file[config['variables']['air_temp']['output_name']][i_time_index, :, :]
        hi_res_out = out_nc_file[config['variables'][var]['output_name']][i_time_index, :, :] * (5.67e-8 * air_temp ** 4)
    else:
        hi_res_out = None
    return i_time_index, hi_res_out

def interp_met_nzcsm_multithread(config_file):

    n_procs = int(os.environ.get("SLURM_CPUS_PER_TASK", '6'))
    print(f'    running with {n_procs} threads')

    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    first_time = parser.parse(config['output_file']['first_timestamp'])
    last_time = parser.parse(config['output_file']['last_timestamp'])
    # Generate a time series of time
    time_series = pd.date_range(start=first_time, end=last_time, freq=f'{config["output_file"]["timestep"]}s')

    # 1, process input orogrpahy
    intput_dict = process_input_orogrpahy(config)

    # 2, read the original data
    dataset_dict = get_dataset_dict(config, first_time, last_time)

    # 3, process output orogrpahy
    out_nc_file, output_grid_dict = process_output_orogrpahy(config, first_time, last_time, intput_dict['rot_pole_crs'])

    # 4, run through each variable

    # for var in config['variables'].keys():
    for var in ['air_pres','air_temp','lw_rad','rh', 'solar_rad',]:
        print(f"{datetime.datetime.now()}: processing {var}")
        # set up variable in output file
        t = out_nc_file.createVariable(config['variables'][var]['output_name'], 'f4', ('time', 'northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
        t.setncatts(config['variables'][var]['output_meta'])

        # open input met including model orography (so we can use input on different grids (as long as they keep the same coordinate system)
        inp_nc_file = dataset_dict[var]

        if config['input_grid']['dem_file'] == 'none': # load coordinates of each file
            process_input_orogrpahy_no_dem_file(config, var, inp_nc_file, intput_dict)

        
        dataset_dict_vars = {}
        if var in ['air_temp','rh','solar_rad','air_pres']:
            dataset_dict_vars[var] = dataset_dict[var]
        elif var == 'lw_rad':
            dataset_dict_vars[var] = dataset_dict[var]
            dataset_dict_vars['air_temp'] = dataset_dict['air_temp']
        with ThreadPoolExecutor(max_workers=n_procs) as executor:
            futures = [executor.submit(process_time_step, config, dataset_dict_vars, i_time, var, intput_dict, output_grid_dict) for i_time in time_series]
            # process task results as they are available
            for future in as_completed(futures):
                try:
                    i_time, hi_res_out = future.result()
                    i_time_index = time_series.get_loc(i_time)
                    if hi_res_out is not None:
                        t[i_time_index, :, :] = hi_res_out
                    # del hi_res_out  # Free memory immediately after use
                    # gc.collect()  # Explicitly call garbage collection
                except Exception as e:
                    print(f"Error processing time step {i_time}: {e}")
                    continue
            # post processing
            if var in ['lw_rad']:
                futures = [executor.submit(post_processing_lw_rad, out_nc_file, config, var, i_time_index) for i_time_index in range(time_series.size)]
                # process task results as they are available
                for future in as_completed(futures):
                    try:
                        i_time_index, hi_res_out = future.result()
                        if hi_res_out is not None:
                            t[i_time_index, :, :] = hi_res_out
                    except Exception as e:
                        i_time = time_series[i_time_index]
                        print(f"Error post processing {i_time}: {e}")
                        continue

            # # 'lw_rad'
            # air_temp = out_nc_file[config['variables']['air_temp']['output_name']][:, :, :]
            # out_nc_file[config['variables']['lw_rad']['output_name']][:, :, :] = out_nc_file[config['variables']['lw_rad']['output_name']][:, :, :] * (5.67e-8 * air_temp ** 4)
        print(f"{datetime.datetime.now()}: Done")
    out_nc_file.close()
        

if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        print(f"{datetime.datetime.now()}: reading configuration file")
    else:
        config_file = r'C:\Users\conwayjp\code\github\nz_snow_tools_jc\nz_snow_tools\hpc_runs\nzcsm_local.yaml'
        print('incorrect number of commandline inputs')

    interp_met_nzcsm_multithread(config_file)
