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
import xarray as xr
import dask.array as da
import dask
from dask.distributed import Client

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
    print(f"{datetime.datetime.now()}: reading the dataset")
    dataset_dict = {}
    # for var in config['variables'].keys():
    for var in ['air_temp']:
        with xr.open_dataset(config['variables'][var]['input_file'], decode_times=True) as ds:
            ds = ds.rename_dims({config['variables'][var]['input_time_var']: 'time'}).rename_vars({config['variables'][var]['input_time_var']: 'time'})
            dataset_dict[var] = ds.sel(time=slice(first_time, last_time))
    # dataset_dict['rh'][config['variables']['rh']['input_var_name']] = dataset_dict['rh'][config['variables']['rh']['input_var_name']] * 100  # convert to %
    # dataset_dict['lw_rad'][config['variables']['lw_rad']['input_var_name']] = dataset_dict['lw_rad'][config['variables']['lw_rad']['input_var_name']] / (5.67e-8 * dataset_dict['air_temp'][config['variables']['air_temp']['input_var_name']] ** 4)

    return dataset_dict

@dask.delayed
def process_time_step(inp_nc_file, config, i_time, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, output_grid_dict):
    i_time_step = inp_nc_file[config['variables'][var]['input_var_name']]['time'].isel(time=i_time).values
    # if var == 'rh':
    #     input_hourly = inp_nc_file[config['variables'][var]['input_var_name']].sel(time=i_time_step).values * 100  # convert to %
    if var == 'wind_speed' or var == 'wind_direction':
        pass
    elif var == 'total_precip':
        pass
    else:
        input_hourly = inp_nc_file[config['variables'][var]['input_var_name']].sel(time=i_time_step).values
    hi_res_out = interpolate_met(input_hourly, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)
    hi_res_out[output_grid_dict['trimmed_mask'] == 0] = np.nan
    return hi_res_out


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
        intput_dict['inp_elev_interp'] = input_elev.copy()
        # inp_elev_interp = np.ma.fix_invalid(input_elev).data
        intput_dict['inp_lats'] = nc_file_orog[config['input_grid']['y_coord_name']].values
        intput_dict['inp_lons'] = nc_file_orog[config['input_grid']['x_coord_name']].values

    return intput_dict

def interp_met_nzcsm_time(config_file):

    n_procs = int(os.environ.get("SLURM_CPUS_PER_TASK", '3'))

    # 0, set up dask client
    # memory_target = dask.config.get('distributed.worker.memory.target') # default 0.6
    # memory_spill = dask.config.get('distributed.worker.memory.spill') # default 0.7
    # memory_pause = dask.config.get('distributed.worker.memory.pause') # default 0.8
    # memory_terminate = dask.config.get('distributed.worker.memory.terminate') # default 0.95
    dask.config.set({
        'distributed.worker.memory.target': 0.90,
        'distributed.worker.memory.spill': 0.95,
        'distributed.worker.memory.pause': 0.98,
        'distributed.worker.memory.terminate': 0.99,
    })
    print(f'    running with {n_procs} processes')
    with Client(n_workers=n_procs) as client:
        print(client.dashboard_link)

        config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        first_time = parser.parse(config['output_file']['first_timestamp'])
        last_time = parser.parse(config['output_file']['last_timestamp'])

        # 1, process input orogrpahy
        intput_dict = process_input_orogrpahy(config)

        # 2, read the original data
        dataset_dict = get_dataset_dict(config, first_time, last_time)

        # 3, process output orogrpahy
        out_nc_file, output_grid_dict = process_output_orogrpahy(config, first_time, last_time, intput_dict['rot_pole_crs'])

        # 4, run through each variable

        # for var in config['variables'].keys():
        for var in ['air_temp']:#,'rh','solar_rad','lw_rad']:
            print(f"{datetime.datetime.now()}: processing {var}")
            # set up variable in output file
            t = out_nc_file.createVariable(config['variables'][var]['output_name'], 'f4', ('time', 'northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
            t.setncatts(config['variables'][var]['output_meta'])

            # open input met including model orography (so we can use input on different grids (as long as they keep the same coordinate system)
            inp_nc_file = dataset_dict[var]

            if config['input_grid']['dem_file'] == 'none': # load coordinates of each file
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
            
            out_rlons = output_grid_dict['out_rlons']
            out_rlats = output_grid_dict['out_rlats']
            elev = output_grid_dict['elev']
            inp_lons = intput_dict['inp_lons']
            inp_lats = intput_dict['inp_lats']
            inp_elev_interp = intput_dict['inp_elev_interp']

            # load variables relevant for interpolation
            if var in ['air_temp']:#,'rh','solar_rad','lw_rad']:
                num_time_steps = inp_nc_file[config['variables'][var]['input_var_name']].sizes['time']
                tasks = [process_time_step(inp_nc_file, config, i_time, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, output_grid_dict) for i_time in range(num_time_steps)]
                results = da.compute(*tasks)
                for i_time, hi_res_out in enumerate(results):
                    t[i_time, :, :] = hi_res_out
                # gc.collect()
            # # 'lw_rad'
            # air_temp = out_nc_file[config['variables']['air_temp']['output_name']][:, :, :]
            # out_nc_file[config['variables']['lw_rad']['output_name']][:, :, :] = out_nc_file[config['variables']['lw_rad']['output_name']][:, :, :] * (5.67e-8 * air_temp ** 4)

        out_nc_file.close()
        print(f"{datetime.datetime.now()}: Done")

def interp_met_nzcsm_squential(config_file):

    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    print('processing input orogrpahy') # load to get coordinate reference system for interpolation

    # 1, read dem_file
    if config['input_grid']['dem_file'] == 'none': # if no specific model orography file then open air pressure input variable file and extract coordinate system out of it
        nc_file_orog = xr.open_dataset(config['variables']['air_pres']['input_file'])
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog['rotated_pole']
            rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print('currently only set up for rotated pole')
    else: # config['input_grid']['dem_file'] != 'none': # assume all files have same coordinates and load coordiantes from separte file
        nc_file_orog = xr.open_dataset(config['input_grid']['dem_file'])
        if 'rotated' in config['input_grid']['coord_system']:
            rot_pole = nc_file_orog['rotated_pole']
            rot_pole_crs = ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
        else:
            print('currently only set up for rotated pole')
            
        input_elev = nc_file_orog[config['input_grid']['dem_var_name']].values # needed for pressure
        inp_elev_interp = input_elev.copy()
        # inp_elev_interp = np.ma.fix_invalid(input_elev).data
        inp_lats = nc_file_orog[config['input_grid']['y_coord_name']].values
        inp_lons = nc_file_orog[config['input_grid']['x_coord_name']].values

    first_time = parser.parse(config['output_file']['first_timestamp'])
    last_time = parser.parse(config['output_file']['last_timestamp'])

    # 2, read the original data
    dataset_dict = get_dataset_dict(config, first_time, last_time)

    # 3, setup output
    out_nc_file, output_grid_dict = process_output_orogrpahy(config, first_time, last_time, rot_pole_crs)

    # 3, run through each variable
    # for var in config['variables'].keys():
    for var in ['air_temp']:
        print('processing {}'.format(var))
        # set up variable in output file
        t = out_nc_file.createVariable(config['variables'][var]['output_name'], 'f4', ('time', 'northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
        t.setncatts(config['variables'][var]['output_meta'])

        # open input met including model orography (so we can use input on different grids (as long as they keep the same coordinate system)
        inp_nc_file = dataset_dict[var]

        if config['input_grid']['dem_file'] == 'none': # load coordinates of each file
            inp_lats = inp_nc_file[config['input_grid']['y_coord_name']].values
            inp_lons = inp_nc_file[config['input_grid']['x_coord_name']].values
            if 'rotated' in config['input_grid']['coord_system']:
                rot_pole = inp_nc_file['rotated_pole']
                assert rot_pole_crs == ccrs.RotatedPole(rot_pole.grid_north_pole_longitude, rot_pole.grid_north_pole_latitude, rot_pole.north_pole_grid_longitude)
            else:
                print('only set up for rotated pole coordinates')
            if var in ['air_temp','air_pres']:
                input_elev = inp_nc_file[config['input_grid']['dem_var_name']].values  # needed for pressure adjustment
                inp_elev_interp = input_elev.copy() # needed for air temp
            else:
                inp_elev_interp = None
        
        out_rlons = output_grid_dict['out_rlons']
        out_rlats = output_grid_dict['out_rlats']
        elev = output_grid_dict['elev']

        # load variables relevant for interpolation
        if var == 'air_temp':
            inp_nc_var = inp_nc_file[config['variables'][var]['input_var_name']]
            for i_time in range(inp_nc_var.sizes['time']):
                input_hourly = inp_nc_var.isel(time=i_time).values
                hi_res_out = interpolate_met(input_hourly, var, inp_lons, inp_lats, inp_elev_interp, out_rlons, out_rlats, elev, single_dt=True)
                hi_res_out[output_grid_dict['trimmed_mask'] == 0] = np.nan
                # save timestep to netCDF
                t[i_time, :, :] = hi_res_out

    out_nc_file.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        print(f"{datetime.datetime.now()}: reading configuration file")
    else:
        config_file = r'C:\Users\conwayjp\code\github\nz_snow_tools_jc\nz_snow_tools\hpc_runs\nzcsm_local.yaml'
        print('incorrect number of commandline inputs')

    interp_met_nzcsm_time(config_file)
