
import yaml
import numpy as np
from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import setup_nztm_dem, setup_nztm_grid_netcdf, trim_lat_lon_bounds


yaml_file = r'C:\Users\conwayjp\code\github\nz_snow_tools_jc\nz_snow_tools\hpc_runs\nzcsm_local.yaml'

config = yaml.load(open(yaml_file), Loader=yaml.FullLoader)


print('processing output orogrpahy')
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

output_file = '/coords_dem_mask_{}_{}.nc'.format(config['output_grid']['catchment_name'],config['output_grid']['dem_name'])
out_nc_file = setup_nztm_grid_netcdf(config['output_file']['output_folder'] + output_file, None, [], None, northings, eastings, wgs84_lats, wgs84_lons,
                                         elev,no_time=True)
t = out_nc_file.createVariable('mask', 'i4', ('northing', 'easting',), zlib=True)  # ,chunksizes=(1, 100, 100)
t.setncatts( {'long_name': 'model catchment mask'})
t[:] = trimmed_mask

out_nc_file.close()
#
# met = xr.open_dataset(config['output_file']['output_folder'] + output_file)
# met.elevation.to_netcdf('elevation.nc')
# met.mask.to_netcdf('mask.nc')