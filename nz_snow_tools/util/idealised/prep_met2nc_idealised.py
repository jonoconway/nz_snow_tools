# read DEM geotiffs and create netCDF required for snow model

# jono conway
#
# from osgeo import gdal, ogr
# from gdalconst import *
# import pyproj
# import argparse

import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem
from nz_snow_tools.util.write_fsca_to_netcdf import setup_nztm_grid_netcdf

lapse = -0.005
climate_file = nc.Dataset(r"T:\DSC-Snow\input_data_hourly\met_inp_Clutha_nztm250m_2016_norton_topleft.nc")
origin = 'topleft' # origin of new DEM surface

# create dem surface
clutha_dem, inp_eastings, inp_northings, _, _ = setup_nztm_dem('Z:/GIS_DATA/Topography/DEM_NZSOS/clutha_dem_250m.tif')

y_point = np.where(climate_file.variables['northing'][:] == inp_northings[703])[0][0]
x_point = np.where(climate_file.variables['easting'][:] == inp_eastings[133])[0][0]
inp_t = climate_file.variables['air_temperature'][:, y_point, x_point]
inp_p = climate_file.variables['precipitation_amount'][:, y_point, x_point]
inp_sw = climate_file.variables['surface_downwelling_shortwave_flux'][:, y_point, x_point]
inp_elev = climate_file.variables['elevation'][y_point, x_point]
hourly_dt = nc.num2date(climate_file.variables['time'][:], climate_file.variables['time'].units)

# create new dem surface
_, eastings, northings, lats, lons = setup_nztm_dem(dem_file=None, extent_w=1.235e6, extent_e=1.26e6, extent_n=5.05e6, extent_s=5.025e6, resolution=250,origin=origin)

for catchment in ['flat_2000', 'north_facing', 'south_facing','bell_2000','bell_4000']:#]:
    if catchment == 'flat_2000':
        elev = np.ones((100, 100)) * 2000
    elif catchment == 'north_facing':
        elev = np.linspace(4000, 0, 100)[:, np.newaxis] * np.ones((100, 100))
        if origin == 'topleft':
            np.flipud(elev)
    elif catchment == 'south_facing':
        elev = np.linspace(0, 4000, 100)[:, np.newaxis] * np.ones((100, 100))
        if origin == 'topleft':
            np.flipud(elev)
    elif catchment == 'bell_4000':
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.25, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        elev = g * 4000
    elif catchment == 'bell_2000':
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.25, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        elev = g * 2000

    # lapse onto new grid
    t_grid = inp_t[:, np.newaxis, np.newaxis] * np.ones((len(inp_t), elev.shape[0], elev.shape[1])) \
             + lapse * (elev[np.newaxis, :, :] * np.ones((len(inp_t), elev.shape[0], elev.shape[1])) - inp_elev)
    p_grid = inp_p[:, np.newaxis, np.newaxis] * np.ones((len(inp_t), elev.shape[0], elev.shape[1]))
    sw_grid = inp_sw[:, np.newaxis, np.newaxis] * np.ones((len(inp_t), elev.shape[0], elev.shape[1]))

    output_dem = 'nztm250m'  # identifier for output dem
    data_id = '{}_{}'.format(catchment, output_dem)  # name to identify the output data
    out_file = 'P:/Projects/DSC-Snow/runs/idealised/met_inp_{}_{}_origin{}.nc'.format(data_id, 2016,origin)
    out_nc_file = setup_nztm_grid_netcdf(out_file, None, ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                         hourly_dt, northings, eastings, lats, lons, elev)
    for var, data in zip(['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                         [t_grid, p_grid, sw_grid]):
        out_nc_file.variables[var][:] = data
    out_nc_file.close()

#
# create real met
catchment = 'real'
y1 = 703 - 50
y2 = 703 + 50
x1 = 133 - 50
x2 = 133 + 50
assert climate_file.variables['easting'][x1:x2][:][0] == eastings[0]
assert climate_file.variables['northing'][y1:y2][:][-1] == northings[0]

t_grid = np.flip(climate_file.variables['air_temperature'][:, y1:y2, x1:x2],axis=1)
plt.imshow(t_grid[0])
plt.show()

p_grid = np.flip(climate_file.variables['precipitation_amount'][:, y1:y2, x1:x2],axis=1)
sw_grid = np.flip(climate_file.variables['surface_downwelling_shortwave_flux'][:, y1:y2, x1:x2],axis=1)
elev = np.flipud(climate_file.variables['elevation'][y1:y2, x1:x2])

plt.figure()
plt.imshow(elev)
plt.show()

output_dem = 'nztm250m'  # identifier for output dem
data_id = '{}_{}'.format(catchment, output_dem)  # name to identify the output data
out_file = 'P:/Projects/DSC-Snow/runs/idealised/met_inp_{}_{}_origin{}.nc'.format(data_id, 2016, origin)
out_nc_file = setup_nztm_grid_netcdf(out_file, None, ['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                                     hourly_dt, northings, eastings, lats, lons, elev)
for var, data in zip(['air_temperature', 'precipitation_amount', 'surface_downwelling_shortwave_flux'],
                     [t_grid, p_grid, sw_grid]):
    out_nc_file.variables[var][:] = data
out_nc_file.close()



