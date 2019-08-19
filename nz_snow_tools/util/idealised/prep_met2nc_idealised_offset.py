# read DEM geotiffs and create netCDF required for snow model

# jono conway
#

import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem


origin = 'topleft'  # origin of new DEM surface
out_file = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/Projects-DSC-Snow/runs/idealised/met_off_origin{}.nc'.format(origin)


inp_t = -2.0
inp_p = 20.0

# create new dem surface
_, eastings, northings, lats, lons = setup_nztm_dem(dem_file=None, extent_w=1.235e6, extent_e=1.26e6, extent_n=5.05e6, extent_s=5.025e6, resolution=250,
                                                    origin=origin)

elev = np.zeros((100, 100))  # set up dummy elevation at 0

# put onto new grid
t_grid = inp_t * np.ones((100, 100))
p_grid = inp_p * np.ones((100, 100))

file_out = nc.Dataset(out_file, 'w')

# start with georeferencing
file_out.createDimension('rows', len(northings))
file_out.createDimension('columns', len(eastings))

file_out.createDimension('latitude', len(northings))
file_out.createDimension('longitude', len(eastings))

# latitude
lat_out = file_out.createVariable('latitude', 'f', ('rows', 'columns'))
lat_out[:] = lats
setattr(lat_out, 'units', 'degrees')
# longitude
lon_out = file_out.createVariable('longitude', 'f', ('rows', 'columns'))
lon_out[:] = lons
setattr(lon_out, 'units', 'degrees')
# easting
east_out = file_out.createVariable('easting', 'f', ('columns',))
east_out[:] = eastings
setattr(east_out, 'units', 'm NZTM')
# northing
north_out = file_out.createVariable('northing', 'f', ('rows',))
north_out[:] = northings
setattr(north_out, 'units', 'm NZTM')


for var, data, units in zip(['TairOff', 'PptnOff'], [t_grid, p_grid], ['K', '%change']):
    raster_out = file_out.createVariable(var, 'f', ('rows', 'columns'))
    raster_out[:] = data
    setattr(raster_out, 'units', units)

file_out.close()

