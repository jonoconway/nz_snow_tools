# read DEM geotiffs and create netCDF required for snow model

# jono conway
#

import pickle
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem

origin = 'topleft'  # origin of new DEM surface
out_file = 'met_offset_origin{}_trimmed.nc'.format(origin)
working_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/from Ruschle/eval/plots_jono/'

# load dem surface
dem_file = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/si_dem_250m.tif'
elev, eastings, northings, lats, lons  = setup_nztm_dem(dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6,
                                                                      extent_s=4.82e6,
                                                                      resolution=250)

# load t offset file
t_grid = pickle.load(open(
    working_folder + 't_bias_optim_t_NS_SouthIsland_swe20_norton_5_t-2_p0_topleft_rs4_smooth10_trimmed_fullres.pkl', 'rb'))
# set precip offset to 0
p_grid = np.zeros(t_grid.shape)

file_out = nc.Dataset(working_folder + out_file, 'w')

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
