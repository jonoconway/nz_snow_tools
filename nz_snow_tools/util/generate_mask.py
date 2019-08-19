"""
code to generate catchment masks for modelling
"""
from __future__ import division

import numpy as np
from nz_snow_tools.util.utils import create_mask_from_shpfile, setup_nztm_dem
# from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import
import os
os.environ['PROJ_LIB']=r'C:\miniconda\envs\nz_snow27\Library\share'

catchments = ['Wilkin']
# catchments = ['Clutha','Wilkin','Wanaka northern inflows','Upper Dart','Rees', 'Shotover', 'Teviot','Taieri','Upper Matukituki','Roaring Meg','Pomahaka','Motutapu',\
#               'Moonlight Creek','Matukituki', 'Manuherikia','Luggate Creek', 'Lochy','Lindis',\
#               'Kawarau','Greenstone','Hawea','Fraser','Clutha above Clyde Dam','Cardrona','Arrow' ,'Bannockburn Creek', 'Nevis'] # string identifying the catchment to run. must match the naming of the catchment shapefile

output_dem = 'nztm250m'  # identifier for output dem
dem_folder = '' #'Z:/GIS_DATA/Topography/DEM_NZSOS/'
dem = 'modis_si_dem_250m'
mask_dem = True  # boolean to set whether or not to mask the output dem
mask_created = False  # boolean to set whether or not the mask has already been created
mask_folder = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS\masks'  # location of numpy catchment mask. must be writeable if mask_created == False
# shapefile containing polyline or polygon of catchment in WGS84
catchment_shp_folder = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS\catchments'

# calculate model grid etc:
# output DEM

if dem == 'clutha_dem_250m':
    dem_file = dem_folder + dem + '.tif'
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file)

if dem == 'si_dem_250m':
    dem_file = dem_folder + dem + '.tif'
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6,
                                                                          extent_s=4.82e6,
                                                                          resolution=250)
if dem == 'modis_si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250)
for catchment in catchments:
    mask_shpfile = catchment_shp_folder + '/{}.shp'.format(catchment)
    mask = create_mask_from_shpfile(lat_array, lon_array, mask_shpfile)
    np.save(mask_folder + '/{}_{}.npy'.format(catchment, dem), mask)
