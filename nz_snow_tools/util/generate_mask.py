"""
code to generate catchment masks for MODIS grid
updated to load for all shapefiles in a folder

"""


import numpy as np
from nz_snow_tools.util.utils import create_mask_from_shpfile, setup_nztm_dem
# from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import
import os


dem = 'modis_nz_dem_250m' # identifier for modis grid - extent specified below
mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/Snowmelt forecast/catchment_masks'  # location of numpy catchment mask. must be writeable if mask_created == False
catchment_shp_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/Snowmelt forecast/catchment_masks/catchment_shapefiles_spatials/'  # shapefile containing polyline or polygon of catchment in WGS84
shapefile_proj = 'NZTM' #  projection of shapefile either NZTM of WGS84
file_type = '.gpkg' # or '.shp'
# read names of shapefiles
contents = os.listdir(catchment_shp_folder)
if file_type == '.gpkg':
    shps = [s.split('.')[0] for s in contents if ".gpkg" in s and ".gpkg-" not in s]#or ".shp" in s and ".xml" not in s]
elif file_type == '.shp':
    shps = [s.split('.')[0] for s in contents if ".shp" in s and ".xml" not in s]

# shps = ['Omarama_13184146_catchment']

# calculate model grid etc:
# output DEM

if dem == 'clutha_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.2e6, extent_e=1.4e6, extent_n=5.13e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')

if dem == 'si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')
if dem == 'modis_si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')

if dem == 'modis_nz_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                          resolution=250, origin='bottomleft')

if dem == 'nz_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6, extent_s=4.70e6,
                                                                          resolution=250, origin='bottomleft')


for catchment in shps:
    if '.shp' in catchment:
        mask_shpfile = catchment_shp_folder + '/{}'.format(catchment)
    else:
        mask_shpfile = catchment_shp_folder + '/{}{}'.format(catchment,file_type)

    if shapefile_proj == 'NZTM':
        mask = create_mask_from_shpfile(y_centres, x_centres, mask_shpfile)
    elif shapefile_proj == 'WGS84':
        mask = create_mask_from_shpfile(lat_array, lon_array, mask_shpfile)
    else:
        print('incorrect shapefile projection')

    np.save(mask_folder + '/{}_{}.npy'.format(catchment, dem), mask)


#
# masks =  os.listdir(mask_folder)
# import matplotlib.pylab as plt
# for m in masks:
#     plt.figure()
#     plt.imshow(plt.load(mask_folder + '/' + m),origin='lower')
#     plt.title(m)
# plt.show()