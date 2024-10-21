"""
code to generate tiled masks

"""
import matplotlib.pyplot as plt
import numpy as np
from nz_snow_tools.util.utils import create_mask_from_shpfile, setup_nztm_dem
# from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import
import os


dem = 'modis_nz_dem_250m' # identifier for modis grid - extent specified below
mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/Snowmelt forecast/catchment_masks'  # location of numpy catchment mask. must be writeable if mask_created == False

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


ns = [10, 20, 30, 60, 100, 200] # different x,y domains (km)
for n in ns:
    minx = 1.32e6
    maxy = 5.12e6
    maxx = minx + (n * 1e3) # + 1e3 make slightly not square to ensure we don't accidentially rotate at some point.
    miny = maxy - (n * 1e3)
    mask = np.logical_and(y_centres > miny, y_centres < maxy)[:, np.newaxis] * np.logical_and(x_centres > minx, x_centres < maxx)[np.newaxis, :]
    np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\Snowmelt forecast\catchment_masks\ahuriri_test_{}_square_{}.npy'.format(n,dem), mask)
    plt.imshow(mask,origin='lower',alpha=0.2)

plt.show()
#
# masks =  os.listdir(mask_folder)
# import matplotlib.pylab as plt
# for m in masks:
#     plt.figure()
#     plt.imshow(plt.load(mask_folder + '/' + m),origin='lower')
#     plt.title(m)
# plt.show()