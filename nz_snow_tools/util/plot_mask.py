"""
code to plot the masks


"""
import os
import matplotlib.pylab as plt
import numpy as np

def trim_data_to_mask(data, mask):
    """
    # trim data to minimum box needed for mask
    :param data: 2D (x,y) or 3D (time,x,y) array
    :param mask: 2D boolean with same x,y dimensions as data
    :return: data trimmed
    """
    valid_lat_bounds = np.nonzero(mask.sum(axis=1))[0]
    lat_min_idx = valid_lat_bounds.min()
    lat_max_idx = valid_lat_bounds.max()
    valid_lon_bounds = np.nonzero(mask.sum(axis=0))[0]
    lon_min_idx = valid_lon_bounds.min()
    lon_max_idx = valid_lon_bounds.max()

    if data.ndim == 2:
        trimmed_data = data[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].astype(data.dtype)
    elif data.ndim == 3:
        trimmed_data = data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].astype(data.dtype)
    else:
        print('data does not have correct dimensions')

    return trimmed_data



mask_folder = '/nesi/project/niwa00026/Observations/Snow_RemoteSensing/catchment_masks/Southland_Fiordland/'
masks =  os.listdir(mask_folder)
 
for m in masks:
    plt.figure()
    a = plt.load(mask_folder + '/' + m)
    try:
        plt.imshow(trim_data_to_mask(a,a),origin='lower')
        plt.title(m)
        #plt.savefig(mask_folder + '/' + m + '.png'
        plt.show()
    except ValueError:
        print('no data to plot' + m)