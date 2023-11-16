"""

reads in and computes statistics on MODIS data for specific catchments. Requires that mask has been created using
nz_snow_tools/util/generate_mask.py

Jono Conway
"""
from __future__ import division

import numpy as np
import netCDF4 as nc
import pickle
import datetime as dt
from time import strftime, gmtime
# from nz_snow_tools.eval.catchment_evaluation import *
from nz_snow_tools.eval.catchment_evaluation_annual import load_subset_modis_annual
from nz_snow_tools.util.utils import make_regular_timeseries

import os

# os.environ['PROJ_LIB']=r'C:\miniconda\envs\nz_snow27\Library\share'

def calc_modis_catchment_metrics(catchment, years_to_take, mask_folder, modis_folder, modis_dem, output_folder):
    """
    reads in and computes statistics on MODIS data for specific catchments. Requires that mask has been created using
    nz_snow_tools/util/generate_mask.py
    saves output to a pickle file
    :param catchment,  # string identifying catchment modelled
    :param years_to_take
    :param modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    :param mask_folder  folder with numpy masks
    :param modis_folder  folder with cloudfilled modis data
    :param modis_dem name of modis grid 'modis_nz_dem_250m'
    :param output_folder folder to put output pickle file
    """

    # set up lists
    ann_ts_av_sca_m = []
    ann_ts_av_sca_thres_m = []
    ann_dt_m = []
    ann_scd_m = []

    for year_to_take in years_to_take:

        print('loading modis data {}'.format(year_to_take))
        # load modis data for evaluation - trims to extent of catchment
        modis_fsca, modis_dt, modis_mask = load_subset_modis_annual(catchment, year_to_take, modis_folder, modis_dem, mask_folder)
        modis_sc = modis_fsca >= modis_sc_threshold

        # print('calculating basin average sca')
        num_modis_gridpoints = np.sum(modis_mask)
        ba_modis_sca = []
        ba_modis_sca_thres = []
        for i in range(modis_fsca.shape[0]):
            ba_modis_sca.append(np.nanmean(modis_fsca[i, modis_mask]) / 100.0)
            ba_modis_sca_thres.append(np.nansum(modis_sc[i, modis_mask]).astype(np.double) / num_modis_gridpoints)

        # print('adding to annual series')
        ann_ts_av_sca_m.append(np.asarray(ba_modis_sca))
        ann_ts_av_sca_thres_m.append(np.asarray(ba_modis_sca_thres))

        # print('calc snow cover duration')
        modis_scd = np.sum(modis_sc, axis=0)  # count days with snow over threshold
        modis_scd[modis_mask == 0] = -999  # set areas outside catchment to -999
        modis_scd[np.logical_and(np.isnan(modis_fsca[0]), modis_mask == 1)] = -1  # set areas of water to -1

        # add to annual series
        ann_scd_m.append(modis_scd)
        ann_dt_m.append(modis_dt)

        # empty arrays so we don't run out of memory (probably not needed)
        modis_fsca = None
        modis_dt = None
        modis_mask = None
        modis_sc = None
        modis_scd = None

    ann = [ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m]
    outfile = '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem, modis_sc_threshold)
    pickle.dump(ann, open(output_folder + outfile, 'wb'), protocol=3)

    # repack into single timeseries
    sca = []
    dts_m = []
    for ts_av_sca_m, dt_m in zip(ann_ts_av_sca_m, ann_dt_m):
        sca.extend(ts_av_sca_m)
        dts_m.extend(dt_m)

    # dts_full = make_regular_timeseries(dt.datetime(years_to_take[0], 1, 1), dt.datetime(years_to_take[-1], 12, 31), 86400)
    outfile_nc = '/ts_MODIS_{}_{}_{}_{}_thres{}.nc'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem, modis_sc_threshold)
    ds = nc.Dataset(output_folder + outfile_nc,'w')
    ds.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    ds.comment= 'catchment averaged fractional snow covered area generated with https://github.com/jonoconway/nz_snow_tools/eval/catchment_evaluation_modis.py'

    ds.createDimension('time', )
    t = ds.createVariable('time', 'f8', ('time',))
    t.long_name = "time"
    t.units = 'days since 1900-01-01 00:00:00'
    t[:] = nc.date2num(dts_m, units=t.units)

    fsca_var = ds.createVariable('fsca', 'f8', ('time',))
    fsca_var.setncatts({'long_name': 'fractional snow covered area'})
    fsca_var[:] = np.asarray(sca)

    ds.close()

if __name__ == '__main__':

    # catchment = 'Tar_Patea_NZTM'  # string identifying catchment modelled
    years_to_take = np.arange(2010, 2019 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    mask_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_masks'  # location of numpy catchment mask. must be writeable if mask_created == False
    modis_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/'
    modis_dem = 'modis_nz_dem_250m'
    output_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_output'

    catchments = ['BOP_Rangitaiki_NZTM',
         'ES_Mataura_NZTM',
         'ES_Waiau_NZTM',
         'Ecan_Clarence_NZTM',
         'Ecan_Hurunui_NZTM',
         'Ecan_Rakaia_NZTM',
         'Ecan_Rangitata_NZTM',
         'Ecan_Waiau_NZTM',
         'Ecan_Waimakariri_NZTM',
         'Ecan_Waitaki_NZTM',
         'HBRC_Mohaka_NZTM',
         'HBRC_Ngaruro_NZTM',
         'HBRC_Wairoa_NZTM',
         'HRZ_Rangitikei_NZTM',
         'HRZ_Whanganui_NZTM',
         'Marl_Wairau_NZTM',
         'ORC_Clutha_NZTM',
         'Tar_Patea_NZTM',
         'Tar_Waiwhakaiho_NZTM',
         'WRC_Waikato_NZTM',
         'WestCoast_Buller_NZTM',
         'WestCoast_Grey_NZTM']


    for catchment in catchments:
        calc_modis_catchment_metrics(catchment, years_to_take, mask_folder, modis_folder, modis_dem, output_folder)
