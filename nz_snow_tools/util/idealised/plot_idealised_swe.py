# script to plot results from the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt

# compatibility for windows or linux
idealised_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/Projects-DSC-Snow/runs/idealised/'
origin = 'topleft'  # origin of new DEM surface
config_id_out = '7'  # string identifying suffix to the namelist configuration file used by the model
years = [2016]

met_inps = ['bell_2000', 'real']#'flat_2000', 'north_facing', 'south_facing', 'bell_4000',

for met_inp in met_inps:
    for year in years:
        for catchment in met_inps:
            # paths to files
            topo_file = nc.Dataset(idealised_folder + 'dem/{}_nztm250m_topo_no_ice_origin{}.nc'.format(catchment, origin))
            # climate_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/met/met_inp_{}_nztm250m_{}_origin{}.nc'.format(met_inp, year, origin))
            output_file = nc.Dataset(idealised_folder + 'output/snow_out_{}_dem{}_met{}_{}_origin{}.nc'.format(year, catchment, met_inp, config_id_out, origin))
            # namelist_file = projects + '/DSC-Snow/runs/idealised/ddf_config_{}_dem{}_met{}_{}_origin{}.txt'.format(year, catchment, met_inp, config_id_out,
            #                                                                                                        origin)

            nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
            # abl = output_file.variables['ablation_total'][:]
            # acc = output_file.variables['accumulation_total'][:]
            swe = output_file.variables['snow_water_equivalent'][:]
            # water = output_file.variables['water_output_total'][:]
            # sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]

            plt.imshow(np.mean(swe, axis=0))  # ,origin=0
            plt.colorbar()
            plt.contour(topo_file.variables['DEM'][:], range(0, 4000, 500), colors='k', linewidths=0.5)
            plt.title('mean SWE (m w.e.) with 500m contours')
            plt.savefig(idealised_folder + 'plots/p_meanSWE_{}_dem{}_met{}_{}_origin{}.png'.format(year, catchment, met_inp, config_id_out, origin))
            plt.close()
