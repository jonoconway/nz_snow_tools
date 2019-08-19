# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml

# compatibility for windows or linux
projects = '/mnt/shareddrive/Projects' #
temp = '/mnt/temp'
origin = 'topleft' # origin of data  - options are 'topleft' or 'bottomleft'
path_to_ddf_run = "/home/bs/dsc_snow/ddf/ddf_run" # path to fortran executable
config_id_in = '6' # string identifying suffix to the namelist configuration file used as default
config_id_out = '6' # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in  )
years = range(2000, 2016 + 1)
#inp_id = 'jobst_ucc' # string identifying the input data used (suffix to input netcdf files)

inp_ids =['norton']# ['jobst_ucc', 'norton', 'vcsn']

for inp_id in inp_ids:
    for year in years:
        # read in namelist
        nml = f90nml.read(path_to_namelist)
        # modify for given run
        nml['ddf_config']['TopographyFile'] = projects + '/DSC-Snow/runs/input_DEM/Clutha_nztm250m_topo_no_ice_origin{}.nc'.format(origin)
        nml['ddf_config']['ClimateSource'] = temp + '/DSC-Snow/input_data_hourly/met_inp_Clutha_nztm250m_{}_{}_{}.nc'.format(year, inp_id, origin)
        nml['ddf_config']['OutputFile'] = temp + '/DSC-Snow/runs/output/clutha_nztm250m_erebus/Clutha_nztm250m_{}_{}_{}_{}.nc'.format(year, inp_id, config_id_out,origin)
        nml['ddf_config']['starttime'][0] = year
        nml['ddf_config']['endtime'][0] = year + 1
        # write namelist
        path_to_output_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}_{}_{}_{}.txt'.format(year, inp_id, config_id_out,origin)
        nml.write(path_to_output_namelist,force=True)

        # # copy the met input onto P drive
        # metinfile = nml['ddf_config']['ClimateSource']
        # shutil.copy(data2 + '/DSC-Snow/input_data_hourly/' + metinfile.split('/')[-1], metinfile)

        # run model
        args = [path_to_ddf_run, path_to_output_namelist]
        subprocess.call(args)

        # move output file to Temp drive
        # outfile = nml['ddf_config']['OutputFile']
        # shutil.move(outfile, temp + '/DSC-Snow/runs/output/clutha_nztm250m_erebus/' + outfile.split('/')[-1])

        # # remove the met input from P drive
        # os.remove(metinfile)
