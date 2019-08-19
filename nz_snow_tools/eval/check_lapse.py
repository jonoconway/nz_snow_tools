
from __future__ import division

import matplotlib.pylab as plt
import netCDF4 as nc
import numpy as np

nc_file2 = nc.Dataset(r"T:\DSC-Snow\input_data_hourly\met_inp_Clutha_nztm250m_2000_norton_topleft.nc")
nc_file = nc.Dataset(r"T:\DSC-Snow\input_data_hourly\met_inp_Clutha_nztm250m_2000_jobst_ucc_topleft.nc")
nc_file3 = nc.Dataset(r"T:\DSC-Snow\input_data_hourly\met_inp_Clutha_nztm250m_2000_vcsn_topleft.nc")
ta_n = nc_file2.variables['air_temperature']
ta_v = nc_file3.variables['air_temperature']
ta_j = nc_file.variables['air_temperature']
elev_v = nc_file3.variables['elevation'][:]
elev_j = nc_file.variables['elevation'][:]
elev_n = nc_file2.variables['elevation'][:]
np.all(elev_j == elev_n)
np.all(elev_j == elev_v)
np.all(elev_n == elev_v)
elev = elev_n

plt.figure()
plt.plot(elev[:,334],np.mean(ta_v[4368:5088,:,334],axis=0),label='vcsn')
plt.plot(elev[:,334],np.mean(ta_j[4368:5088,:,334],axis=0),label='jobst')
plt.plot(elev[:,334],np.mean(ta_n[4368:5088,:,334],axis=0),label='norton')
plt.plot(np.arange(0,1600),277 - 0.005 *np.arange(0,1600),'k--',label='-0.005K')
plt.legend()
plt.xlabel('Elevation (m)')
plt.ylabel('Average ta (K)')
plt.title('Average ta for July 2000 for S-N slice through centre of Clutha domain')
plt.savefig(r'D:\Snow project\Oct2018 Results\check_july_lapse.png')

plt.figure()
plt.plot(np.mean(ta_v[4368:5088,:,334],axis=0),label='vcsn')
plt.plot(np.mean(ta_j[4368:5088,:,334],axis=0),label='jobst')
plt.plot(np.mean(ta_n[4368:5088,:,334],axis=0),label='norton')
plt.show()