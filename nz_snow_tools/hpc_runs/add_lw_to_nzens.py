"""
take incoming lw from nzcsm dataset and insert into nzens dataset (correcting for air temperature differences)
"""

import xarray as xr
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import os
import glob

# nzcsm  '/esi/project/niwa00004/jonoconway/FSM_input/NZCSM/nz_30_30/met_interp_nz_30_30_15_11_202204010300_202304010200_250m_nztm_274.nc'

# infile_nzens = r"C:\Users\conwayjp\Downloads\met_interp_nzens000_nz_30_30_15_11_202009011600_202010011500_250m_nztm_274.nc"
# infile_nzcsm = r"C:\Users\conwayjp\Downloads\met_interp_nz_30_30_15_11_202009011600_202010011500_250m_nztm_274.nc"
#
# ds1 = xr.open_dataset(infile_nzens)
# ds2 = xr.open_dataset(infile_nzcsm)

infile_folder = '/esi/project/uoo04017/jonoconway/cylc-run/musa_catchment_workflow/run137/share/forcing_data/nz_30_30_15_11/'
paths = sorted(glob.glob(os.path.join(infile_folder, '*nz_30_30_15_11*.nc')))

ds1 = xr.open_mfdataset(paths, combine='by_coords', chunks={'time': 240})
for var in ['elevation', 'lat', 'lon']:
    ds1[var] = ds1[var].mean(dim='time')

nzcsm_folder = '/esi/project/niwa00004/jonoconway/FSM_input/NZCSM/nz_30_30/'
paths_nzcsm = sorted(glob.glob(os.path.join(nzcsm_folder, '*nz_30_30_15_11*.nc')))

ds2 = xr.open_mfdataset(paths_nzcsm, combine='by_coords', chunks={'time': 240})
for var in ['elevation', 'lat', 'lon']:
    ds2[var] = ds2[var].mean(dim='time')

# fill single nan values with mean of either side
# fill in two timesteps with nans in sfc_temp
shift_forward = ds1.sfc_temp.shift(time=1)
# 2. Get the timestep immediately after
shift_backward = ds1.sfc_temp.shift(time=-1)
# 3. Calculate the linear interpolation (the average of the two)
# Dask handles this lazily and chunk-by-chunk
local_mean = (shift_forward + shift_backward) / 2
# 4. Fill ONLY the NaN values in your original dataset with this local mean
ds1['sfc_temp'] = ds1.sfc_temp.fillna(local_mean)

# identify longer gaps where should be nan values
threshold = 320  # threshold for realistic temperatures
ds1['sfc_temp'] = ds1.sfc_temp.where(ds1.sfc_temp <= threshold, other=np.nan)

data_vars = ['sfc_temp',
             'sfc_rh',
             'total_precip',
             'snowfall_rate',
             'rainfall_rate',
             'sfc_dw_sw_flux',
             'sfc_air_press',
             'sfc_wind_speed',
             'sfc_wind_direction']

for var in data_vars:
    shift_forward24 = ds1[var].shift(time=24)
    ds1[var] = ds1[var].fillna(shift_forward24)
# # fill these longer gaps with the monthly mean.
# monthly_climatology = ds1.sfc_temp.groupby('time.month').mean(dim='time')
# ds1['sfc_temp'] = ds1.sfc_temp.groupby('time.month').fillna(monthly_climatology)
# check for nans

for var in data_vars:
    print('{} {}'.format(var, np.isnan(ds1[var]).values.sum()))

# trim NZCSM to only NZENS period
ds2 = ds2.sel(time=slice(ds1.time.min().values, ds1.time.max().values))

# calculate LW for NZENS temperature
# lw = effective_emissivity * 5.67e-8 * air_temp_K ** 4 so to adjust the temperature keeping the effective emissivity the smae
# lw_new = lw_orig * (ta_new_K / ta_orig_K) ** 4
ds1['sfc_dw_lw_flux'] = ds2.sfc_dw_lw_flux * (ds1.sfc_temp / ds2.sfc_temp) ** 4

dso = ds1

max_chunk = 512
enc = {}
for v, da in dso.data_vars.items():
    # chunk spatial dims in output, keep time=1 if present
    if da.ndim == 3 and "time" in da.dims:
        enc[v] = {"zlib": True, "complevel": 4,
                  "chunksizes": (1, min(max_chunk, da.sizes.get("northing", 1)),
                                 min(max_chunk, da.sizes.get("easting", 1)))}
    elif da.ndim == 2:
        enc[v] = {"zlib": True, "complevel": 4,
                  "chunksizes": (min(max_chunk, da.sizes.get("northing", 1)),
                                 min(max_chunk, da.sizes.get("easting", 1)))}

ds1.sel(time=slice('2021-04-01 01:00', '2022-04-01 00:00')).to_netcdf('/esi/project/niwa00004/jonoconway/FSM_input/nzens_nz_30_30_15_11_20212022.nc',
                                                                      engine='netcdf4', encoding=enc)

for year in np.arange(2021, 2024):
    ds1.sel(time=slice('{}-04-01 01:00'.format(year), '{}-04-01 00:00'.format(year + 1))).to_netcdf(
        '/esi/project/niwa00004/jonoconway/FSM_input/met_interp_nzens_nz_30_30_15_11_{}04010100_{}04010000_250m_nztm_274.nc'.format(year, year + 1),
        engine='netcdf4', encoding=enc)
    print()
