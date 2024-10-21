"""
add coordinates to FSM2_NZ gridded output after a run
"""

import xarray as xr

dsm = xr.open_dataset('met.nc')
dss = xr.load_dataset('SWE.nc')
dso = dsm[['elevation','lat','lon']].copy()

da = xr.DataArray(
    data=dss.snw.values,
    dims=["time", "northing", "easting"],
    coords=dict(
        easting=dso.easting,
        northing=dso.northing,
        time=dsm.time[dsm['time.hour'] == 12], # take timestamp everytime hour=12
    ),
    attrs=dict(
        long_name="mass of snowpack",
        units="kg m-2",
    ),
)

first_time =da.time[0].dt.strftime('%Y%m%d%H%M').values
last_time =  da.time[-1].dt.strftime('%Y%m%d%H%M').values
dso.assign(snw=da).to_netcdf('swe_{}_{}.nc'.format(first_time,last_time))
