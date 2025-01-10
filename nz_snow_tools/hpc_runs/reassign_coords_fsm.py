"""
add coordinates to FSM2_NZ gridded output after a run
"""

import xarray as xr
import numpy as np

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
        grid_mapping='nztm'
    ),
)

dso.attrs["Conventions"] = "CF-1.8"
dso.easting.attrs['standard_name'] = "projection_x_coordinate"
dso.northing.attrs['standard_name'] = "projection_y_coordinate"

# Define the coordinate system using EPSG:2193
crs = xr.DataArray(np.array(0),
    attrs={
        "grid_mapping_name": "transverse_mercator",
        "longitude_of_central_meridian": 173.0,
        "latitude_of_projection_origin": 0.0,
        "false_easting": 1600000.0,
        "false_northing": 10000000.0,
        "scale_factor_at_central_meridian": 0.9996,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257222101,
        "spatial_ref": "EPSG:2193"
    }
)

first_time =da.time[0].dt.strftime('%Y%m%d%H%M').values
last_time =  da.time[-1].dt.strftime('%Y%m%d%H%M').values
dso = dso.assign(snw=da, nztm=crs)
dso.to_netcdf('swe_{}_{}.nc'.format(first_time,last_time))

dso['dswe'] = dso.snw * np.nan
dso['dswe'][1:] = dso.snw.diff(dim='time', label='upper')  # label at end of the day - because the timestamp is in UTC it will return for the NZ day
# queried e.g.  ds_full.sel('2022-07-18') will return the value for the 24-hour change to midnight '2022-07-19 00:00+12:00'
dso['dswe'] = dso.dswe.assign_attrs(long_name='24-hour change in SWE', units='mm w.e.',
                                            description='change in snow pack storage in mm water equivalent for 24-hour preceeding timestamp',grid_mapping='nztm')

