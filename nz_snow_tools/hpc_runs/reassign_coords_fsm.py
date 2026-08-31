"""
add coordinates to FSM2_NZ gridded output after a run
"""

import xarray as xr
import numpy as np

dsm = xr.open_dataset('met.nc')
dss = xr.load_dataset('SWE.nc')
dso = dsm[['elevation', 'lat', 'lon']].copy()

da = xr.DataArray(
    data=dss.snw.values,
    dims=["time", "northing", "easting"],
    coords=dict(
        easting=dso.easting,
        northing=dso.northing,
        time=dsm.time[dsm['time.hour'] == 12],  # take timestamp everytime hour=12
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
dso.elevation.attrs['grid_mapping'] = "nztm"

# Define the coordinate system using EPSG:2193
# crs = xr.DataArray(np.array(0),
#                    attrs={
#                        "grid_mapping_name": "transverse_mercator",
#                        "longitude_of_central_meridian": 173.0,
#                        "latitude_of_projection_origin": 0.0,
#                        "false_easting": 1600000.0,
#                        "false_northing": 10000000.0,
#                        "scale_factor_at_central_meridian": 0.9996,
#                        "semi_major_axis": 6378137.0,
#                        "inverse_flattening": 298.257222101,
#                        "spatial_ref": "EPSG:2193"
#                    }
#                    )

# from pyproj import CRS
# CRS('epsg:2193').to_cf()
# define crs as per ESRI instructions using ESRI WKT from epsg.io and adding additional description to make cf compliant in other programs.
# https://www.esri.com/arcgis-blog/products/arcgis/data-management/creating-netcdf-files-for-analysis-and-visualization-in-arcgis/
# https://epsg.io/2193
crs = xr.DataArray(np.array(0),
                   attrs={
                       "standard_name": 'crs',
                       "grid_mapping_name": "transverse_mercator",
                       "crs_wkt": ('PROJCS["NZGD_2000_New_Zealand_Transverse_Mercator",'
                                   'GEOGCS["GCS_NZGD_2000",'
                                   'DATUM["D_NZGD_2000",'
                                   'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
                                   'PRIMEM["Greenwich",0.0],'
                                   'UNIT["Degree",0.0174532925199433]],'
                                   'PROJECTION["Transverse_Mercator"],'
                                   'PARAMETER["False_Easting",1600000.0],'
                                   'PARAMETER["False_Northing",10000000.0],'
                                   'PARAMETER["Central_Meridian",173.0],'
                                   'PARAMETER["Scale_Factor",0.9996],'
                                   'PARAMETER["Latitude_Of_Origin",0.0],'
                                   'UNIT["Meter",1.0]]'),
                       "projected_crs_name": "NZGD2000 New Zealand Transverse Mercator",
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

#from what I can tell ESRI needs the netCDF to have a minimal set of attrubutes in the crs definition, for all variables (except for lat/lon/easting/northing/time) to have the grid_mapping set


crs_esri = xr.DataArray(np.array(0),
                   attrs={
                       "standard_name": 'crs',
                       "grid_mapping_name": "transverse_mercator",
                       "crs_wkt": ('PROJCS["NZGD_2000_New_Zealand_Transverse_Mercator",'
                                   'GEOGCS["GCS_NZGD_2000",'
                                   'DATUM["D_NZGD_2000",'
                                   'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
                                   'PRIMEM["Greenwich",0.0],'
                                   'UNIT["Degree",0.0174532925199433]],'
                                   'PROJECTION["Transverse_Mercator"],'
                                   'PARAMETER["False_Easting",1600000.0],'
                                   'PARAMETER["False_Northing",10000000.0],'
                                   'PARAMETER["Central_Meridian",173.0],'
                                   'PARAMETER["Scale_Factor",0.9996],'
                                   'PARAMETER["Latitude_Of_Origin",0.0],'
                                   'UNIT["Meter",1.0]]'),
                   }
                   )


first_time = da.time[0].dt.strftime('%Y%m%d%H%M').values
last_time = da.time[-1].dt.strftime('%Y%m%d%H%M').values
dso = dso.assign(snw=da, nztm=crs)
dso.to_netcdf('swe_{}_{}.nc'.format(first_time, last_time))

dso.assign(nztm=crs_esri).to_netcdf('swe_{}_{}_esri.nc'.format(first_time, last_time))

dso['dswe'] = dso.snw * np.nan
dso['dswe'][1:] = dso.snw.diff(dim='time', label='upper')  # label at end of the day - because the timestamp is in UTC it will return for the NZ day
# queried e.g.  ds_full.sel('2022-07-18') will return the value for the 24-hour change to midnight '2022-07-19 00:00+12:00'
dso['dswe'] = dso.dswe.assign_attrs(long_name='24-hour change in SWE', units='mm w.e.',
                                    description='change in snow pack storage in mm water equivalent for 24-hour preceeding timestamp', grid_mapping='nztm')

