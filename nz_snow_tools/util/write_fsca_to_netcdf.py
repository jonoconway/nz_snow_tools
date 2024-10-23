"""
writes arrays of fractional snow covered area to a net cdf file
"""

import netCDF4 as nc
from time import strftime, gmtime
import numpy as np

from nz_snow_tools.util.utils import nztm_to_wgs84

def create_ncvar_temperaure(ds, no_time=False):
    if no_time == False:
        temp_var = ds.createVariable('air_temperature', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    else:
        temp_var = ds.createVariable('air_temperature', 'f4', ('northing', 'easting',), zlib=True, complevel=4)
    temp_var.setncatts({
        'long_name': 'surface air temperature',
        'standard_name': 'air_temperature',
        'units': 'K',
        'description': "mean value over previous 1 hour",
        'cell_methods': 'time: mean',
        'missing': -9999.,
        'valid_min': 230.,
        'valid_max': 333.
    })
    return temp_var


def create_ncvar_shortwave(ds, no_time=False):
    if no_time == False:
        temp_var = ds.createVariable('surface_downwelling_shortwave_flux', 'f8', ('time', 'northing', 'easting',),
                                     zlib=True, complevel=4)
    else:
        temp_var = ds.createVariable('surface_downwelling_shortwave_flux', 'f8', ('northing', 'easting',), zlib=True,
                                     complevel=4)
    temp_var.setncatts({
        'long_name': 'downwelling shortwave radiation flux at surface',
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'units': 'W / m^2',
        'description': "mean value over previous 1 hour",
        'cell_methods': 'time: mean',
        'missing': -9999.,
        'valid_min': 0.,
        'valid_max': 1500.
    })
    return temp_var


def create_ncvar_precipitation(ds, no_time=False):
    if no_time == False:
        precip_var = ds.createVariable('precipitation_amount', 'f8', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    else:
        precip_var = ds.createVariable('precipitation_amount', 'f8', ('northing', 'easting',), zlib=True, complevel=4)
    precip_var.setncatts({
        'long_name': 'precipitation amount (mm)',
        'standard_name': 'precipitation_amount',
        'units': 'mm',
        'description': "total value over previous 1 hour",
        'cell_methods': 'time: sum',
        'missing': -9999.,
        'valid_min': 0.,
        'valid_max': 2000.
    })

    return precip_var


def create_ncvar_fsca(ds):
    fsca_var = ds.createVariable('fsca', 'u8', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    fsca_var.setncatts({
        'long_name': 'fractional snow covered area'
        # 'missing': -9999.,
        # 'valid_min': 0,
        # 'valid_max': 100
    })
    return fsca_var


def create_ncvar_swe(ds):
    swe_var = ds.createVariable('swe', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    swe_var.setncatts({
        'long_name': 'snow water equivalent',
        'missing': -9999.
        # 'valid_min': 0,
        # 'valid_max': 100
    })
    return swe_var


def create_ncvar_acc(ds):
    acc_var = ds.createVariable('acc', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    acc_var.setncatts({
        'long_name': 'snowfall in mm snow water equivalent',
        'missing': -9999.
        # 'valid_min': 0,
        # 'valid_max': 100
    })
    return acc_var


def create_ncvar_melt(ds):
    melt_var = ds.createVariable('melt', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    melt_var.setncatts({
        'long_name': 'melt in mm snow water equivalent',
        'missing': -9999.
        # 'valid_min': 0,
        # 'valid_max': 100
    })
    return melt_var


def create_ncvar_rain(ds):
    melt_var = ds.createVariable('rain', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    melt_var.setncatts({
        'long_name': 'rainfall amount (mm) ',
        'standard_name': 'rainfall_amount',
        'units': 'mm',
        'cell_methods': 'time: sum',
        'missing': -9999.,
        'valid_min': 0.,
        'valid_max': 2000.
    })
    return melt_var


def create_ncvar_ros(ds):
    melt_var = ds.createVariable('ros', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    melt_var.setncatts({
        'long_name': 'amount of rainfall (mm) occuring over snow surface (swe > 0) ',
        'standard_name': 'rainfall_onto_snow_amount',
        'units': 'mm',
        'cell_methods': 'time: sum',
        'missing': -9999.,
        'valid_min': 0.,
        'valid_max': 2000.
    })
    return melt_var

def create_ncvar_ros_melt(ds):
    melt_var = ds.createVariable('ros_melt', 'f4', ('time', 'northing', 'easting',), zlib=True, complevel=4)
    melt_var.setncatts({
        'long_name': 'amount of melt (mm) due to rainfall occuring over snow surface (swe > 0) ',
        'standard_name': 'surface_snow_melt_amount_due_to_rainfall',
        'units': 'mm',
        'cell_methods': 'time: sum',
        'missing': -9999.,
        'valid_min': 0.,
        'valid_max': 2000.
        })
    return melt_var



def create_lat_lons_for_NZTMgrid(extent_w=1.2e6, extent_e=1.4e6, extent_n=5.13e6, extent_s=4.82e6, resolution=250):
    """create grids of latitude and longitude corresponding to grid centres of data in nztm grid
    """
    # create coordinates
    x_centres = np.arange(extent_w + resolution / 2, extent_e, resolution)
    y_centres = np.arange(extent_s + resolution / 2, extent_n, resolution)
    y_array, x_array = np.meshgrid(y_centres, x_centres, indexing='ij')
    lon_array, lat_array = nztm_to_wgs84(y_array, x_array)
    return lat_array, lon_array


def write_nztm_grids_to_netcdf(fname, list_of_data_arrays, var_names, datetime_list, northings, eastings, lat_array,
                               lon_array, elevation, no_time=False):
    """
    Write a netCDF file containing fractional snow covered area data
    :param fname: string, full pathname of file to be created
    :param list_of_data_arrays: list, list containing data arrays to be saved [[time, northings, eastings],[time, northings, eastings]]
    :param var_names: list of strings corresponding to names of data arrays
    :param datetime_list: list of datetime objects corresponding to data
    :param northings: vector containing northings associated with data grid
    :param eastings: vector containing eastings associated with data grid
    :param lat_array: array containing longitudes of data grid
    :param lon_array: array containing latitudes of data grid
    :param elevation: array containing elevation of data grid

    :return:
    """

    ds = nc.Dataset(fname, 'w')

    # add common attributes
    ds.title = ''
    ds.source = ''

    ds.history = ''
    ds.references = ''
    ds.author = ''
    ds.email = ''
    ds.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if no_time == False:
        ds.featureType = "timeSeries"
    else:
        ds.comment = 'timestamp {}'.format(datetime_list.strftime('%Y%m%d%H%M'))
    ds.Conventions = "CF-1.6"

    if no_time == False:
        ds.createDimension('time', )
        t = ds.createVariable('time', 'f8', ('time',))
        t.long_name = "time"
        t.units = 'days since 1900-01-01 00:00:00'
        t[:] = nc.date2num(datetime_list, units=t.units)

    ds.createDimension('northing', len(northings))
    ds.createDimension('easting', len(eastings))
    ds.createDimension('latitude', len(northings))
    ds.createDimension('longitude', len(eastings))
    # add northing and easting dimensions as well as lat/lon variables
    t = ds.createVariable('northing', 'f8', ('northing',))
    t.axis = 'Y'
    t.long_name = "northing in NZTM"
    t.units = 'metres'
    t[:] = northings

    t = ds.createVariable('easting', 'f8', ('easting',))
    t.axis = 'X'
    t.long_name = "easting in NZTM"
    t.units = 'metres'
    t[:] = eastings

    t = ds.createVariable('lat', 'f8', ('northing', 'easting',))
    t.long_name = "latitude"
    t.standard_name = "latitude"
    t.units = "degrees north"
    t[:] = lat_array

    t = ds.createVariable('lon', 'f8', ('northing', 'easting',))
    t.long_name = "longitude"
    t.standard_name = "longitude"
    t.units = "degrees east"
    t[:] = lon_array

    elevation_var = ds.createVariable('elevation', 'f8', ('northing', 'easting',), fill_value=-9999.)
    elevation_var.long_name = "elevation (meters)"
    elevation_var.standard_name = "surface_altitude"
    elevation_var.units = "meters"
    elevation_var[:] = elevation

    if 'precipitation_amount' in var_names:
        precip_var = create_ncvar_precipitation(ds, no_time=no_time)
        precip_var[:] = list_of_data_arrays[var_names.index('precipitation_amount')]

    if 'air_temperature' in var_names:
        temp_var = create_ncvar_temperaure(ds, no_time=no_time)
        temp_var[:] = list_of_data_arrays[var_names.index('air_temperature')]

    if 'surface_downwelling_shortwave_flux' in var_names:
        temp_var = create_ncvar_shortwave(ds, no_time=no_time)
        temp_var[:] = list_of_data_arrays[var_names.index('surface_downwelling_shortwave_flux')]

    if 'fsca' in var_names:
        fsca_var = create_ncvar_fsca(ds)
        fsca_var[:] = list_of_data_arrays[var_names.index('fsca')]

    if 'swe' in var_names:
        swe_var = create_ncvar_swe(ds)
        swe_var[:] = list_of_data_arrays[var_names.index('swe')]

    ds.close()


def setup_nztm_grid_netcdf(fname, list_of_data_arrays, var_names, datetime_list, northings, eastings, lat_array,
                               lon_array, elevation, no_time=False):
    """
    Write a netCDF file containing fractional snow covered area data
    :param fname: string, full pathname of file to be created
    :param list_of_data_arrays: list, list containing data arrays to be saved [[time, northings, eastings],[time, northings, eastings]]
    :param var_names: list of strings corresponding to names of data arrays
    :param datetime_list: list of datetime objects corresponding to data
    :param northings: vector containing northings associated with data grid
    :param eastings: vector containing eastings associated with data grid
    :param lat_array: array containing longitudes of data grid
    :param lon_array: array containing latitudes of data grid
    :param elevation: array containing elevation of data grid

    :return:
    """

    ds = nc.Dataset(fname, 'w')

    # add common attributes
    ds.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    if no_time == False:
        ds.createDimension('time', )
        t = ds.createVariable('time', 'f8', ('time',))
        t.long_name = "time"
        t.units = 'hours since 1970-01-01 00:00:00'
        t[:] = nc.date2num(datetime_list, units=t.units)

    ds.createDimension('northing', len(northings))
    ds.createDimension('easting', len(eastings))
    # ds.createDimension('latitude', len(northings))
    # ds.createDimension('longitude', len(eastings))
    # add northing and easting dimensions as well as lat/lon variables
    t = ds.createVariable('northing', 'f8', ('northing',))
    t.axis = 'Y'
    t.long_name = "northing in NZTM"
    t.units = 'metres'
    t[:] = northings

    t = ds.createVariable('easting', 'f8', ('easting',))
    t.axis = 'X'
    t.long_name = "easting in NZTM"
    t.units = 'metres'
    t[:] = eastings

    t = ds.createVariable('lat', 'f8', ('northing', 'easting',))
    t.long_name = "latitude"
    t.standard_name = "latitude"
    t.units = "degrees north"
    t[:] = lat_array

    t = ds.createVariable('lon', 'f8', ('northing', 'easting',))
    t.long_name = "longitude"
    t.standard_name = "longitude"
    t.units = "degrees east"
    t[:] = lon_array

    elevation_var = ds.createVariable('elevation', 'f8', ('northing', 'easting',), fill_value=-9999.)
    elevation_var.long_name = "elevation (meters)"
    elevation_var.standard_name = "surface_altitude"
    elevation_var.units = "meters"
    elevation_var[:] = elevation

    if 'precipitation_amount' in var_names:
        precip_var = create_ncvar_precipitation(ds, no_time=no_time)

    if 'air_temperature' in var_names:
        temp_var = create_ncvar_temperaure(ds, no_time=no_time)

    if 'surface_downwelling_shortwave_flux' in var_names:
        temp_var = create_ncvar_shortwave(ds, no_time=no_time)

    if 'fsca' in var_names:
        fsca_var = create_ncvar_fsca(ds)

    if 'swe' in var_names:
        swe_var = create_ncvar_swe(ds)

    if 'acc' in var_names:
        acc_var = create_ncvar_acc(ds)

    if 'melt' in var_names:
        melt_var = create_ncvar_melt(ds)

    if 'rain' in var_names:
        rain_var = create_ncvar_rain(ds)

    if 'ros' in var_names:
        ros_var = create_ncvar_ros(ds)

    if 'ros_melt' in var_names:
        ros_melt_var = create_ncvar_ros_melt(ds)

    return ds
