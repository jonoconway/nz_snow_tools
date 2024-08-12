"""
a collection of utilities to handle dates / grids etc to support snow model
"""
from __future__ import division

import datetime
import numpy as np
from matplotlib.path import Path
from pyproj import Transformer



### Date utilities

def convert_dt_to_timestamp(dt_list):
    """
    returns a list with the timesince unix time for a given list of datetimes
    :param dt_list:
    :return:
    """
    timestamp = [(dt - datetime.datetime(1970, 1, 1)).total_seconds() for dt in dt_list]
    return timestamp


def convert_datetime_julian_day(dt_list):
    """ returns the day number for each date object in d_list
    :param d_list: list of dateobjects to evaluate
    :return: list of the julian day for each date
    """
    day = [(d - datetime.datetime(d.year, 1, 1)).days + 1 for d in dt_list]
    return day


def convert_unix_time_seconds_to_dt(uts_list):
    """ converts list of timestamps of seconds since the start of unix time to datetime objects
    :param uts_list: a list of times in seconds since the start of unix time
    :return: dt_list: list of datetime objects
    """
    dt_list = []
    for i in range(len(uts_list)):
        dt_list.append(datetime.datetime.utcfromtimestamp(uts_list[i]))
    return dt_list


def make_regular_timeseries(start_dt, stop_dt, num_secs):
    """
    makes a regular timeseries between two points. The difference between the start and end points must be a multiple of the num_secs
    :param start_dt: first datetime required
    :param stop_dt: last datetime required
    :param num_secs: number of seconds in timestep required
    :return: list of datetime objects between
    """
    epoch = datetime.datetime.utcfromtimestamp(0)
    st = (start_dt - epoch).total_seconds()
    et = (stop_dt - epoch).total_seconds()
    new_timestamp = np.linspace(st, et, int((et - st) / num_secs + 1))

    return convert_unix_time_seconds_to_dt(new_timestamp)


def convert_dt_to_hourdec(dt_list):
    """
    convert datetime to decimal hour
    :param dt_list:
    :return:
    """

    decimal_hour = [dt1.hour + dt1.minute / 60. + dt1.second / 3600. + dt1.microsecond / 3600000. for dt1 in dt_list]
    return np.asarray(decimal_hour)


def convert_datetime_decimal_day(dt_list):
    """ returns the decimal day number for each datetime object in dt_list
    :param dt_list: list of datetime objects to evaluate
    :return: numpy array of the decimal day for each datetime
    """
    timestamp = np.zeros(len(dt_list))
    for i in range(len(dt_list)):
        timestamp[i] = (dt_list[i] - datetime.datetime(dt_list[i].year, 1, 1)).total_seconds() / 86400. + 1
    return timestamp


def convert_date_hydro_DOY(d_list, hemis='south'):
    """ returns the day of the hydrological year for each date object in d_list
    :param d_list: list of dateobjects to evaluate
    :return: array containing the  day of the hydological year for each date
    """
    if hemis == 'south':
        end_month = 3
    elif hemis == 'north':
        end_month = 9
    h_DOY = []
    for d in d_list:
        if d.month <= end_month:  #
            h_DOY.append((d.date() - datetime.date(d.year - 1, end_month + 1, 1)).days + 1)
        else:
            h_DOY.append((d.date() - datetime.date(d.year, end_month + 1, 1)).days + 1)
    return np.asarray(h_DOY)


def convert_hydro_DOY_to_date(h_DOY, year, hemis='south'):
    """ converts day of the hydrological year into a datetime object
    :param d_list: array containing day of the hydological year
    :param year: the hydrological year the data is from . hydrological years are denoted by the year of the last day
    of the HY i.e. 2011 is HY ending 31 March 2011 in the SH
    :return: array containing datetime objects for a given hydrological year
    """
    if hemis == 'south':
        epoch = datetime.date(year - 1, 4, 1)
    elif hemis == 'north':
        epoch = datetime.date(year - 1, 10, 1)
    d = []
    for doy in h_DOY:
        d.append(epoch + datetime.timedelta(days=doy - 1))
    return np.asarray(d)


### temporal interpolation utilities

def process_precip(precip_daily, one_day=False):
    """
    Generate hourly precip fields using a multiplicative cascade model

    Described in Rupp et al 2009 (http://onlinelibrary.wiley.com/doi/10.1029/2008WR007321/pdf)

    :param model:
    :return:
    """

    if one_day == True:  # assume is 2d and add a time dimension on the start
        precip_daily = precip_daily.reshape([1, precip_daily.shape[0], precip_daily.shape[1]])

    if precip_daily.ndim == 2:
        hourly_data = np.zeros((precip_daily.shape[0] * 24, precip_daily.shape[1]), dtype=np.float32)
    elif precip_daily.ndim == 3:
        hourly_data = np.zeros((precip_daily.shape[0] * 24, precip_daily.shape[1], precip_daily.shape[2]), dtype=np.float32)

    store_day_weights = []
    for idx in range(precip_daily.shape[0]):
        # Generate an new multiplicative cascade function for the day
        day_weights = random_cascade(24)
        store_day_weights.append(day_weights)
        # multiply to create hourly data
        if precip_daily.ndim == 2:
            hourly_data[idx * 24: (idx + 1) * 24] = day_weights[:, np.newaxis] * precip_daily[idx]
        elif precip_daily.ndim == 3:
            hourly_data[idx * 24: (idx + 1) * 24] = day_weights[:, np.newaxis, np.newaxis] * precip_daily[idx]
    return hourly_data, store_day_weights


def random_cascade(l):
    """
    A recursive implementation of a Multiplicative Random Cascade (MRC)

    :param l: The size of the resulting output. Must be divisible by 2
    :return: Weights that sum to 1.0
    """
    res = np.ones(l)

    if l <= 3:
        res = np.random.random(l)
        res /= res.sum()
        return res

    weights = np.random.random(2)
    weights /= weights.sum()

    res[:l // 2] = weights[0] * random_cascade(l // 2)
    res[l // 2:] *= weights[1] * random_cascade(l // 2)
    return res


def process_temp(max_temp_daily, min_temp_daily):
    """
    Generate hourly fields

    Sine curve through max/min. 2pm/8am for max, min as a first gues
    :param model:
    :return:
    """

    def hour_func(dec_hours):
        """
        Piecewise fit to daily tmax and tmin using sine curves
        """
        f = np.piecewise(dec_hours, [dec_hours < 8, (dec_hours >= 8) & (dec_hours < 14), dec_hours >= 14], [
            lambda x: np.cos(2. * np.pi / 36. * (x + 10.)),  # 36 hour period starting 10 hours through
            lambda x: np.cos(2. * np.pi / 12. * (x - 14.)),  # 12 hour period (only using rising 6 hours between 8 am and 2pm)
            lambda x: np.cos(2. * np.pi / 36. * (x - 14.))  # 36 hour period starting at 2pm
        ])
        return (f + 1.) / 2.  # Set the range to be 0 - 1 0 is tmin and 1 being tmax

    scaling_factors = hour_func(np.arange(1., 25.))

    hourly_data = np.zeros((max_temp_daily.shape[0] * 24, max_temp_daily.shape[1]), dtype=np.float32)
    hours = np.array(list(range(1, 25)) * max_temp_daily.shape[0])

    # Calculate each piecewise element seperately as some need the previous days data
    mask = hours < 8
    max_temp_prev_day = np.concatenate((max_temp_daily[0][np.newaxis, :], max_temp_daily[:-1]))
    hourly_data[mask] = _interp_temp(scaling_factors[:7], max_temp_prev_day, min_temp_daily)

    mask = (hours >= 8) & (hours < 14)
    hourly_data[mask] = _interp_temp(scaling_factors[7:13], max_temp_daily, min_temp_daily)

    mask = hours >= 14
    min_temp_next_day = np.concatenate((min_temp_daily[1:], min_temp_daily[-1][np.newaxis, :]))
    hourly_data[mask] = _interp_temp(scaling_factors[13:], max_temp_daily, min_temp_next_day)

    return hourly_data


def _interp_temp(scaling, max_temp, min_temp):
    res = np.zeros((len(scaling) * len(max_temp),) + max_temp.shape[1:])

    for i in range(len(max_temp)):
        res[i * len(scaling): (i + 1) * len(scaling)] = scaling[:, np.newaxis] * max_temp[i] + (1 - scaling[:, np.newaxis]) * min_temp[i]
    return res



def process_temp_flex(max_temp_daily, min_temp_daily,hr_min=8,hr_max=15):
    """
    Generate hourly fields

    Sine curve through max/min. 2pm/8am for max, min as a first gues
    :param model:
    :return:
    """

    def hour_func_flex(dec_hours,hr_min,hr_max):
        """
        Piecewise fit to daily tmax and tmin using sine curves
        """
        r = hr_max-hr_min
        r1 = 24 - r
        f = np.piecewise(dec_hours, [dec_hours < hr_min, (dec_hours >= hr_min) & (dec_hours < hr_max), dec_hours >= hr_max], [
            lambda x: np.cos(2. * np.pi / (2.*r1) * (x + r1 - hr_min)),  # 36 hour period starting 10 hours through
            lambda x: np.cos(2. * np.pi / (2.*r) * (x - hr_max)),  # 12 hour period (only using rising 6 hours between 8 am and 2pm)
            lambda x: np.cos(2. * np.pi / (2.*r1) * (x - hr_max))  # 36 hour period starting at 2pm
        ])
        return (f + 1.) / 2.  # Set the range to be 0 - 1 0 is tmin and 1 being tmax

    scaling_factors = hour_func_flex(np.arange(1., 25.),hr_min,hr_max)

    hourly_data = np.zeros((max_temp_daily.shape[0] * 24, max_temp_daily.shape[1]), dtype=np.float32)
    hours = np.array(list(range(1, 25)) * max_temp_daily.shape[0])

    # Calculate each piecewise element seperately as some need the previous days data
    mask = hours < hr_min
    max_temp_prev_day = np.concatenate((max_temp_daily[0][np.newaxis, :], max_temp_daily[:-1]))
    hourly_data[mask] = _interp_temp(scaling_factors[:hr_min-1], max_temp_prev_day, min_temp_daily)

    mask = (hours >= hr_min) & (hours < hr_max)
    hourly_data[mask] = _interp_temp(scaling_factors[hr_min-1:hr_max-1], max_temp_daily, min_temp_daily)

    mask = hours >= hr_max
    min_temp_next_day = np.concatenate((min_temp_daily[1:], min_temp_daily[-1][np.newaxis, :]))
    hourly_data[mask] = _interp_temp(scaling_factors[hr_max-1:], max_temp_daily, min_temp_next_day)

    return hourly_data


def daily_to_hourly_temp_grids_new(max_temp_grid, min_temp_grid, single_dt=False,dt_step = 3600,time_min=9,time_max=2):
    """
    run through and process daily data into hourly, one slice at a time.
    :param max_temp_grid: input data with dimension [time,y,x]
    :param min_temp_grid: input data with dimension [time,y,x]
    :return: hourly data with dimension [time*24,y,x]
    """
    if dt_step != 3600:
        print('only set up for hourly timestep - check dt_step')

    if single_dt == True:  # assume is 2d and add a time dimension on the start
        max_temp_grid = max_temp_grid.reshape([1, max_temp_grid.shape[0], max_temp_grid.shape[1]])
        min_temp_grid = min_temp_grid.reshape([1, min_temp_grid.shape[0], min_temp_grid.shape[1]])
    hourly_grid = np.empty([max_temp_grid.shape[0] * int(86400/dt_step), max_temp_grid.shape[1], max_temp_grid.shape[2]], dtype=np.float32) * np.nan
    for i in range(max_temp_grid.shape[1]):
        hourly_grid[:, i, :] = process_temp_flex(max_temp_grid[:, i, :], min_temp_grid[:, i, :],hr_min=time_min,hr_max=time_max)
    return hourly_grid


def daily_to_hourly_swin_grids_new(swin_grid, lats, lons, hourly_dt, single_dt=False):
    """
    converts daily mean SW into hourly using TOA rad, applying

    :param hi_res_sw_rad: daily sw in data with dimension [time,y,x]
    :param hourly_dt timezone naive datetime
    :return:
    """
    if single_dt == True:  # assume is 2d and add a time dimension on the start
        swin_grid = swin_grid.reshape([1, swin_grid.shape[0], swin_grid.shape[1]])

    num_steps_in_day = int(86400. / (hourly_dt[1] - hourly_dt[0]).total_seconds())
    hourly_grid = np.ones([swin_grid.shape[0] * num_steps_in_day, swin_grid.shape[1], swin_grid.shape[2]])

    lon_ref = np.mean(lons)
    lat_ref = np.mean(lats)
    # compute hourly TOA for reference in middle of domain #TODO explicit calculation for each grid point?
    toa_ref = calc_toa(lat_ref, lon_ref, hourly_dt)
    # compute daily average TOA and atmospheric transmissivity
    daily_av_toa = []
    for i in range(0, len(toa_ref), num_steps_in_day):
        daily_av_toa.append(np.mean(toa_ref[i:i + num_steps_in_day]))
    daily_trans = swin_grid / np.asarray(daily_av_toa)[:, np.newaxis, np.newaxis]
    # calculate hourly sw from daily average transmisivity and hourly TOA
    for ii, i in enumerate(range(0, len(toa_ref), num_steps_in_day)):
        hourly_grid[i:i + num_steps_in_day] = hourly_grid[i:i + num_steps_in_day] * toa_ref[i:i + num_steps_in_day, np.newaxis, np.newaxis] * daily_trans[ii]

    return hourly_grid


# grid utilities

def create_mask_from_shpfile(lat, lon, shp_path, idx=0):
    """
    Creates a mask for numpy array

    creates a boolean array on the same grid as lat,long where all cells,
    whose centroid is inside a line shapefile, are true. The shapefile must be
    a line, in the same CRS as the data, and have only 1 feature (only the first
    feature will be used as a mask)

    :param lat: a 1D array of latitiude positions on a regular grid
    :param long: a 1D array of longitute position on a regular grid
    :param shp_path: the path to a line shape file (.shp) only the first feature will be used to mask.
    :return: boolean array on the same grid as lat,long
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim == 1 and lon.ndim == 1:  # create array of lat and lon
        nx, ny = len(lon), len(lat)
        longarray, latarray = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        assert lat.shape == lon.shape
        nx, ny = lat.shape[1], lat.shape[0]
        longarray = lon
        latarray = lat
    else:
        raise ValueError("lat and lon are not the same shape")

    # Create vertex coordinates for each grid cell
    x, y = longarray.flatten(), latarray.flatten()
    points = np.vstack((x, y)).T

    if '.shp' in shp_path:
        # load shapefile
        import shapefile
        shp = shapefile.Reader(shp_path)
        shapes2 = shp.shapes()
        shppath = Path(shapes2[idx].points)

        # check to find errors at start of file, and remove points
        diffs = np.sqrt(np.diff(shppath.vertices[:], axis=0)[:, 0] ** 2 + np.diff(shppath.vertices[:], axis=0)[:, 1] ** 2)
        where_large_diff = np.where(diffs > 5 * np.mean(diffs))[0]
        if where_large_diff.size > 0:  # if large jumps in distance
            # find indexes of parts
            part_idx = np.concatenate([shapes2[idx].parts[:],np.array((len(shapes2[idx].points),))])
            keep_idx = np.full(len(shapes2[idx].points),True)
            # check to see if any parts correspond to a large jump
            for i, start_idx in enumerate(shapes2[idx].parts[:]):
                if start_idx - 1 in where_large_diff:
                    # remove points from this part
                    keep_idx[part_idx[i]:part_idx[i+1]] = False
            # remove bad points
            shppath.vertices = shppath.vertices[keep_idx, :]
            print('trimmed {} points from shapefile {}'.format(sum(~keep_idx), shp_path))

        # plt.scatter(shppath.vertices[:,0],shppath.vertices[:,1])
        # plt.scatter(shppath.vertices[:200,0],shppath.vertices[:200,1],color='r')
        # plt.scatter(shppath.vertices[200:,0],shppath.vertices[200:,1],color='g') # green points plot over red, so first red points are duplicate and can be excluded

        # create the mask array
        grid = shppath.contains_points(points)
        grid = grid.reshape((ny, nx))

    elif '.gpkg' in shp_path:
        import geopandas as gpd
        from shapely.geometry import Point
        import fiona

        if 'catchment' in fiona.listlayers(shp_path):
            shp = gpd.read_file(shp_path,layer='catchment')
        elif len(fiona.listlayers(shp_path)) == 0:
            shp = gpd.read_file(shp_path)
        else:
            print('multiple layers with no catchment layer in geopackage file {}'.format(shp_path))

        grid = np.asarray([Point(p).intersects(shp.geometry[0]) for p in points])

        # nx, ny = len(lon), len(lat)
        grid = grid.reshape((ny, nx))

    return grid


def nztm_to_wgs84(in_y, in_x):
    """converts from NZTM to WGS84  Inputs and outputs can be arrays.
    updated to pyproj2 style https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrade-transformer
    """
    transformer = Transformer.from_crs('EPSG:2193', 'EPSG:4326')
    out_x, out_y = transformer.transform(in_x, in_y)
    return out_y, out_x


def wgs84_to_nztm(in_y, in_x):
    """converts from WGS84  to NZTM Inputs and outputs can be arrays.
    updated to pyproj2 style https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrade-transformer
    """
    transformer = Transformer.from_crs('EPSG:4326','EPSG:2193')
    out_x, out_y = transformer.transform(in_x, in_y)
    return out_y, out_x


def trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres):
    # Trim down the number of latitudes requested so it all stays in memory
    valid_lat_bounds = np.nonzero(mask.sum(axis=1))[0]
    lat_min_idx = valid_lat_bounds.min()
    lat_max_idx = valid_lat_bounds.max()
    valid_lon_bounds = np.nonzero(mask.sum(axis=0))[0]
    lon_min_idx = valid_lon_bounds.min()
    lon_max_idx = valid_lon_bounds.max()

    lats = lat_array[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    lons = lon_array[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    elev = nztm_dem[lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1]
    northings = y_centres[lat_min_idx:lat_max_idx + 1]
    eastings = x_centres[lon_min_idx:lon_max_idx + 1]

    return lats, lons, elev, northings, eastings


def resample_to_fsca(snow_grid, rl):
    """

    :param snow_grid: grid of fractional or binary (0/1 snow)
    :param rl: resample length - the number of grid cells in each direction to include in new fsca. e.g. 5  = 25 grid points to fsca
    :return: fcsa = fractional snow covered area for the same area as snow_grid, but may be smaller if grid size is not a multiple of resample length
    """
    ny = snow_grid.shape[0]
    nx = snow_grid.shape[1]
    ny_out = ny // rl  # integer divide to ensure fits
    nx_out = nx // rl

    fsca = np.zeros((ny_out, nx_out))

    for i in range(ny_out):
        for j in range(nx_out):
            snow = snow_grid[i * rl:(i + 1) * rl, j * rl:(j + 1) * rl]
            fsca[i, j] = np.sum(snow) / (rl * rl)

    return fsca


# misc

def calc_toa(lat_ref, lon_ref, hourly_dt):
    """
    calculate top of atmopshere radiation for given lat, lon and datetime
    :param lat_ref:
    :param lon_ref:
    :param hourly_dt: timezone naive datetime that is in the local_time as specified by the timezone variable
    :return:
    """
    dtstep = (hourly_dt[1] - hourly_dt[0]).total_seconds()
    # compute at midpoint between timestep and previous timestep
    jd = convert_datetime_decimal_day(hourly_dt) - 0.5 * (hourly_dt[1] - hourly_dt[0]).total_seconds() / 86400.
    hourdec = convert_dt_to_hourdec(hourly_dt) - 0.5 * (hourly_dt[1] - hourly_dt[0]).total_seconds() / 3600.

    latitude = lat_ref  # deg
    longitude = lon_ref  # deg
    timezone = -12
    # Calculating    correction    factor    for direct beam radiation
    d0_rad = 2 * np.pi * (jd - 1) / 365.  # day    angle,
    # solar        declination        Iqbal, 1983
    Declination_rad = np.arcsin(
        0.006918 - 0.399912 * np.cos(d0_rad) + 0.070257 * np.sin(d0_rad) - 0.006758 * np.cos(2 * d0_rad)
        + 0.000907 * np.sin(2 * d0_rad) - 0.002697 * np.cos(3 * d0_rad) + 0.00148 * np.sin(3 * d0_rad))
    HourAngle_rad = (-1 * (180 - hourdec * 15) + (longitude - 15 * timezone)) * np.pi / 180.
    ZenithAngle_rad = np.arccos(np.cos(latitude * np.pi / 180) * np.cos(Declination_rad) * np.cos(HourAngle_rad)
                                + np.sin(latitude * np.pi / 180) * np.sin(Declination_rad))
    ZenithAngle_deg = ZenithAngle_rad * 180 / np.pi
    sundown = 0 * ZenithAngle_deg
    sundown[ZenithAngle_deg > 90] = 1
    E0 = 1.000110 + 0.034221 * np.cos(d0_rad) + 0.00128 * np.sin(d0_rad) + 0.000719 * np.cos(
        2 * d0_rad) + 0.000077 * np.sin(2 * d0_rad)
    SRtoa = 1372 * E0 * np.cos(ZenithAngle_rad)  # SRin    at    the    top    of    the    atmosphere
    SRtoa[sundown == 1] = 0
    return SRtoa


def setup_nztm_dem(dem_file, extent_w=1.2e6, extent_e=1.4e6, extent_n=5.13e6, extent_s=4.82e6, resolution=250, origin='bottomleft'):
    """
    load dem tif file. defaults to clutha 250 dem.
    :param dem_file: string specifying path to dem
    :param extent_w: extent in nztm
    :param extent_e: extent in nztm
    :param extent_n: extent in nztm
    :param extent_s: extent in nztm
    :param resolution: resolution in m
    :param origin: option to specify whether you want the dem to have its origin in the 'bottomleft' (i.e. xy) or in the 'topleft' ie. ij
    :return:
    """
    from PIL import Image
    if dem_file is not None:
        nztm_dem = Image.open(dem_file)
        if origin == 'bottomleft':
            # np.array(nztm_dem).shape is (1240L, 800L) but origin is in NW corner. Move to SW to align with increasing Easting and northing.
            nztm_dem = np.flipud(np.array(nztm_dem))
        if origin == 'topleft':
            nztm_dem = np.array(nztm_dem)
    else:
        nztm_dem = None
    # extent_w = 1.2e6
    # extent_e = 1.4e6
    # extent_n = 5.13e6
    # extent_s = 4.82e6
    # resolution = 250
    # create coordinates
    x_centres = np.arange(extent_w + resolution / 2, extent_e, resolution)
    y_centres = np.arange(extent_s + resolution / 2, extent_n, resolution)
    if origin == 'topleft':
        y_centres = y_centres[::-1]
    y_array, x_array = np.meshgrid(y_centres, x_centres, indexing='ij')  # this makes an array with northings and eastings increasing
    lat_array, lon_array = nztm_to_wgs84(y_array, x_array)
    # plot to check the dem
    # plt.imshow(nztm_dem, origin=0, interpolation='none', cmap='terrain')
    # plt.colorbar(ticks=np.arange(0, 3000, 100))
    # plt.show()
    return nztm_dem, x_centres, y_centres, lat_array, lon_array


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


def nash_sut(y_sim, y_obs):
    """
    calculate the nash_sutcliffe efficiency criterion (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1

    ns = 1 - np.sum((y_sim - y_obs) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)

    return ns


def mean_bias(y_sim, y_obs):
    """
    calculate the mean bias difference (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    mbd = np.sum(y_sim - y_obs) / len(y_sim)

    return mbd


def rmsd(y_sim, y_obs):
    """
    calculate the mean bias difference (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    rs = np.sqrt(np.mean((y_sim - y_obs) ** 2))

    return rs


def mean_absolute_error(y_sim, y_obs):
    """
    calculate the mean absolute error

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    mbd = np.sum(np.abs(y_sim - y_obs)) / len(y_sim)

    return mbd


def coef_determ(y_sim, y_obs):
    """
    calculate the coefficient of determination

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    r = np.corrcoef(y_sim, y_obs)
    r2 = r[0, 1] ** 2

    return r2


def basemap_interp(datain, xin, yin, xout, yout, interpolation='NearestNeighbour'):
    """
       Interpolates a 2D array onto a new grid (only works for linear grids),
       with the Lat/Lon inputs of the old and new grid. Can perfom nearest
       neighbour interpolation or bilinear interpolation (of order 1)'

       This is an extract from the basemap module (truncated)

       https://github.com/matplotlib/basemap/blob/3646ba1122a52cea0779b7e08dd1036a1b7ff200/packages/basemap/src/mpl_toolkits/basemap/__init__.py#L4882

    """

    # Mesh Coordinates so that they are both 2D arrays
    # xout, yout = np.meshgrid(xout, yout)
    # ensure that coordiates are both 2D arrays
    assert xout.shape == yout.shape
    assert xout.shape.ndim == 2

    # compute grid coordinates of output grid.
    xcoords = (len(xin) - 1) * (xout - xin[0]) / (xin[-1] - xin[0])
    ycoords = (len(yin) - 1) * (yout - yin[0]) / (yin[-1] - yin[0])

    xcoords = np.clip(xcoords, 0, len(xin) - 1)
    ycoords = np.clip(ycoords, 0, len(yin) - 1)

    # Interpolate to output grid using nearest neighbour
    if interpolation == 'NearestNeighbour':
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi, xcoordsi]

    # Interpolate to output grid using bilinear interpolation.
    elif interpolation == 'Bilinear':
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi + 1
        yip1 = yi + 1
        xip1 = np.clip(xip1, 0, len(xin) - 1)
        yip1 = np.clip(yip1, 0, len(yin) - 1)
        delx = xcoords - xi.astype(np.float32)
        dely = ycoords - yi.astype(np.float32)
        dataout = (1. - delx) * (1. - dely) * datain[yi, xi] + \
                  delx * dely * datain[yip1, xip1] + \
                  (1. - delx) * dely * datain[yip1, xi] + \
                  delx * (1. - dely) * datain[yi, xip1]

    return dataout


def fill_timeseries_dud(inp_dt, inp_dat, tstep, max_gap=None):
    """
        fill in gaps in a timeseries using linear interpolation between valid points (method doesn't work that well)

    :param inp_dt: array or list of datetimes corresponding to your data
    :param inp_dat: variable you wish to fill
    :param tstep: timestep of data in seconds
    :param max_gap: maximum gap  to fill (seconds)
    :return: out_dt, out_dat (datetimes and data)
    """
    assert len(inp_dt) == len(inp_dat)

    out_dt = []
    out_dat = []
    if max_gap == None:
        max_gap = tstep * 1.
    for j in range(len(inp_dt) - 1):
        gap = (inp_dt[j + 1] - inp_dt[j]).total_seconds()
        if gap != tstep and gap <= max_gap:
            fill_dt = make_regular_timeseries(inp_dt[j], inp_dt[j + 1], tstep)
            fill_dat = np.interp(convert_dt_to_timestamp(fill_dt), convert_dt_to_timestamp(inp_dt[j:j + 2]), inp_dat[j:j + 2])
            out_dt.extend(fill_dt[1:-1])
            out_dat.extend(fill_dat[1:-1])
        elif gap != tstep and gap >= max_gap:
            fill_dt = make_regular_timeseries(inp_dt[j], inp_dt[j + 1], tstep)
            fill_dat = np.ones(len(fill_dt)) * np.nan
            out_dt.extend(fill_dt[1:-1])
            out_dat.extend(fill_dat[1:-1])
        else:
            out_dt.extend([inp_dt[j]])
            out_dat.extend([inp_dat[j]])

    return np.asarray(out_dt), np.asarray(out_dat, dtype=inp_dat.dtype)


def fill_timeseries(inp_dt, inp_dat, tstep):
    """
    fill in gaps in a timeseries using linear interpolation between valid points

    :param inp_dt: array or list of datetimes corresponding to your data
    :param inp_dat: variable you wish to fill
    :param tstep: timestep of data in seconds
    :return: out_dt, out_dat (datetimes and data)
    """
    assert len(inp_dt) == len(inp_dat)

    out_dt = make_regular_timeseries(inp_dt[0], inp_dt[-1], tstep)
    out_dat = np.interp(convert_dt_to_timestamp(out_dt), convert_dt_to_timestamp(inp_dt), inp_dat)

    return np.asarray(out_dt), out_dat


def ws_wd_from_u_v(u,v):
    """ function to calculate geographic wind speed and direction with respect to north from u and v components
    """
    wd = np.rad2deg(np.arctan2(-u, -v))
    ws = np.sqrt((u ** 2) + (v ** 2))
    return ws, wd


def u_v_from_ws_wd(ws,wd):
    """ function to calculate geographic wind speed and direction with respect to north from u and v components
    """
    u = -ws * np.sin(np.deg2rad(wd))
    v = -ws * np.cos(np.deg2rad(wd))
    return u, v


def ea_from_tc_rh(tc, rh, pres_hpa=None):
    # vapour pressure in hPa calculated according to Buck
    # Rh should be with respect to water, not ice.
    if pres_hpa == None:
        ea = 6.1121 * np.exp(17.502 * tc / (240.97 + tc)) * rh / 100
    else:
        ea = (1.0007 + (3.46 * 1e-6 * pres_hpa)) * 6.1121 * np.exp(17.502 * tc / (240.97 + tc)) * rh / 100
    return ea


def blockfun(datain, blocksize, filter=None, method='mean', keep_nan=False):
    """
    function to block average data using nan mean. can filter out bad values, and can upsample data. use negative blocksize to upsample data.
    :param datain_copy: array with data to average. data is avearged along First dimension
    :param blocksize: interger specifying the number of elements to average. if positive, must be a multiple of the length of datain. if negative, data is upsampled.
    :param filter: values to filter out. must have same length as first dimension of datain. all values with filter == True are set to nan
    :param method 'mean' or 'sum'
    :param keep_nan True = preserves nans in input, so will return nan if any nan's within the block.
    :return:
    """

    # datain = np.ones([192,2])
    # blocksize = 96
    datain_copy = datain.copy()
    #
    #     if datain_copy.ndim > 1:
    #         datain_copy[filter, :] = np.nan
    #     else:
    #         datain_copy[filter] = np.nan
    filter_array = np.full(datain_copy.shape, True)
    if filter is not None:
        if datain_copy.ndim > 1:
            filter_array[filter, :] = False
        else:
            filter_array[filter] = False

    num_rows = datain_copy.shape[0]

    if blocksize > 0:  # downsample
        if datain_copy.ndim > 1:
            num_col = datain_copy.shape[1]
            dataout = np.ones([int(num_rows / blocksize), num_col])
            for ii, i in enumerate(range(0, num_rows, blocksize)):
                for j in range(num_col):
                    block = datain_copy[i:i + blocksize, j]
                    block = block[filter_array[i:i + blocksize, j]]
                    if method == 'mean':
                        if keep_nan:
                            dataout[ii, j] = np.mean(block)
                        else:
                            dataout[ii, j] = np.nanmean(block)
                    elif method == 'sum':
                        if keep_nan:
                            dataout[ii, j] = np.sum(block)
                        else:
                            dataout[ii, j] = np.nansum(block)
        else:
            dataout = np.ones([int(num_rows / blocksize), ])
            for ii, i in enumerate(range(0, num_rows, blocksize)):
                block = datain_copy[i:i + blocksize]
                block = block[filter_array[i:i + blocksize]]
                if method == 'mean':
                    if keep_nan:
                        dataout[ii] = np.mean(block)
                    else:
                        dataout[ii] = np.nanmean(block)

                elif method == 'sum':
                    if keep_nan:
                        dataout[ii] = np.sum(block)
                    else:
                        dataout[ii] = np.nansum(block)

    elif blocksize < 0:  # upsample ignoring the filter

        blocksize_copy = blocksize * -1

        if datain_copy.ndim > 1:
            dataout = np.ones([int(num_rows * blocksize_copy), datain_copy.shape[1]])
            for i in range(num_rows):
                if method == 'mean':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy, :] = datain_copy[i, :]
                elif method == 'sum':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy, :] = datain_copy[i, :] / blocksize_copy
        else:
            dataout = np.ones([int(num_rows * blocksize_copy), ])
            for i in range(num_rows):
                if method == 'mean':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy] = datain_copy[i]
                if method == 'sum':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy] = datain_copy[i] / blocksize_copy

    return dataout
