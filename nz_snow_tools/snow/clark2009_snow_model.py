"""
core model to calculate evolution of snow cover
"""

from __future__ import division

import numpy as np
import netCDF4 as nc

from nz_snow_tools.util.utils import convert_dt_to_hourdec, convert_datetime_julian_day


def calc_melt_Clark2009(ta, precip, d_snow, doy, acc, tmelt=274.16, **config):
    """
    calculate the melt rate (mm day^-1) at current timestep according to Clark 2009
    :param ta: grid of current temperature (K)
    :param precip: grid of current precipitation (mm)
    :param d_snow: grid with current days since last snowfall (days)
    :param doy: current day of year (can be decimal)
    :param acc: grid of accumulation in current timestep (mm)
    :param tmelt: temperature threshold for melt (K)
    :return: grid of melt rate at current timestep (mm day^-1)
    """

    # determine melt factor
    mf = calc_melt_factor_Clark2009(doy, d_snow, precip, acc,
                                    **config)  # mf_mean=config['mf_mean'], mf_amp=config['mf_amp'], mf_alb=config['mf_alb'],                                    mf_alb_decay=config['mf_alb_decay'], mf_ros=config['mf_ros']
    melt_rate = mf * (ta - tmelt)
    melt_rate[(ta < tmelt)] = 0
    return melt_rate


def calc_melt_factor_Clark2009(doy, d_snow, precip, acc, mf_mean=5.0, mf_amp=5.0, mf_alb=2.5, mf_alb_decay=5.0,
                               mf_ros=2.5, mf_doy_max_ddf=356, **config):
    """
    calculate the grid melt factor according to Clark 2009. Depends on day of year, time since snowfall and rain on snow.
    :param doy: current day of year
    :param d_snow: grid of days since last snowfall
    :param precip: grid of precip at current timestep
    :param acc: grid of accumulation at current timestep
    :param mf_mean: Mean melt factor
    :param mf_amp: Seasonal amplitude in the melt factor
    :param mf_alb: Decrease in melt factor due to higher fresh snow albedo
    :param mf_alb_decay: Timescale for decrease in snow albedo
    :param mf_ros: Increase in the melt factor in rain-on-snow events
    :param mf_doy_max_ddf: Day of year for maximum melt factor. Minimum ddf is 6 month opposite the max.
    :return: grid of melt factors at current timestep
    """

    # compute change in melt factor due to season
    dmf_seas = mf_amp * np.sin(2 * np.pi * (doy - mf_doy_max_ddf + 91.5) / 366)

    # compute change in melt factor due to increased albedo after snowfall
    dmf_alb = - mf_alb * np.exp(- d_snow / mf_alb_decay)

    # compute change in melt factor due to rain on snow
    dmf_ros = np.zeros(precip.shape)
    dmf_ros[(precip - acc > 0)] = mf_ros

    # compute melt factor, limiting to 0 to avoid spurious accumulation
    mf = np.max([mf_mean + dmf_seas + dmf_alb + dmf_ros, np.zeros(precip.shape)])

    return mf


def calc_acc(ta, precip, tacc=274.16, **config):
    """
    calculate accumulation when precip is below threshold
    :param ta: grid of current temperature (K)
    :param precip: grid of current precipitation (mm)
    :param tacc: temperature threshold for accumlation  (K)
    :return: acc: grid of accumulation in current timestep (mm)
    """
    acc = precip.copy()
    acc[(ta >= tacc)] = 0

    return acc


def calc_melt_dsc_snow(ta, sw, d_snow, swe, tmelt=274.16, tf=0.04 * 24, rf=0.009 * 24, alb=None, precip_per_s=None,**config):
    """
    calculate the melt rate (mm day^-1) at current timestep according to dsc_snow model. calculates albedo if not available in the configuration
    :param ta: grid of current temperature (K)
    :param precip: grid of current precipitation (mm)
    :param sw: grid of current incoming shortwave rad (W m^-2)
    :param d_snow: grid with current days since last snowfall (days)
    :param swe: grid with current swe (mm w.e.)
    :param tmelt: temperature threshold for melt (K)
    :return: grid of melt rate at current timestep (mm day^-1)
    """
    #
    if alb is None:
        alb = calc_albedo_snow(d_snow, swe, **config)
    # dc=config['dc'], tc=config['tc'], a_ice=config['a_ice'], a_freshsnow=config['a_freshsnow'], a_firn=config['a_firn']
    sw_net = sw * (1 - alb)
    if config['ros'] == True:
        #  specific heat capacity of liquid water = 4220  J / (kg * K)
        qprc = 4220 * precip_per_s * (ta - 273.16)
        # use theoretical value for precipitation melt factor 0.0108  mm w.e. per hour per W m-2
        if config['ta_m_tt'] == True: # use tmelt as baseline when calculating degree days
            melt_rate = tf * (ta - tmelt) + rf * sw_net + 0.0108 * 24 * qprc
        else:
            melt_rate = tf * (ta - 273.16) + rf * sw_net + 0.0108 * 24 * qprc
    else:
        if config['ta_m_tt'] == True: # use tmelt as baseline when calculating degree days
            melt_rate = tf * (ta - tmelt) + rf * sw_net
        else:
            melt_rate = tf * (ta - 273.16) + rf * sw_net
    melt_rate[(ta < tmelt)] = 0
    return melt_rate


def calc_albedo_snow(ts, snow, dc=11.0, tc=21.9, a_ice=0.34, a_freshsnow=0.9, a_firn=0.53, **config):
    """
    !   albedo calculated using 'time' - based on time since last snowfall(Oerlemans + Knap 1998)
    !   albedo = albedo    output    grid
    !   ts = time(days)    since    last    snowfall
    !   snow = grid    of    snow    thickness( in mm    w.e.)
    !   dc = 'characteristic depth'(11    mm    w.e.)
    !   tc = 'characterstic time scale'(21.9    days)
    !   a_ice = albedo    of    ice    surface(0.34)
    !   a_freshsnow = albedo    of    fresh    snow    surface(0.9)
    !   a_firn = albedo    of    firn    surface(0.53)
    """
    a_snow = a_firn + (a_freshsnow - a_firn) * np.exp(-ts / tc)
    albedo = a_snow + (a_ice - a_snow) * np.exp(-snow / dc)
    return albedo


def calc_dswe(swe, d_snow, ta, precip, doy, dtstep, sw=None, which_melt='clark2009', **config):
    """
    calculate the change in swe for a given timestep (mm water equivalent)
    :param swe: grid with current state of SWE (mm water equivalent)
    :param d_snow: grid with current days since last snowfall (days)
    :param ta: grid of current temperature (K)
    :param precip: grid of current precipitation (mm)
    :param doy: current day of year (can be decimal)
    :return: swe: updated grid of swe at end of current timestep (mm w.e.)
    """
    acc = calc_acc(ta, precip, **config)
    swe = swe + acc  # Precip is in mm. so doesn't need time dependence
    # d_snow[(acc > 0)] = 0  # reset days since snowfall when is new snow # now handled with daily threshold in snow_main or snow_main_simple
    if which_melt == 'clark2009':
        melt_rate = calc_melt_Clark2009(ta, precip, d_snow, doy, acc, **config)  # tmelt=config['tmelt'],
    elif which_melt == 'dsc_snow':
        melt_rate = calc_melt_dsc_snow(ta, sw, d_snow, swe, precip_per_s=precip/dtstep,**config)  # tmelt=config['tmelt'], tf=config['tf'], rf=config['rf'],
    melt = melt_rate * dtstep / 86400.0
    melt[(melt >= swe)] = swe[(melt >= swe)]  # limit melt to total swe
    swe = swe - melt
    # swe[(swe < 0)] = 0  # reset to 0 if all snow melts

    return swe, d_snow, melt, acc


def snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep, init_swe=None, init_d_snow=None, inp_sw=None, which_melt='clark2009', alb_swe_thres=5.0, \
                     num_secs_output=86400, **config):
    """
    main snow model loop. handles timestepping and storage of output.
    assumes input data is at same temporal and spatial resolution as model time step (dtstep). Output is writen at the end of each day, then passed out
    :param inp_ta: air temperature in K. dimensions (time,spatial:)
    :param inp_precip: precipitation in mm, dimensions (time,spatial:)
    :param inp_doy: day of year, dimensions (time)
    :param inp_hourdec: decimal hour corresponding to the input data, dimensions (time)
    :param dtstep: timestep (seconds) of input data
    :param init_swe: inital grid of snow water equivalent (SWE; mm w.e.), dimensions (spatial:)
    :param init_d_snow: inital grid of times since last snowfall dimensions (spatial:)
    :param inp_ta: incoming sw radiation in Wm^-2. dimensions (time,spatial:)
    :param which_melt: string specifying which melt model to be run. options include 'clark2009', 'dsc_snow'
    :param alb_swe_thres: threshold for daily snowfall resetting albedo (mm w.e). calculated from daily swe change so that accumulation must be > melt
    :param  **config: configuration dictionary
    :return: st_swe - calculated SWE (mm w.e.) at the end of each day. (n = number of days + 1)
    """

    num_timesteps = inp_ta.shape[0]

    storage_interval = num_secs_output / dtstep
    # calculate how many days in input file
    num_out_steps = int(1 + num_timesteps / storage_interval)

    # set up storage arrays
    shape_xy = inp_ta.shape[1:]
    if len(shape_xy) == 2:
        st_swe = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
        st_melt = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
        st_acc = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
        st_alb = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
    elif len(shape_xy) == 1:
        st_swe = np.empty((num_out_steps, shape_xy[0])) * np.nan
        st_melt = np.empty((num_out_steps, shape_xy[0])) * np.nan
        st_acc = np.empty((num_out_steps, shape_xy[0])) * np.nan
        st_alb = np.empty((num_out_steps, shape_xy[0])) * np.nan
    # set up initial states of prognostic variables if not passed in
    if init_swe is None:
        init_swe = np.zeros(shape_xy)  # default to no snow
    swe = init_swe
    if init_d_snow is None:
        init_d_snow = np.ones(shape_xy) * 30  # default to a month since snowfall
    d_snow = init_d_snow
    # set up daily buckets for melt and accumulation
    bucket_melt = swe * 0
    bucket_acc = swe * 0
    swe_day_before = swe * 0

    # store initial swe value
    st_swe[0, :] = init_swe
    st_melt[0, :] = 0
    st_acc[0, :] = 0
    ii = 1

    # run through and update SWE for each timestep in input data
    for i in range(num_timesteps):
        # d_snow += dtstep / 86400.0 # now handled with daily threshold in at end of each day
        if 'inp_alb' in config:
            config['alb'] = config['inp_alb'][i, :]

        swe, d_snow, melt, acc = calc_dswe(swe, d_snow, inp_ta[i, :], inp_precip[i, :], inp_doy[i], dtstep, sw=inp_sw[i, :], which_melt=which_melt, **config)

        # print swe[0]
        bucket_melt = bucket_melt + melt
        bucket_acc = bucket_acc + acc

        if i != 0 and inp_hourdec[i] == 0 or inp_hourdec[i] == 24:  # output daily
            d_snow += 1
            swe_alb = swe - swe_day_before
            d_snow[(swe_alb > alb_swe_thres)] = 0
            swe_day_before = swe

        if (i + 1) % storage_interval == 0:  # if timestep divisible by the storage interval
            st_swe[ii, :] = swe
            st_melt[ii, :] = bucket_melt
            st_acc[ii, :] = bucket_acc
            st_alb[ii, :] = calc_albedo_snow(d_snow, swe, **config)
            ii = ii + 1  # move storage counter for next output timestep
            bucket_melt = bucket_melt * 0  # reset buckets
            bucket_acc = bucket_acc * 0

    return st_swe, st_melt, st_acc, st_alb


def snow_main(inp_file, init_swe=None, init_d_snow=None, which_melt='clark2009', alb_swe_thres=5.0, **config):
    """
    main snow model loop. handles timestepping and storage of output.
    assumes all the input data is on the same spatial and temporal grid.
    Runs the model for the length of the netCDF file
    Output is writen at the end of each day.
    :param inp_file: full path to netCDF input file
    :param init_swe: inital grid of snow water equivalent (SWE; mm w.e.), dimensions (spatial:)
    :param init_d_snow: inital grid of times since last snowfall dimensions (spatial:)
    :param which_melt: string specifying which melt model to be run. options include 'clark2009', 'dsc_snow'
    :param alb_swe_thres: threshold for daily snowfall resetting albedo (mm w.e). calculated from daily swe change so that accumulation must be > melt
    :return: st_swe - calculated SWE (mm w.e.) at the end of each day. (n = number of days + 1)
    """

    # load netCDF file and get the spatial dimensions out of it.
    inp_nc_file = nc.Dataset(inp_file)
    inp_dt = nc.num2date(inp_nc_file.variables['time'][:], inp_nc_file.variables['time'].units)
    num_timesteps = len(inp_dt)
    inp_hourdec = convert_dt_to_hourdec(inp_dt)
    inp_doy = convert_datetime_julian_day(inp_dt)
    # assume timestep is constant through input data
    dtstep = int((inp_dt[1] - inp_dt[0]).total_seconds())
    # calculate how many days in input file
    num_out_steps = int(1 + num_timesteps * dtstep / 86400.0)

    inp_shape = inp_nc_file.get_variables_by_attributes(standard_name='air_temperature')[0].shape
    shape_xy = inp_shape[1:]

    # set up storage arrays
    if len(shape_xy) == 2:
        st_swe = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
        st_melt = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
        st_acc = np.empty((num_out_steps, shape_xy[0], shape_xy[1])) * np.nan
    elif len(shape_xy) == 1:
        st_swe = np.empty((num_out_steps, shape_xy[0])) * np.nan
        st_melt = np.empty((num_out_steps, shape_xy[0])) * np.nan
        st_acc = np.empty((num_out_steps, shape_xy[0])) * np.nan

    # set up initial states of prognostic variables if not passed in
    if init_swe is None:
        init_swe = np.zeros(shape_xy)  # default to no snow
    swe = init_swe
    if init_d_snow is None:
        init_d_snow = np.ones(shape_xy) * 30  # default to a month since snowfall
    d_snow = init_d_snow
    # set up daily buckets for melt and accumulation
    bucket_melt = swe * 0
    bucket_acc = swe * 0

    # store initial swe value
    st_swe[0, :] = init_swe
    st_melt[0, :] = 0
    st_acc[0, :] = 0
    ii = 1

    # run through and update SWE for each timestep in input data
    for i in range(num_timesteps):
        # d_snow += dtstep / 86400.0

        inp_ta, inp_precip, inp_sw = read_met_input(inp_nc_file, i)
        swe, d_snow, melt, acc = calc_dswe(swe, d_snow, inp_ta, inp_precip, inp_doy[i], dtstep, sw=inp_sw, which_melt=which_melt, **config)

        # print swe[0]
        bucket_melt = bucket_melt + melt
        bucket_acc = bucket_acc + acc
        if i != 0 and inp_hourdec[i] == 0 or inp_hourdec[i] == 24:  # output daily don't store if
            st_swe[ii, :] = swe
            st_melt[ii, :] = bucket_melt
            st_acc[ii, :] = bucket_acc
            swe_alb = st_swe[ii, :] - st_swe[ii - 1, :]
            d_snow[(swe_alb > alb_swe_thres)] = 0
            ii = ii + 1  # move storage counter for next output timestep
            bucket_melt = bucket_melt * 0  # reset buckets
            bucket_acc = bucket_acc * 0

    return st_swe, st_melt, st_acc


def read_met_input(inp_nc_file, i):
    """
    read met input from net CDF file
    variable and units required are
    air_temperature' "K" 'precipitation_amount' "mm per timestep" 'surface_downwelling_shortwave_flux' "W / m2"
    :param inp_nc_file:
    :return: inp_ta, inp_p, inp_sw: arrays containing the input data for a single timestamp
    """
    inp_ta = inp_nc_file.get_variables_by_attributes(standard_name='air_temperature')[0][i, :]
    inp_p = inp_nc_file.get_variables_by_attributes(standard_name='precipitation_amount')[0][i, :]
    inp_sw = inp_nc_file.get_variables_by_attributes(standard_name='surface_downwelling_shortwave_flux_in_air')[0][i, :]

    return inp_ta, inp_p, inp_sw
