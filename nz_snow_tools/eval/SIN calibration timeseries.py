"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import netCDF4 as nc

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']=3600
config['tacc'] = 274.16
config['tmelt'] = 274.16

# clark2009 melt parameters
config['mf_mean'] = 5.0
config['mf_amp'] = 5.0
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5.0
config['mf_ros'] = 2.5 # default 2.5
config['mf_doy_max_ddf'] = 356 # default 356

# dsc_snow melt parameters
config['tf'] = 0.05*24  # hamish 0.13. ruschle 0.04, pelliciotti 0.05
config['rf'] = 0.0108*24 # hamish 0.0075,ruschle 0.009, pelliciotti 0.0094

# albedo parameters
config['dc'] = 11.0
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.90
config['a_firn'] = 0.62
config['alb_swe_thres'] = 10
config['ros'] = True
config['ta_m_tt'] = False

# npy files for each year
# MUELLER station data for each year [2011-2018]
# Y_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_{}.npy"
# MAHANGA station data for each year [2009-2018]
# Y_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_{}.npy"
# LARKINS station data for each year [2014-2018]
# Y_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_{}.npy"
# CASTLE MOUNT station data for each year [2012-2016]
# Y_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_{}.npy"
# MURCHISON station data for each year [2009-2018]
Y_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_{}.npy"
# PHILISTINE station data for each year [2011-2018]
# Y_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_{}.npy"

# VCSN files
# CASTLE MOUNT
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_CastleMo_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_CastleMo_strahler3-VN.nc",'r')
# LARKINS
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Larkins_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Larkins_strahler3-VN.nc",'r')
# MAHANGA
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Mahanga_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Mahanga_strahler3-VN.nc", 'r')
# MUELLER
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Mueller_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Mueller_strahler3-VN.nc",'r')
# PHILISTINE
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Philisti_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Philisti_strahler3-VN.nc",'r')
# MURCHISON
nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Murchiso_strahler3-VC.nc",'r')
nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Murchiso_strahler3-VN.nc", 'r')

Stname = ['Murchison']
for i in range (0,10) :
    year = 2009 + i


    # load npy data
    inp_dat = np.load(Y_file.format(year),allow_pickle=True)
    inp_doy = np.asarray(convert_datetime_julian_day(inp_dat[:, 0]))
    inp_hourdec = convert_dt_to_hourdec(inp_dat[:, 0])
    plot_dt = inp_dat[:, 0] # model stores initial state
    inp_precip_obs = np.asarray(inp_dat[:, 4], dtype=np.float)

    # load VCSN files
    # snow storage VC&VN files
    swe_VC = nc_file_VC.variables['snwstor'][:,0,0,0]
    swe_VN = nc_file_VN.variables['snwstor'][:,0,0,0]

    # time VC&VN files
    nc_datetimes_VC = nc.num2date(nc_file_VC.variables['time'][:], nc_file_VC.variables['time'].units)
    nc_datetimes_VN = nc.num2date(nc_file_VN.variables['time'][:], nc_file_VN.variables['time'].units)

    # accumulate precipitation VCSN files
    precip_VC = nc_file_VC.variables['aprecip'][:,0,0,0]

    #average temperature VC&VN files
    avgtemp_VC = nc_file_VC.variables['avgtemp'][:,0,0]
    avgtemp_VN = nc_file_VN.variables['avgtemp'][:,0,0]

    # empirical estimates of net radiation
    net_rad_VC = nc_file_VC.variables['net_rad'][:,0,0]
    net_rad_VN = nc_file_VN.variables['net_rad'][:,0,0]

    # keep the same time than the other files
    ind_VC = np.logical_and(nc_datetimes_VC >= plot_dt[0],nc_datetimes_VC <= plot_dt[-1])
    ind_VN = np.logical_and(nc_datetimes_VN >= plot_dt[0],nc_datetimes_VN <= plot_dt[-1])
    swe_VC_year = swe_VC[ind_VC]
    swe_VN_year = swe_VN[ind_VN]
    year_VC = nc_datetimes_VC[ind_VC]
    year_VN = nc_datetimes_VN[ind_VN]
    precip_VC_year = precip_VC[ind_VC]*1000 # precipitation for one year in mm
    cum_precip_VC = np.cumsum(precip_VC_year) # cumulated precipitation for one year
    avgtemp_VC_year = avgtemp_VC[ind_VC]  # average temperature for one year
    avgtemp_VN_year = avgtemp_VN[ind_VN]
    rad_VC_year =net_rad_VC[ind_VC]
    rad_VN_year = net_rad_VN[ind_VN]

    # make grids of input data
    grid_size = 1
    grid_id = np.arange(grid_size)
    # inp_ta = np.asarray(inp_dat[:,2],dtype=np.float)[:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre observed air temp in C
    inp_ta = np.asarray(avgtemp_VC_year, dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  # VC air temp in K
    inp_ta_VN = np.asarray(avgtemp_VN_year, dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  #  VN air temp in K
    inp_precip = np.asarray(inp_dat[:,4],dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  # observed precip in mm
    # inp_precip = np.asarray(precip_VC_year, dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  # VCSN precip in mm
    inp_sw_obs = np.asarray(inp_dat[:,3],dtype=np.float)[:, np.newaxis] * np.ones(grid_size) # observed
    inp_sw = np.asarray(rad_VC_year, dtype=np.float)[:, np.newaxis] * np.ones(grid_size) # VC radiation in W/m2
    inp_sw_VN = np.asarray(rad_VN_year, dtype=np.float)[:, np.newaxis] * np.ones(grid_size) # VN radiation in W/m2

    init_swe = np.ones(inp_ta.shape[1:],dtype=np.float) * 0  # give initial value of swe as starts in spring
    init_d_snow = np.ones(inp_ta.shape[1:],dtype=np.float) * 30  # give initial value of days since snow

    # plt.title('Mueller Radiations in {}'.format(year))
    # plt.plot(plot_dt,inp_sw_obs/2, label = 'Observed radiation')
    # plt.plot(year_VC, rad_VC_year, label = 'VC rad')
    # plt.plot(year_VN, rad_VN_year, label = 'VN rad')
    # plt.gcf().autofmt_xdate()
    # months = mdates.MonthLocator()  # every month
    # monthsFmt = mdates.DateFormatter('%b')
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # plt.ylabel('radiations in W/m2')
    # plt.legend()
    # plt.show()
    # plt.close

    # call main function once hourly/sub-hourly temp and precip data available.
    try :
        st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600,init_swe=init_swe,
                                                    init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
        st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                    init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
    except :
        print('No VC data', year)
    try :
        st_swea, st_melta, st_acca, st_alba = snow_main_simple(inp_ta_VN, inp_precip, inp_doy, inp_hourdec, dtstep=3600,init_swe=init_swe,
                                                    init_d_snow=init_d_snow, inp_sw=inp_sw_VN,which_melt='clark2009', **config)
        st_swe1a, st_melt1a, st_acc1a, st_alb1a = snow_main_simple(inp_ta_VN, inp_precip, inp_doy, inp_hourdec, dtstep=3600,init_swe=init_swe,
                                                    init_d_snow=init_d_snow, inp_sw=inp_sw_VN,which_melt='dsc_snow', **config)
    except :
        print('No VN data', year)

    # MUELLER SWE csv file
    # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv"
    # MAHANGA SWE csv file
    # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mahanga SWE.csv"
    # LARKINS SWE csv file
    # csv_file ="C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Larkins SWE.csv"
    # CASTLE MOUNT SWE csv file
    # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Castle Mount SWE.csv"
    # MURCHISON SWE csv file
    csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Murchison SWE.csv"
    # PHILISTINE SWE csv file
    # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Philistine SWE.csv"

    # load observed data
    inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),
                            skip_header=4)*1000
    inp_timeobs = np.genfromtxt(csv_file, usecols=(0),
                             dtype=(str), delimiter=',', skip_header=4)
    inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])

    ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])

    plt.plot(inp_dtobs[ind],inp_datobs[ind],"o", label = "Observed SWE")

    plt.plot(plot_dt,st_swe[1:, 0], linewidth = 2, label='clark2009 VC rad + ta')
    plt.plot(plot_dt,st_swe1[1:, 0],linewidth = 2, label='dsc_snow-param albedo VC rad + ta')

    try :
        plt.plot(plot_dt, st_swea[1:, 0],'--', linewidth = 2, label='clark2009 VN rad + ta', color = 'gold')
        plt.plot(plot_dt, st_swe1a[1:, 0],'--', linewidth = 2, label='dsc_snow-param albedo VN rad + ta', color ='limegreen')
    except:
        print('No data')

    plt.legend()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax2.plot(plot_dt,np.cumsum(inp_precip), label = "Precipitation", color = 'navy')

    plt.gcf().autofmt_xdate()
    months = mdates.MonthLocator()  # every month

    monthsFmt = mdates.DateFormatter('%b')

    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax1.set_ylabel(r"SWE mm w.e.")
    ax2.set_ylabel(r"Precipitation mm")
    plt.xlabel('Month')

    plt.legend()
    plt.title('Cumulative mass balance TF:{:2.4f}, RF: {:2.4f}, Tmelt:{:3.2f}, Year : {}, Station : {}'.format(config['tf'],config['rf'],config['tmelt'], year, Stname[0]))

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.savefig("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_Plots/{}_VCSN rad + ta/{}_{} daily TF{:2.4f}RF{:2.4f}Tmelt{:3.2f}_ros.png".format( Stname[0], Stname[0],Stname[0],Stname[0],year, config['tf'],config['rf'],config['tmelt']))
    plt.show()
    plt.close()

    # plt.get_backend() -> to determine the backend used
    # plt.title('Models VC and VN for the year {}, Station : {},TF:{:2.4f}, RF: {:2.4f}, Tmelt:{:3.2f}'.format(year, Stname[0],config['tf'],config['rf'],config['tmelt']))
    # plt.plot(inp_dtobs[ind],inp_datobs[ind],"o", label = "Observed SWE")
    # plt.plot(year_VC, swe_VC_year, label = 'VC', color = "magenta")
    # plt.plot(year_VN, swe_VN_year, label = 'VN', color = "salmon")
    # plt.legend()
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # ax2.plot(plot_dt,np.cumsum(inp_precip_obs), label = "Precipitation")
    # ax2.plot(year_VC,cum_precip_VC, label = "Precipitation VCSN", color = 'purple')
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    #
    # ax1.set_ylabel(r"SWE mm w.e.")
    # ax2.set_ylabel(r"Precipitation mm")
    #
    # plt.legend()
    # manager1 = plt.get_current_fig_manager()
    # manager1.window.showMaximized()
    #
    # plt.savefig("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_Plots/{}_VCSN/{} VCSN_{}.png".format(Stname[0], Stname[0], Stname[0],Stname[0], year))
    # plt.show()
    # plt.close()

    # plot summarising every other plots
    plt.plot(inp_dtobs[ind],inp_datobs[ind],"o", label = "Observed SWE")
    plt.plot(plot_dt,st_swe[1:, 0],linewidth = 2,label='clark2009 VC rad + ta')
    plt.plot(plot_dt,st_swe1[1:, 0],linewidth = 2,label='dsc_snow-param albedo VC rad + ta')
    try :
        plt.plot(plot_dt, st_swea[1:, 0],'--', linewidth = 2, label='clark2009 VN rad + ta', color='gold')
        plt.plot(plot_dt, st_swe1a[1:, 0],'--', linewidth = 2,label='dsc_snow-param albedo VN rad + ta', color='limegreen')
    except :
        print('No data')
    plt.plot(year_VC, swe_VC_year,linewidth = 3,  label = 'VC', color = "magenta")
    plt.plot(year_VN, swe_VN_year, linewidth = 3, label = 'VN', color = "crimson")
    plt.legend()
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax2.plot(plot_dt,np.cumsum(inp_precip_obs), label = "Precipitation", color = 'navy')
    ax2.plot(year_VC,cum_precip_VC, label = "Precipitation VCSN", color = 'darkorchid')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=7)
    ax1.set_ylabel(r"SWE mm w.e.")
    ax2.set_ylabel(r"Precipitation mm")
    plt.legend()

    plt.gcf().autofmt_xdate()
    months = mdates.MonthLocator()  # every month
    monthsFmt = mdates.DateFormatter('%b')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_xlabel(r"Month")

    plt.title('Cumulative mass balance TF:{:2.4f}, RF: {:2.4f}, Tmelt:{:3.2f}, Year : {}, Station : {}'.format(config['tf'],config['rf'],config['tmelt'], year, Stname[0]))
    plt.legend()

    manager2 = plt.get_current_fig_manager()
    manager2.window.showMaximized()
    plt.savefig("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_Plots/{}_VCSN rad + ta/{}_{} daily TF{:2.4f}RF{:2.4f}Tmelt{:3.2f}_ros_VCSN.png".format(Stname[0], Stname[0], Stname[0], Stname[0], year, config['tf'], config['rf'], config['tmelt']))
    plt.show()
    plt.close()

# saved files
# LARKINS
#     save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/VCSN rad + ta/Larkins_clark2009_{}"
#     save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/VCSN rad + ta/Larkins_dsc_snow-param albedo_{}"
# save_file_VC ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_VC_{}"
# save_file_VN ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_VN_{}"
# PHILISTINE
#     save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/VCSN rad + ta/Philistine_clark2009_{}"
#     save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/VCSN rad + ta/Philistine_dsc_snow-param albedo_{}"
# save_file_VC = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_VC_{}"
# save_file_VN = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_VN_{}"
# CASTLE MOUNT
#     save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/Castle Mount_npy files/VCSN rad + ta/Castle Mount_clark2009_{}"
#     save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/Castle Mount_npy files/VCSN rad + ta/Castle Mount_dsc_snow-param albedo_{}"
# save_file_VC ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_VC_{}"
# save_file_VN ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_VN_{}"
# MUELLER
#     save_file_clark2009 = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/VCSN rad + ta/Mueller_clark2009_{}"
#     save_file_albedo = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/VCSN rad + ta/Mueller_dsc_snow-param albedo_{}"
# save_file_VC = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_VC_{}"
# save_file_VN = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_VN_{}"
# MURCHISON
    save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/VCSN rad + ta/Murchison_clark2009_{}"
    save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/VCSN rad + ta/Murchison_dsc_snow-param albedo_{}"
    # save_file_VC ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_VC_{}"
    # save_file_VN ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_VN_{}"
# MAHANGA
#     save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/VCSN rad + ta/Mahanga_clark2009_{}"
#     save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/VCSN rad + ta/Mahanga_dsc_snow-param albedo_{}"
#     save_file_VC ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/VCSN/Mahanga_VC_{}"
#     save_file_VN ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/VCSN/Mahanga_VN_{}"

# save the data : Clark 2009
    if year < 2017 :
        output1 = np.transpose(np.vstack((plot_dt,st_swe[1:, 0],st_swea[1:, 0] )))
        np.save(save_file_clark2009.format(year),output1)

    # save the data : Albedo
        output2 = np.transpose(np.vstack((plot_dt,st_swe1[1:, 0],st_swe1a[1:, 0])))
        np.save(save_file_albedo.format(year),output2)
    else :
        output1 = np.transpose(np.vstack((plot_dt, st_swe[1:, 0])))
        np.save(save_file_clark2009.format(year), output1)
        output2 = np.transpose(np.vstack((plot_dt, st_swe1[1:, 0])))
        np.save(save_file_albedo.format(year), output2)
#
#     # save the data : VC
#     output3 = np.transpose(np.vstack((year_VC,swe_VC_year)))
#     np.save(save_file_VC.format(year),output3)
#
#     # save the data : VN
#     if year <= 2017 :
#         output4 = np.transpose(np.vstack((year_VN,swe_VN_year)))
#         np.save(save_file_VN.format(year),output4)
#     else :
#         print('no data')