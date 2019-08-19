
# -*- coding: utf-8 -*-

"""
code to plot with the subplots the different models for each year and each station
"""




import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import netCDF4 as nc



data_folder = "C:/Users/conwayjp/NIWA/Ambre Bonnamour - SIN calibration timeseries"
swe_folder = "C:/Users/conwayjp/NIWA/Ambre Bonnamour - CSV SWE"

# npy files
# clark_file with different VCSN parameters
clark_file_obs = data_folder + "/{}/{}_npy files/Observed precipitation/{}_clark2009_{}.npy"
clark_file_precip_ta = data_folder + "/{}/{}_npy files/VCSN precip + ta/{}_clark2009_{}.npy"
clark_file_precip_rad = data_folder + "/{}/{}_npy files/VCSN precip + rad/{}_clark2009_{}.npy"
clark_file_precip_rad_ta = data_folder + "/{}/{}_npy files/VCSN precip + rad + ta/{}_clark2009_{}.npy"
clark_file_precip = data_folder + "/{}/{}_npy files/VCSN precip/{}_clark2009_{}.npy"
clark_file_ta = data_folder + "/{}/{}_npy files/VCSN ta/{}_clark2009_{}.npy"
clark_file_rad = data_folder + "/{}/{}_npy files/VCSN rad/{}_clark2009_{}.npy"
clark_file_rad_ta = data_folder + "/{}/{}_npy files/VCSN rad + ta/{}_clark2009_{}.npy"
# albedo_file with different VCSN parameters
albedo_file_obs = data_folder + "/{}/{}_npy files/Observed precipitation/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_ta = data_folder + "/{}/{}_npy files/VCSN precip + ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_rad = data_folder + "/{}/{}_npy files/VCSN precip + rad/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_rad_ta = data_folder + "/{}/{}_npy files/VCSN precip + rad + ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip = data_folder + "/{}/{}_npy files/VCSN precip/{}_dsc_snow-param albedo_{}.npy"
albedo_file_ta = data_folder + "/{}/{}_npy files/VCSN ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_rad = data_folder + "/{}/{}_npy files/VCSN rad/{}_dsc_snow-param albedo_{}.npy"
albedo_file_rad_ta = data_folder + "/{}/{}_npy files/VCSN rad + ta/{}_dsc_snow-param albedo_{}.npy"


def plot_year(j, Stname):


    year = 2008 + j
    # load npy data
    # observed data
    # clark
    # try:
    try:
        inp_clark_obs = np.load(clark_file_obs.format(Stname, Stname, Stname, year), allow_pickle=True)
        # inp_clark_obs1 = np.load(clark_file_obs.format(Stname,Stname,Stname,year1), allow_pickle=True)
        # inp_dt_obs = inp_clark_obs[:, 0]  # model stores initial state
        # inp_swe_obs = np.asarray(inp_clark_obs[:, 1], dtype=np.float)
        # plot_dt = inp_clark_obs[:, 0] # model stores initial state
    except:
        pass
    try:
        # albedo
        inp_alb_obs = np.load(albedo_file_obs.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_obs = inp_alb_obs[:, 0]  # model stores initial state
        inp_swe1_obs = np.asarray(inp_alb_obs[:, 1], dtype=np.float)
    except:
        pass
    # VCSN precipitation + temperature
    # clark
    # inp_clark_precip_ta = np.load(clark_file_precip_ta.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_precip_ta = inp_clark_precip_ta[:, 0]  # model stores initial state
    # inp_swe_VC_precip_ta = np.asarray(inp_clark_precip_ta[:, 1], dtype=np.float)
    # inp_swe_VN_precip_ta = np.asarray(inp_clark_precip_ta[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_precip_ta = np.load(albedo_file_precip_ta.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_precip_ta = inp_alb_precip_ta[:, 0]  # model stores initial state
        inp_swe1_VC_precip_ta = np.asarray(inp_alb_precip_ta[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_precip_ta = np.asarray(inp_alb_precip_ta[:, 2], dtype=np.float)
    except:
        print('No VN precip ta for {}'.format(year))

    # VCSN precipitation + radiation
    # clark
    # inp_clark_precip_rad = np.load(clark_file_precip_rad.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_precip_rad = inp_clark_precip_rad[:, 0]  # model stores initial state
    # inp_swe_VC_precip_rad = np.asarray(inp_clark_precip_rad[:, 1], dtype=np.float)
    # inp_swe_VN_precip_rad = np.asarray(inp_clark_precip_rad[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_precip_rad = np.load(albedo_file_precip_rad.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_precip_rad = inp_alb_precip_rad[:, 0]  # model stores initial state
        inp_swe1_VC_precip_rad = np.asarray(inp_alb_precip_rad[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_precip_rad = np.asarray(inp_alb_precip_rad[:, 2], dtype=np.float)
    except:
        print('No VN precip rad for {}'.format(year))

    # VCSN precipitation + radiation + temperature
    # clark
    # inp_clark_precip_rad_ta = np.load(clark_file_precip_rad_ta.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_precip_rad_ta = inp_clark_precip_rad_ta[:, 0]  # model stores initial state
    # inp_swe_VC_precip_rad_ta = np.asarray(inp_clark_precip_rad_ta[:, 1], dtype=np.float)
    # inp_swe_VN_precip_rad_ta = np.asarray(inp_clark_precip_rad_ta[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_precip_rad_ta = np.load(albedo_file_precip_rad_ta.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_precip_rad_ta = inp_alb_precip_rad_ta[:, 0]  # model stores initial state
        inp_swe1_VC_precip_rad_ta = np.asarray(inp_alb_precip_rad_ta[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_precip_rad_ta = np.asarray(inp_alb_precip_rad_ta[:, 2], dtype=np.float)
    except:
        print('No VN precip rad ta for {}'.format(year))

    # VCSN precipitation
    # clark
    # inp_clark_precip = np.load(clark_file_precip.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_precip = inp_clark_precip[:, 0]  # model stores initial state
    # inp_swe_precip = np.asarray(inp_clark_precip[:, 1], dtype=np.float)
    # albedo
    try:
        inp_alb_precip = np.load(albedo_file_precip.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_precip = inp_alb_precip[:, 0]  # model stores initial state
        inp_swe1_precip = np.asarray(inp_alb_precip[:, 1], dtype=np.float)
    except:
        pass
    # VCSN temperature
    # clark
    # inp_clark_ta = np.load(clark_file_ta.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_ta = inp_clark_ta[:, 0]  # model stores initial state
    # inp_swe_VC_ta = np.asarray(inp_clark_ta[:, 1], dtype=np.float)
    # inp_swe_VN_ta = np.asarray(inp_clark_ta[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_ta = np.load(albedo_file_ta.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_ta = inp_alb_ta[:, 0]  # model stores initial state
        inp_swe1_VC_ta = np.asarray(inp_alb_ta[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_ta = np.asarray(inp_alb_ta[:, 2], dtype=np.float)
    except:
        print('No VN ta for {}'.format(year))

    # VCSN radiation
    # clark
    # inp_clark_rad = np.load(clark_file_rad.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_rad = inp_clark_rad[:, 0]  # model stores initial state
    # inp_swe_VC_rad = np.asarray(inp_clark_rad[:, 1], dtype=np.float)
    # inp_swe_VN_rad = np.asarray(inp_clark_rad[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_rad = np.load(albedo_file_rad.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_rad = inp_alb_rad[:, 0]  # model stores initial state
        inp_swe1_VC_rad = np.asarray(inp_alb_rad[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_rad = np.asarray(inp_alb_rad[:, 2], dtype=np.float)
    except:
        print('No VN precip rad for {}'.format(year))

    # VCSN radiation + temperature
    # clark
    # inp_clark_rad_ta = np.load(clark_file_rad_ta.format(Stname,Stname,Stname,year), allow_pickle=True)
    # inp_dt_rad_ta = inp_clark_rad_ta[:, 0]  # model stores initial state
    # inp_swe_VC_rad_ta = np.asarray(inp_clark_rad_ta[:, 1], dtype=np.float)
    # inp_swe_VN_rad_ta = np.asarray(inp_clark_rad_ta[:, 2], dtype=np.float)
    # albedo
    try:
        inp_alb_rad_ta = np.load(albedo_file_rad_ta.format(Stname, Stname, Stname, year), allow_pickle=True)
        inp_dt1_rad_ta = inp_alb_rad_ta[:, 0]  # model stores initial state
        inp_swe1_VC_rad_ta = np.asarray(inp_alb_rad_ta[:, 1], dtype=np.float)
    except:
        pass
    try:
        inp_swe1_VN_rad_ta = np.asarray(inp_alb_rad_ta[:, 2], dtype=np.float)
    except:
        print('No VN rad ta for {}'.format(year))

    # load csv file
    inp_datobs = np.genfromtxt(csv_file, delimiter=',', usecols=(1), skip_header=4) * 1000
    inp_timeobs = np.genfromtxt(csv_file, usecols=(0), dtype=(str), delimiter=',', skip_header=4)
    inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])
    ind = np.logical_and(inp_dtobs >= dt.datetime(year, 4, 1), inp_dtobs <= dt.datetime(year + 1, 4, 1))
    datobs_year = inp_datobs[ind]
    dtobs_year = inp_dtobs[ind]

    # plot data in new frame
    plt.sca(axs[j])  # set the plot to the correct subplot
    plt.title('Year : April {} - March {}'.format(year, year+1))
    # plot

    try:
        plt.plot(inp_dt1_obs, inp_swe1_obs, linewidth=2, color='r', label='Obs precip')
    except:
        plt.plot(dt.datetime(year, 4, 1),0, linewidth=2, color='r', label='Obs precip')
        print('No Obs precip for {}'.format(year))

    try:
        plt.plot(inp_dt1_precip, inp_swe1_precip, linewidth=2, color='firebrick', label='VCSN precip')
    except:
        plt.plot(dt.datetime(year, 4, 1),0, linewidth=2, color='firebrick', label='VCSN precip')
        print('No VC precip for {}'.format(year))

    # try:
    #     plt.plot(inp_dt1_precip_ta, inp_swe1_VC_precip_ta, linewidth=2, color='seagreen', label='VC precip + ta')
    # except:
    #     print('No VC precip + ta for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_rad_ta, inp_swe1_VC_rad_ta, linewidth=2, color='darkslategrey', label='VC rad + ta')
    # except:
    #     print('No VC rad + ta for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_precip_rad, inp_swe1_VC_precip_rad, linewidth=2, color='teal', label='VC precip + rad')
    # except:
    #     print('No VC precip + rad for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_ta, inp_swe1_VC_ta, linewidth=2, color='limegreen', label='VC ta')
    # except:
    #     print('No VC ta for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_rad, inp_swe1_VC_rad, linewidth=2, color='forestgreen', label='VC rad')
    # except:
    #     print('No VC rad for {}'.format(year))

    try:
        plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VC_precip_rad_ta, linewidth=2, color='lightseagreen', label='VC precip + rad + ta')
    except:
        plt.plot(dt.datetime(year, 4, 1),0, linewidth=2, color='lightseagreen', label='VC precip + rad + ta') # plot point for legend
        print('No VC precip + rad + ta for {}'.format(year))

    # try:
    #     plt.plot(inp_dt1_precip_ta, inp_swe1_VN_precip_ta, '--', linewidth=2, color='violet', label='VN precip + ta')
    # except:
    #     print('No VN precip + ta for {}'.format(year))
    #
    # try :
    #     plt.plot(inp_dt1_rad_ta, inp_swe1_VN_rad_ta,'--',linewidth = 2,color = 'hotpink', label = 'VN rad + ta')
    # except :
    #     print('No VN rad + ta for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_precip_rad, inp_swe1_VN_precip_rad, '--', linewidth=2, color='darkmagenta', label='VN precip + rad')
    # except:
    #     print('No VN precip + rad for {}'.format(year))
    #
    # try :
    #     plt.plot(inp_dt1_ta, inp_swe1_VN_ta,'--',linewidth = 2,color = 'lightcoral', label = 'VN ta')
    # except :
    #     print('No VN ta for {}'.format(year))
    #
    # try:
    #     plt.plot(inp_dt1_rad, inp_swe1_VN_rad,'--',linewidth = 2,color = 'mediumpurple', label = 'VN rad')
    # except:
    #     print('No VN rad for {}'.format(year))

    try :
        plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VN_precip_rad_ta, '--',linewidth = 2,color = 'magenta', label = 'VN precip + rad + ta')
    except :
        plt.plot(dt.datetime(year, 4, 1), 0, '--', linewidth=2, color='magenta', label='VN precip + rad + ta')
        print('No VN precip + rad + ta for {}'.format(year))

    try:
        plt.plot(dtobs_year, datobs_year, 'o', markersize=4, color='b', label='Observed')
    except:
        print('No obs for {}'.format(year))

    plt.gcf().autofmt_xdate()
    months = mdates.MonthLocator()  # every month
    monthsFmt = mdates.DateFormatter('%b')

    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_ylabel(r"SWE mm w.e.")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    ax.set_xlabel('Month')
    ax.set_ylim([0, ymax])

    # castle Mount 1000
    # larkins 600
    # mahange 800
    # mueller 2000
    # Murchison 300
    # Philistine 800

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    if j ==0:
        plt.legend()

    # plt.savefig(data_folder + "/Mueller/Mueller_Plots/{}_{}".format(Stname, year))
    # plt.plot()  # add in your existing plotting including any other modifications to each axis here e.g. labels, tick marks
    plt.tight_layout()  # makes the axis fill the available space
    # except:
    #     print('something missing for year {}'.format(year))
    # plt.show()
    # plt.close()

# Stname = ['Philistine']

sites = ['Castle Mount','Larkins', 'Mahanga', 'Mueller', 'Murchison','Philistine']
maxswe = [3000,1000,1000,2000,600,1500]


for Stname,ymax in zip(sites,maxswe):
# csv file
    csv_file = swe_folder + "/{}_SWE.csv".format(Stname)

    f2, axs2 = plt.subplots(4, 3, figsize=(12, 12))  # sets number of rows and columns of subplot as well as figure size in inches
    axs = axs2.ravel()
    for j in range(12):  # run through each subplot (here there are 9 because of 3 rows and columns)
    #
    # for i in range (0,7) :
        plot_year(j,Stname)

    f2.savefig(data_folder + "/small_{}_plots_allyears.png".format(Stname))  # save the figure once it’s done.