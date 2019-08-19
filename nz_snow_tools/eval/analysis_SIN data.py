import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
from nz_snow_tools.util.utils import fill_timeseries
from nz_snow_tools.eval.utils_Ambre import maxmin
from nz_snow_tools.eval.utils_Ambre import amount_snowmelt
import netCDF4 as nc
from nz_snow_tools.eval.utils_Ambre import amount_precipitation

#CASTLE MOUNT [2012-2016]
# LARKINS [2014-2018]
# MAHANGA [2009-2018]
# MUELLER [2011-2018]
# MURCHISON [2009-2018]
# PHILISTINE [2011-2018]
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

C_precip_obs_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/Observed precipitation/{}_clark2009_{}.npy"
C_precip_VCSN_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip/{}_clark2009_{}.npy"
A_precip_obs_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/Observed precipitation/{}_dsc_snow-param albedo_{}.npy"
A_precip_VCSN_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip/{}_dsc_snow-param albedo_{}.npy"
VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN/{}_VC_{}.npy"
VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN/{}_VN_{}.npy"
precip_obs_file = "C:/Users/Bonnamourar/Desktop/SIN/{}/{}_2007-2019/{}_2007-2019_Rain.txt"

import csv
Stname = ['Murchison']
with open("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Analysis/{}_Max&Min.csv".format(Stname[0]),mode='w', newline='') as maxmin_file:
    maxmin_writer = csv.writer(maxmin_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    fieldnames = ['Data', 'Year', 'SWE max', 'Date max', 'SWE min', 'Date min','Amount of obs precipitation','Amount of VCSN precipitation', 'Amount of snow melt']
    maxmin_writer.writerow(['Data', 'Year', 'SWE max', 'Date max', 'SWE min', 'Date min','Amount of obs precipitation','Amount of VCSN precipitation', 'Amount of snow melt'])
    maxmin_writer = csv.DictWriter(maxmin_file, fieldnames=fieldnames)
    for i in range (0,10) :
        Year = 2009 + i

        #########################################
        # load clark2009 model
        inp_clark_obs = np.load(C_precip_obs_file.format(Stname[0],Stname[0],Stname[0],Year),allow_pickle=True) # Observed precipitation
        inp_time1 = inp_clark_obs[:,0]
        inp_swe1 = np.asarray(inp_clark_obs[:,1],dtype=np.float)
        plot_dt = inp_clark_obs[:, 0] # model stores initial state
        try :
            inp_clark_VCSN = np.load(C_precip_VCSN_file.format(Stname[0],Stname[0],Stname[0],Year), allow_pickle=True)  # VCSN precipitation
            inp_time1a = inp_clark_VCSN[:, 0]
            inp_swe1a = np.asarray(inp_clark_VCSN[:, 1], dtype=np.float)
        except :
            print('No data')

        #########################################
        # load dsc_param_albedo model
        inp_albedo_obs = np.load(A_precip_obs_file.format(Stname[0],Stname[0],Stname[0],Year),allow_pickle=True) # Observed precipitation
        inp_time2 = inp_albedo_obs[:,0]
        inp_swe2 = np.asarray(inp_albedo_obs[:,1],dtype=np.float)
        try :
            inp_albedo_VCSN = np.load(A_precip_VCSN_file.format(Stname[0],Stname[0],Stname[0],Year), allow_pickle=True)  # VCSN  precipitation
            inp_time2a = inp_albedo_VCSN[:, 0]
            inp_swe2a = np.asarray(inp_albedo_VCSN[:, 1], dtype=np.float)
        except :
            print('No data')

        #########################################
        # load VC model
        inp_VC = np.load(VC_file.format(Stname[0],Stname[0],Stname[0],Year),allow_pickle=True)
        inp_time3 = inp_VC[:,0]
        inp_swe3 = np.asarray(inp_VC[:,1],dtype=np.float)

        # load VN model
        if Year <= 2017 :
            inp_VN = np.load(VN_file.format(Stname[0],Stname[0],Stname[0],Year),allow_pickle=True)
            inp_time4 = inp_VN[:,0]
            inp_swe4 = np.asarray(inp_VN[:,1],dtype=np.float)
        else :
            print('No data')

        #########################################
        # MUELLER SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller_SWE.csv"
        # MAHANGA SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mahanga_SWE.csv"
        # LARKINS SWE csv file
        # csv_file ="C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Larkins_SWE.csv"
        # CASTLE MOUNT SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Castle Mount_SWE.csv"
        # MURCHISON SWE csv file
        csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Murchison_SWE.csv"
        # PHILISTINE SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Philistine_SWE.csv"

        # load observed data
        inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),
                                skip_header=4)*1000
        inp_timeobs = np.genfromtxt(csv_file, usecols=(0),
                                 dtype=(str), delimiter=',', skip_header=4)
        inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])
        ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])
        try :
            inp_dtobs_clean, inp_datobs_clean = fill_timeseries(inp_dtobs[ind], inp_datobs[ind], 3600)
        except :
            inp_dtobs_clean = inp_dtobs[ind]
            inp_datobs_clean = inp_datobs[ind]

        #########################################
        #  Observed precipitation data
        inp_datobs_precip = np.genfromtxt(precip_obs_file.format(Stname[0], Stname[0], Stname[0]), delimiter=',',
                                          skip_header=9, skip_footer=8)
        inp_timobs_precip = np.genfromtxt(precip_obs_file.format(Stname[0], Stname[0], Stname[0]), usecols=(1),
                                          dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
        inp_dtobs_precip = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_timobs_precip])
        precipitation = inp_datobs_precip[:, 2]
        ind_obs_precip = np.logical_and(inp_dtobs_precip >= plot_dt[0], inp_dtobs_precip <= plot_dt[-1])
        obs_precip = precipitation[ind_obs_precip]
        time_obs_precip = inp_dtobs_precip[ind_obs_precip]
        obs_sumprecip = np.cumsum(obs_precip)

        # VCSN precipitation data
        nc_datetimes_VC = nc.num2date(nc_file_VC.variables['time'][:], nc_file_VC.variables['time'].units)
        nc_datetimes_VN = nc.num2date(nc_file_VN.variables['time'][:], nc_file_VN.variables['time'].units)
        precip_VC = nc_file_VC.variables['aprecip'][:, 0, 0, 0]
        ind_VC = np.logical_and(nc_datetimes_VC >= plot_dt[0], nc_datetimes_VC <= plot_dt[-1])
        ind_VN = np.logical_and(nc_datetimes_VN >= plot_dt[0], nc_datetimes_VN <= plot_dt[-1])
        year_VC = nc_datetimes_VC[ind_VC]
        year_VN = nc_datetimes_VN[ind_VN]
        precip_VC_year = precip_VC[ind_VC] * 1000  # precipitation for one year in mm
        VCSN_sumprecip = np.cumsum(precip_VC_year)  # cumulated precipitation for one year

        ##################################################################################
        #########################################
        # Max and Min values, observed data
        try :
            maximum_observed, minimum_observed, date_max_observed, date_min_observed = maxmin(inp_dtobs_clean, inp_datobs_clean)
            try :
                snw_melt_obs = amount_snowmelt(maximum_observed, inp_dtobs_clean, inp_datobs_clean)
            except :
                snw_melt_obs = 'No data'
            try :
                amnt_precip_obs = amount_precipitation(maximum_observed, inp_datobs_clean, obs_sumprecip)
            except :
                amnt_precip_obs = 'No data'
            try:
                amnt_precip_VCSN = amount_precipitation(maximum_observed, inp_datobs_clean, VCSN_sumprecip)
            except :
                amnt_precip_VCSN = 'No data'
        except :
            maximum_observed = 'No data'
            minimum_observed = 'No data'
            date_max_observed = 'No data'
            date_min_observed = 'No data'
            snw_melt_obs = 'No data'
            amnt_precip_obs = 'No data'
            amnt_precip_VCSN = 'No data'
            print('Observed ERROR {}'.format(Year))

        # csv file writing
        try :
            maxmin_writer.writerow({'Data' : 'Observed', 'Year': Year, 'SWE max' : maximum_observed, 'Date max' : date_max_observed, 'SWE min' : minimum_observed, 'Date min' : date_min_observed
                                        ,'Amount of obs precipitation':amnt_precip_obs,'Amount of VCSN precipitation' : amnt_precip_VCSN, 'Amount of snow melt':snw_melt_obs})
        except :
            print('ERROR {} observed'.format(Year))

        #########################################
        # Max and Min values, clark2009
        try :
            maximum_clark_precip_obs, minimum_clark_precip_obs, date_max_clark_precip_obs, date_min_clark_precip_obs = maxmin(inp_time1, inp_swe1)
            try :
                snw_melt_clark_precip_obs = amount_snowmelt(maximum_clark_precip_obs, inp_time1, inp_swe1)
            except :
                snw_melt_clark_precip_obs = 'No data'
            try :
                amnt_precip_obs_clark1 = amount_precipitation(maximum_clark_precip_obs, inp_swe1, obs_sumprecip)
            except :
                amnt_precip_obs_clark1 = 'No data'
            try :
                amnt_precip_VCSN_clark1 = amount_precipitation(maximum_clark_precip_obs, inp_swe1, VCSN_sumprecip)
            except :
                amnt_precip_VCSN_clark1 = 'No data'
            print('Max clark precip obs :', maximum_clark_precip_obs,'Date max :', date_max_clark_precip_obs, 'Snow melt clark precip obs :', snw_melt_clark_precip_obs, 'Obs precip : ',amnt_precip_obs_clark1, 'VCSN precip :', amnt_precip_VCSN_clark1)
        except:
            maximum_clark_precip_obs = 'No data'
            minimum_clark_precip_obs = 'No data'
            date_max_clark_precip_obs = 'No data'
            date_min_clark_precip_obs = 'No data'
            snw_melt_clark_precip_obs = 'No data'
            amnt_precip_obs_clark1 = 'No data'
            amnt_precip_VCSN_clark1 = 'No data'
            print('Clark precip obs ERROR {}'.format(Year))

        try:
            maximum_clark_precip_VCSN, minimum_clark_precip_VCSN, date_max_clark_precip_VCSN, date_min_clark_precip_VCSN = maxmin(inp_time1a, inp_swe1a)
            try :
                snw_melt_clark_precip_VCSN = amount_snowmelt(maximum_clark_precip_VCSN, inp_time1a, inp_swe1a)
            except :
                snw_melt_clark_precip_VCSN = 'No data'
            try :
                amnt_precip_obs_clark1a = amount_precipitation(maximum_clark_precip_VCSN, inp_swe1a, obs_sumprecip)
            except :
                amnt_precip_obs_clark1a = 'No data'
            try :
                amnt_precip_VCSN_clark1a = amount_precipitation(maximum_clark_precip_VCSN, inp_swe1a, VCSN_sumprecip)
            except :
                amnt_precip_VCSN_clark1a = 'No data'
            print('Max clark precip VCSN :', maximum_clark_precip_VCSN, 'Date max :', date_max_clark_precip_VCSN,'Snow melt clark precip VCSN :', snw_melt_clark_precip_VCSN, 'Obs precip : ',amnt_precip_obs_clark1a, 'VCSN precip :', amnt_precip_VCSN_clark1a)
        except:
            maximum_clark_precip_VCSN = 'No data'
            minimum_clark_precip_VCSN = 'No data'
            date_max_clark_precip_VCSN = 'No data'
            date_min_clark_precip_VCSN = 'No data'
            snw_melt_clark_precip_VCSN = 'No data'
            amnt_precip_obs_clark1a = 'No data'
            amnt_precip_VCSN_clark1a = 'No data'
            print('Clark precip VCSN ERROR {}'.format(Year))

        # csv file writing
        try :
            maxmin_writer.writerow({'Data': 'Observed precipitation Clark2009', 'Year': Year, 'SWE max': maximum_clark_precip_obs, 'Date max': date_max_clark_precip_obs, 'SWE min': minimum_clark_precip_obs,'Date min': date_min_clark_precip_obs
                                       ,'Amount of obs precipitation':amnt_precip_obs_clark1,'Amount of VCSN precipitation' : amnt_precip_VCSN_clark1, 'Amount of snow melt':snw_melt_clark_precip_obs})
        except :
            print('ERROR {} observed precipitation clark'.format(Year))
        try:
            maxmin_writer.writerow({'Data': 'VCSN precipitation Clark2009', 'Year': Year, 'SWE max': maximum_clark_precip_VCSN,'Date max': date_max_clark_precip_VCSN, 'SWE min': minimum_clark_precip_VCSN,'Date min': date_min_clark_precip_VCSN
                                       ,'Amount of obs precipitation':amnt_precip_obs_clark1a,'Amount of VCSN precipitation' : amnt_precip_VCSN_clark1a, 'Amount of snow melt':snw_melt_clark_precip_VCSN})
        except:
            print('ERROR {} VCSN precipitation clark'.format(Year))

        #########################################
        # Max and Min values, albedo
        try :
            maximum_albedo_precip_obs, minimum_albedo_precip_obs, date_max_albedo_precip_obs, date_min_albedo_precip_obs = maxmin(inp_time2, inp_swe2)
            try :
                snw_melt_albedo_precip_obs = amount_snowmelt(maximum_albedo_precip_obs, inp_time2, inp_swe2)
            except :
                snw_melt_albedo_precip_obs = 'No data'
            try :
                amnt_precip_obs_albedo2 = amount_precipitation(maximum_albedo_precip_obs, inp_swe2, obs_sumprecip)
            except :
                amnt_precip_obs_albedo2 = 'No data'
            try :
                amnt_precip_VCSN_albedo2 = amount_precipitation(maximum_albedo_precip_obs, inp_swe2, VCSN_sumprecip)
            except :
                amnt_precip_VCSN_albedo2 = 'No data'
            print('Max albedo obs precip :', maximum_albedo_precip_obs, 'Date max obs precip :',date_max_albedo_precip_obs,'Snow melt albedo precip obs :', snw_melt_albedo_precip_obs, 'Obs precip : ',amnt_precip_obs_albedo2, 'VCSN precip :', amnt_precip_VCSN_albedo2)
        except:
            maximum_albedo_precip_obs = 'No data'
            minimum_albedo_precip_obs = 'No data'
            date_max_albedo_precip_obs = 'No data'
            date_min_albedo_precip_obs = 'No data'
            snw_melt_albedo_precip_obs = 'No data'
            amnt_precip_obs_albedo2 = 'No data'
            amnt_precip_VCSN_albedo2 = 'No data'
            print('Albedo obs ERROR {}'.format(Year))

        try :
            maximum_albedo_precip_VCSN, minimum_albedo_precip_VCSN, date_max_albedo_precip_VCSN, date_min_albedo_precip_VCSN = maxmin(inp_time2a, inp_swe2a)
            try :
                snw_melt_albedo_precip_VCSN = amount_snowmelt(maximum_albedo_precip_VCSN, inp_time2a, inp_swe2a)
            except :
                snw_melt_albedo_precip_VCSN = 'No data'
            try :
                amnt_precip_obs_albedo2a = amount_precipitation(maximum_albedo_precip_VCSN, inp_swe2a, obs_sumprecip)
            except :
                amnt_precip_obs_albedo2a = 'No data'
            try :
                amnt_precip_VCSN_albedo2a = amount_precipitation(maximum_albedo_precip_VCSN, inp_swe2a, VCSN_sumprecip)
            except :
                amnt_precip_VCSN_albedo2a = 'No data'
            print('Max albedo VCSN precip :', maximum_albedo_precip_VCSN,'Date max VCSN precip :', date_max_albedo_precip_VCSN)
        except :
            maximum_albedo_precip_VCSN = 'No data'
            minimum_albedo_precip_VCSN = 'No data'
            date_max_albedo_precip_VCSN = 'No data'
            date_min_albedo_precip_VCSN = 'No data'
            snw_melt_albedo_precip_VCSN = 'No data'
            amnt_precip_obs_albedo2a = 'No data'
            amnt_precip_VCSN_albedo2a = 'No data'
            print('Albedo VCSN ERROR {}'.format(Year))

        # csv file writing
        try :
            maxmin_writer.writerow({'Data': 'Observed precipitation dsc_snow-param albedo', 'Year': Year, 'SWE max': maximum_albedo_precip_obs, 'Date max': date_max_albedo_precip_obs, 'SWE min': minimum_albedo_precip_obs,'Date min': date_min_albedo_precip_obs,
                                    'Amount of obs precipitation':amnt_precip_obs_albedo2,'Amount of VCSN precipitation' : amnt_precip_VCSN_albedo2, 'Amount of snow melt':snw_melt_albedo_precip_obs})
        except :
            print('ERROR {} observed precipitation albedo'.format(Year))
        try :
            maxmin_writer.writerow({'Data': 'VCSN precipitation dsc_snow-param albedo', 'Year': Year,'SWE max': maximum_albedo_precip_VCSN, 'Date max': date_max_albedo_precip_VCSN,'SWE min': minimum_albedo_precip_VCSN, 'Date min': date_min_albedo_precip_VCSN
                                       ,'Amount of obs precipitation':amnt_precip_obs_albedo2a,'Amount of VCSN precipitation' : amnt_precip_VCSN_albedo2a, 'Amount of snow melt':snw_melt_albedo_precip_VCSN})
        except:
            print('ERROR {} VCSN precipitation albedo'.format(Year))

        #########################################
        # Max and Min values, VC
        maximum_VC, minimum_VC, date_max_VC, date_min_VC = maxmin(inp_time3, inp_swe3)
        try :
            snw_melt_VC = amount_snowmelt(maximum_VC, inp_time3, inp_swe3)
        except :
            snw_melt_VC = 'No data'
        try :
            amnt_precip_obs_VC = amount_precipitation(maximum_VC, inp_swe3, obs_sumprecip)
        except :
            amnt_precip_obs_VC = 'No data'
        try :
            amnt_precip_VCSN_VC = amount_precipitation(maximum_VC, inp_swe3, VCSN_sumprecip)
        except :
            amnt_precip_VCSN_VC = 'No data'
        print('Max VC :', maximum_VC,'Min VC :', minimum_VC,'Date max :', date_max_VC,'Date min :', date_min_VC)

        # csv file writing
        maxmin_writer.writerow({'Data': 'VC', 'Year': Year, 'SWE max': maximum_VC, 'Date max': date_max_VC, 'SWE min': minimum_VC,'Date min': date_min_VC, 'Amount of obs precipitation': amnt_precip_obs_VC,
             'Amount of VCSN precipitation': amnt_precip_VCSN_VC, 'Amount of snow melt': snw_melt_VC})

        #########################################
        # Max and Min values, VN
        try :
            maximum_VN, minimum_VN, date_max_VN, date_min_VN = maxmin(inp_time4, inp_swe4)
            try :
                snw_melt_VN = amount_snowmelt(maximum_VN, inp_time4, inp_swe4)
            except :
                snw_melt_VN = 'No data'
            try :
                amnt_precip_obs_VN = amount_precipitation(maximum_VN, inp_swe4, obs_sumprecip)
            except :
                amnt_precip_obs_VN = 'No data'
            try :
                amnt_precip_VCSN_VN = amount_precipitation(maximum_VN, inp_swe4, VCSN_sumprecip)
            except :
                amnt_precip_VCSN_VN = 'No data'
            print('Max albedo obs precip :', maximum_VN, 'Date max obs precip :',date_max_VN)
        except :
            maximum_VN = 'No data'
            minimum_VN = 'No data'
            date_max_VN = 'No data'
            date_min_VN = 'No data'
            snw_melt_VN = 'No data'
            amnt_precip_obs_VN = 'No data'
            amnt_precip_VCSN_VN = 'No data'
            print('VN ERROR {}'.format(Year))

        # csv file writing
        try:
            maxmin_writer.writerow({'Data': 'VN', 'Year': Year, 'SWE max': maximum_VN, 'Date max': date_max_VN, 'SWE min': minimum_VN,'Date min': date_min_VN,
                                    'Amount of obs precipitation': amnt_precip_obs_VN, 'Amount of VCSN precipitation': amnt_precip_VCSN_VN,'Amount of snow melt': snw_melt_VN})
        except:
            print('ERROR {} VN'.format(Year))