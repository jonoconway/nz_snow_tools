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

#MUELLER station data for each year
# Y2011_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2011.npy"
# Y2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2012.npy"
# Y2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2013.npy"
# Y2014_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2014.npy"
# Y2015_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2015.npy"
# Y2016_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2016.npy"
# Y2017_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2017.npy"
# Y2018_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_2018.npy"

#MAHANGA station data for each year
Y2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2009.npy"
Y2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2010.npy"
Y2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2011.npy"
Y2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2012.npy"
Y2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2013.npy"
Y2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2014.npy"
Y2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2015.npy"
Y2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2016.npy"
Y2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2017.npy"
Y2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mahanga/Mahanga_2018.npy"

#LARKINS station data for each year
# Y2014_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_2014.npy"
# Y2015_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_2015.npy"
# Y2016_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_2016.npy"
# Y2017_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_2017.npy"
# Y2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Larkins/Larkins_2018.npy"

#CASTLE MOUNT station data for each year
# Y2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_2012.npy"
# Y2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_2013.npy"
# Y2014_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_2014.npy"
# Y2015_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_2015.npy"
# Y2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_2016.npy"

#MURCHISON station data for each year
# Y2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2009.npy"
# Y2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2010.npy"
# Y2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2011.npy"
# Y2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2012.npy"
# Y2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2013.npy"
# Y2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2014.npy"
# Y2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2015.npy"
# Y2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2016.npy"
# Y2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2017.npy"
# Y2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Murchison/Murchison_2018.npy"
#
#PHILISTINE station data for each year
# Y2011_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2011.npy"
# Y2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2012.npy"
# Y2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2013.npy"
# Y2014_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2014.npy"
# Y2015_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2015.npy"
# Y2016_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2016.npy"
# Y2017_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2017.npy"
# Y2018_file = "C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Philistine/Philistine_2018.npy"

year = 2009
Stname = ['Mahanga']

# load data
inp_dat = np.load(Y2009_file,allow_pickle=True)
inp_doy = np.asarray(convert_datetime_julian_day(inp_dat[:, 0]))
inp_hourdec = convert_dt_to_hourdec(inp_dat[:, 0])
plot_dt = inp_dat[:, 0] # model stores initial state

# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = np.asarray(inp_dat[:,2],dtype=np.float)[:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = np.asarray(inp_dat[:,4],dtype=np.float)[:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = np.asarray(inp_dat[:,3],dtype=np.float)[:, np.newaxis] * np.ones(grid_size)


init_swe = np.ones(inp_ta.shape[1:],dtype=np.float) * 0  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:],dtype=np.float) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600,init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

# MUELLER SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv"
# MAHANGA SWE csv file
csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mahanga SWE.csv"
# LARKINS SWE csv file
# csv_file ="C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Larkins SWE.csv"
# CASTLE MOUNT SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Castle Mount SWE.csv"
# MURCHISON SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Murchison SWE.csv"
# PHILISTINE SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Philistine SWE.csv"

# load observed data
inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),
                        skip_header=4)*1000
inp_timeobs = np.genfromtxt(csv_file, usecols=(0),
                         dtype=(str), delimiter=',', skip_header=4)
inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])

ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])

# print('rmsd = {}'.format(rmsd(mod,obs)))
plt.plot(inp_dtobs[ind],inp_datobs[ind],"o", label = "Observed SWE")

plt.plot(plot_dt,st_swe[1:, 0],label='clark2009')
plt.plot(plot_dt,st_swe1[1:, 0],label='dsc_snow-param albedo')
plt.legend()
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(plot_dt,np.cumsum(inp_precip), label = "Precipitation")
#plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('SWE mm w.e.')
plt.legend(loc = 'upper right')
#Change the DATE
plt.title('Cumulative mass balance TF:{:2.4f}, RF: {:2.4f}, Tmelt:{:3.2f}, Year : {}'.format(config['tf'],config['rf'],config['tmelt'], year))
#Change the DATE on the name for each year
plt.savefig("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{} {} daily TF{:2.4f}RF{:2.4f}Tmelt{:3.2f}_ros.png".format(Stname[0],Stname[0],year, config['tf'],config['rf'],config['tmelt']))
plt.show()
plt.close()

# load VCSN files
# CASTLE MOUNT
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_CastleMo_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_CastleMo_strahler3-VN.nc",'r')
# LARKINS
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Larkins_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Larkins_strahler3-VN.nc",'r')
#MAHANGA
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Mahanga_strahler3-VC.nc",'r')
#MUELLER
# nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Mueller_strahler3-VC.nc",'r')
# nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Mueller_strahler3-VN.nc",'r')
#PHILISTINE
nc_file_VC = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VC_2007-2019/tseries_2007010122_2019013121_utc_topnet_Philisti_strahler3-VC.nc",'r')
nc_file_VN = nc.Dataset(r"C:/Users/Bonnamourar/Desktop/SIN/VCSN/VN_2007-2017/tseries_2007010122_2017123121_utc_topnet_Philisti_strahler3-VN.nc",'r')

swe_VC = nc_file_VC.variables['snwstor'][:,0,0,0] #snow storage VC files
swe_VN = nc_file_VN.variables['snwstor'][:,0,0,0] #snow storage VN files
nc_datetimes_VC = nc.num2date(nc_file_VC.variables['time'][:], nc_file_VC.variables['time'].units) #VC dates
nc_datetimes_VN = nc.num2date(nc_file_VN.variables['time'][:], nc_file_VN.variables['time'].units) #VN dates

ind_VC = np.logical_and(nc_datetimes_VC >= plot_dt[0],nc_datetimes_VC <= plot_dt[-1])
ind_VN = np.logical_and(nc_datetimes_VN >= plot_dt[0],nc_datetimes_VN <= plot_dt[-1])

swe_VC_year = swe_VC[ind_VC]
swe_VN_year = swe_VN[ind_VN]
year_VC = nc_datetimes_VC[ind_VC] #one year VC
year_VN = nc_datetimes_VN[ind_VN] #one year VN

plt.title('Models VC and VN for the year {}, Station : {}'.format(year, Stname[0]))
plt.plot(year_VC, swe_VC_year, label = 'VC', color = "magenta")
plt.plot(year_VN, swe_VN_year, label = 'VN', color = "salmon")
plt.legend()
plt.show()

# saved files
# LARKINS
# save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_{}"
# save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_{}"
# PHILISTINE
# save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_{}"
# save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_{}"
# CASTLE MOUNT
# save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_{}"
# save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_{}"
# MUELLER
# save_file_clark2009 = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_{}"
# save_file_albedo = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_{}"
# MURCHISON
# save_file_clark2009 ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_{}"
# save_file_albedo ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_{}"
# MAHANGA
save_file_clark2009 = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_{}"
save_file_albedo = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_{}"
# save the data : Clark 2009
output1 = np.transpose(np.vstack((plot_dt,st_swe[1:, 0])))
np.save(save_file_clark2009.format(year),output1)
# save the data : Albedo
output2 = np.transpose(np.vstack((plot_dt,st_swe1[1:, 0])))
np.save(save_file_albedo.format(year),output2)

