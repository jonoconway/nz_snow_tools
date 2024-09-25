"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries, convert_datetime_julian_day
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates


# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']=1800
config['tacc'] = 274.15
config['tmelt'] = 273.65

# clark2009 melt parameters
config['mf_mean'] = 5
config['mf_amp'] = 1.5
config['mf_alb'] = 2.5
config['mf_alb_decay'] = 5
config['mf_ros'] = 4 # default 2.5
config['mf_doy_max_ddf'] = 356 # default 356
config['mf_doy_min_ddf'] = 173 # default 210

# load brewster glacier data
inp_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/Brewster/Final Brewster Datasets/updated_met_data/BrewsterGlacier_Oct10_Sep12_mod3.dat')
start_t = 9600 -1# 9456 = start of doy 130 10th May 2011 9600 = end of 13th May,18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 21360  # 20783 = end of doy 365, 21264 = end of 10th January 2012, 21360 = end of 12th Jan
inp_dt = make_regular_timeseries(dt.datetime(2010,10,25,00,30),dt.datetime(2012,9,2,00,00),1800)

inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[start_t:end_t, 7][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[start_t:end_t, 21][:, np.newaxis] * np.ones(grid_size) # precip in mm
inp_sw = inp_dat[start_t:end_t, 15][:, np.newaxis] * np.ones(grid_size)
inp_sfc = inp_dat[start_t-1:end_t, 19] # surface height change
inp_sfc -= inp_sfc[0]# reset to 0 at beginning of period

# validation data
seb_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/Brewster/Final Brewster Datasets/SEB_output/cdf - code2p0_MC_meas_noQPS_single_fixed output_fixed_B/modelOUT_br1_headings.txt',skip_header=3)
seb_mb = seb_dat[start_t-1:end_t, -1]
seb_mb -= seb_mb[0] # reset to 0

# read in measured daily SEB change
mb_dat = np.genfromtxt(
    'C:/Users/conwayjp/OneDrive - NIWA/projects/Brewster/Final Brewster Datasets/mass_balance_validation/5 MB scatters/mchange.dat')
# note that the measured MB interprets surface height loss in the winter as mass loss, rather than compaction.
mb_dt = make_regular_timeseries(dt.datetime(2010,10,26,00,00),dt.datetime(2012,9,2,00,00),86400)
ts_mb = plt.cumsum(mb_dat[:,0])
np.where(np.asarray(mb_dt)==dt.datetime(2011,5,13,00,00))
ts_mb -= ts_mb[199]
#

init_swe = np.ones(inp_ta.shape[1:]) * 0  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:]) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.
st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                           init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

plot_dt = inp_dt[start_t-1:end_t] # model stores initial state


def objective(x_t, a_t, b_t, c_t, d_t):
    return (a_t - d_t) / (1 + b_t ** (x_t - c_t)) + d_t


plot_doy = np.asarray(convert_datetime_julian_day(plot_dt))
plot_doy[plot_doy<90]+=365
plot_dens = objective(plot_doy,567.1,0.925,270.3,383.5)

plt.plot(plot_dt,seb_mb, label='SEB')
plt.plot(plot_dt,st_swe[:, 0],label='clark2009')
# plt.plot(plot_dt,inp_sfc*492,label='sfc*492')
plt.plot(plot_dt,inp_sfc*plot_dens,label='sfc*calc_dens')
plt.plot([dt.datetime(2011,7,18),dt.datetime(2011,10,27),dt.datetime(2011,11,13)],[577,1448,1291],'o',label='stake_mb') # measured accumualation to 27th November 2011
#plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('month')
plt.ylabel('SWE mm w.e.')
plt.legend()
plt.title('{},{},{},{},{},{},{},{},{}'.format(config['tacc'],config['tmelt'],config['mf_mean'],config['mf_amp'],config['mf_alb'],config['mf_alb_decay'],config['mf_ros'],config['mf_doy_max_ddf'],config['mf_doy_min_ddf']))
plt.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/archive/2019 SIN internship/Brewster winter Spring 2011 {}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(config['tacc'],config['tmelt'],config['mf_mean'],config['mf_amp'],config['mf_alb'],config['mf_alb_decay'],config['mf_ros'],config['mf_doy_max_ddf'],config['mf_doy_min_ddf']))
plt.close()

