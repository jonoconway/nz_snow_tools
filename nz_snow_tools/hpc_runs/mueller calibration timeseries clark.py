"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import pickle as pkl


plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2201/snow_model_ensembles/clark_output'
model = 'clark'
# run_id = 'eti - optA Tmelt276.16 TF.04,RF.0108, frs=0.85,os=.5'

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output'] = 3600
config['tacc'] = 274.15 # default 274.15
config['tmelt'] = 273.65 # default 273.15

# clark2009 melt parameters
config['mf_mean'] = 5# default 5, new 4
config['mf_amp'] = 1.5 # default 5, new 2.5
config['mf_alb'] = 2.5 # default 2.5, new 1.5
config['mf_alb_decay'] = 5 # default 5 (clark) or 1 (topnet), new 5 # timescale for adjustment (after this time effect will be 37% of full effect)
config['mf_ros'] = 4 # default 2.5 (clark) or 0 (topnet), new 4
config['mf_doy_max_ddf'] = 356  # default 356, new 356
config['mf_doy_min_ddf'] = 173 # default 173 (clark) or 210 (topnet), new 173
#
# # dsc_snow melt parameters
config['tf'] = 0.04 * 24  # hamish 0.13. ruschle 0.04, pelliciotti 0.05
config['rf'] = 0.0108 * 24  # hamish 0.0075,ruschle 0.009, pelliciotti 0.0094 # ruschle before 1 Oct 0.157. theoretical 0.0108
# dsc_snow albedo parameters
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.85
config['a_firn'] = 0.5
config['dc'] = 11.0
config['alb_swe_thres'] = 10
config['ros'] = False
config['ta_m_tt'] = False

# load input data for mueller hut
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/FWWR_SIN/data processing/point_model_inputs/mueller_hut/mueller_hut_met_20170501_20200401_with_hs_swe_rain_withcloud_precip_harder_update1.pkl'
aws_df = pkl.load(open(infile, 'rb'))
start_t = 0
end_t = 15840
inp_dt = [i.to_pydatetime() for i in aws_df.index[start_t:end_t]]
inp_doy = [i.dayofyear for i in aws_df.index[start_t:end_t]]
inp_hourdec = [i.hour for i in aws_df.index[start_t:end_t]]

# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = aws_df.tc[start_t:end_t][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = aws_df.precip[start_t:end_t][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = aws_df.srad[start_t:end_t][:, np.newaxis] * np.ones(grid_size)

init_swe = np.ones(inp_ta.shape[1:]) * 0  # give initial value of swe
init_d_snow = np.ones(inp_ta.shape[1:]) * 30  # give initial value of days since snow

# call main function once hourly/sub-hourly temp and precip data available.

if model == 'clark':
    st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
elif model == 'eti':
    st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
# st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
#                                                        init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)


# observed SWE
obs_swe = aws_df.swe[start_t:end_t] * 1000  # measured swe - convert to mm w.e.
plot_dt = inp_dt  # model stores initial state

plt.figure(figsize=(5.5,4))
plt.plot(plot_dt, obs_swe, label='obs')
plt.plot(plot_dt, st_swe[1:, 0], label='mod')
# plt.plot(plot_dt, st_swe1[1:, 0], label='dsc_snow-param albedo')

# plt.xticks(range(0,len(st_swe[:, 0]),48*30),np.linspace(inp_doy[0],inp_doy[-1]+365+1,len(st_swe[:, 0])/(48*30.)+1,dtype=int))
# plt.gcf().autofmt_xdate()
months = mdates.MonthLocator(interval=3)  # every month
# days = mdates.DayLocator(interval=1)  # every day interval = 7 every week
monthsFmt = mdates.DateFormatter('%b')
ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)

plt.xlabel('Month')
plt.ylabel('SWE (mm w.e.)')
plt.legend()
# plt.savefig(plot_folder + '/{}.png'.format(run_id),dpi=300)
plt.title('{},{},{},{},{},{},{},{},{}'.format(config['tacc'],config['tmelt'],config['mf_mean'],config['mf_amp'],config['mf_alb'],config['mf_alb_decay'],config['mf_ros'],config['mf_doy_max_ddf'],config['mf_doy_min_ddf']))
plt.tight_layout()
plt.savefig(plot_folder + '/Mueller_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(model,config['tacc'],config['tmelt'],config['mf_mean'],config['mf_amp'],config['mf_alb'],config['mf_alb_decay'],config['mf_ros'],config['mf_doy_max_ddf'],config['mf_doy_min_ddf']),dpi=300)
plt.close()

#sensitivity
if model == 'clark':
    st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta + 1, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)

elif model == 'eti':
    st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta + 1, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

swe = st_swe.mean()
swe1 = st_swe1.mean()
print('SWE {:.1f}'.format(swe))
print('SWE +1K {:.1f}'.format(swe1))
print('dSWE {:.1f}'.format(swe-swe1))

scd = np.sum(st_swe > 10)/24/2
scd1 = np.sum(st_swe1 > 10)/24/2

print('SCD {:.1f}'.format(scd))
print('SCD +1K {:.1f}'.format(scd1))
print('dSCD {:.1f}'.format(scd-scd1))
print('SWE sensitivity {:.1f}'.format((1 - (swe1/swe))*100))
print('SCD sensitivity {:.1f}'.format((scd-scd1)/scd*100))

print('{:.1f}'.format((1 - (swe1/swe))*100))
print('{:.1f}'.format((1 - (scd1/scd))*100))
print('{:.1f}'.format((swe-swe1)))
print('{:.1f}'.format((scd-scd1)))

# simulation for 400 m lower

if model == 'clark':
    st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta + 2, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                               init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
    st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta + 3, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='clark2009', **config)
elif model == 'eti':
    st_swe, st_melt, st_acc, st_alb = snow_main_simple(inp_ta + 2, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)
    st_swe1, st_melt1, st_acc1, st_alb1 = snow_main_simple(inp_ta + 3, inp_precip, inp_doy, inp_hourdec, dtstep=3600, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)

print('simulation for 400 m lower')
swe = st_swe.mean()
swe1 = st_swe1.mean()
print('SWE {:.1f}'.format(swe))
print('SWE +1K {:.1f}'.format(swe1))
print('dSWE {:.1f}'.format(swe-swe1))
scd = np.sum(st_swe > 10) / 24 / 3
scd1 = np.sum(st_swe1 > 10) / 24 / 3

print('SCD {:.1f}'.format(scd))
print('SCD +1K {:.1f}'.format(scd1))
print('dSCD {:.1f}'.format(scd - scd1))
print('SWE sensitivity {:.1f}'.format((1 - (swe1 / swe)) * 100))
print('SCD sensitivity {:.1f}'.format((scd - scd1) / scd * 100))

print('{:.1f}'.format((1 - (swe1/swe))*100))
print('{:.1f}'.format((1 - (scd1/scd))*100))
print('{:.1f}'.format((swe-swe1)))
print('{:.1f}'.format((scd-scd1)))