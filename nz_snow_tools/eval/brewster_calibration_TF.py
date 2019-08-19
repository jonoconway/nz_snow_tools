"""
code to call the snow model for a simple test case using brewster glacier data
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import datetime as dt

from nz_snow_tools.util.utils import resample_to_fsca, nash_sut, mean_bias, rmsd, mean_absolute_error, coef_determ

seb_dat = np.genfromtxt(
    'S:\Scratch\Jono\Final Brewster Datasets\SEB_output\cdf - code2p0_MC_meas_noQPS_single_fixed output_fixed_B\modelOUT_br1_headings.txt', skip_header=3)
sw_net = seb_dat[:, 14 - 1]
lw_net = seb_dat[:, 17 - 1]
qs = seb_dat[:, 19 - 1]
ql = seb_dat[:, 20 - 1]
qc = seb_dat[:, 21 - 1]
qprc = seb_dat[:, 22 - 1]
qst = seb_dat[:, 24 - 1]
qm = seb_dat[:, 25 - 1]
t_dep_flux = lw_net + qs + ql + qc + qst
qm_wo_sw_prc = qm - sw_net - qprc
qm_wo_sw_prc[(qm == 0)] = 0

ta = seb_dat[:, 8 - 1]
ea = seb_dat[:, 10 - 1]
ws = seb_dat[:, 7 - 1]

r2_ea = coef_determ(qm_wo_sw_prc, ea)
r2_ta = coef_determ(qm_wo_sw_prc, ta)
r2_ea_ws = coef_determ(qm_wo_sw_prc, ea*ws)

r2_ea_pos = coef_determ(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ea[(qm_wo_sw_prc > 0)])
r2_ta_pos = coef_determ(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ta[(qm_wo_sw_prc > 0)])
r2_ea_ws_pos = coef_determ(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ea[(qm_wo_sw_prc > 0)]*ws[(qm_wo_sw_prc > 0)])

print(r2_ea)
print(r2_ta)
print (r2_ea_ws)

print(r2_ea_pos)
print(r2_ta_pos)
print (r2_ea_ws_pos)

print(
np.sum(ta>0),
np.sum(np.logical_and(ta>0,qm_wo_sw_prc > 0)),
np.sum(qm_wo_sw_prc > 0),
np.sum(np.logical_and(ta>0,qm_wo_sw_prc > 0))/np.sum(ta>0),
)
print(
np.sum(ea>6.112),
np.sum(np.logical_and(ea>6.1120,qm_wo_sw_prc > 0)),
np.sum(qm_wo_sw_prc > 0),
np.sum(np.logical_and(ea>6.1120,qm_wo_sw_prc > 0))/np.sum(ea>6.112),
)



plt.figure()
plt.hexbin(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ta[(qm_wo_sw_prc > 0)], cmap=plt.cm.inferno_r)
plt.plot(range(200), np.arange(200) / 14.7,'k')
plt.plot(range(100), np.arange(100) / 8.7,'r')
plt.xlabel('QM - SWnet - Qprecip')
plt.ylabel('Air temperature (C)')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ta posQM.png')

plt.figure()
plt.hexbin(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ea[(qm_wo_sw_prc > 0)], cmap=plt.cm.inferno_r)
plt.plot(range(200), 6.112 + np.arange(200) / 42.0,'k')
plt.xlabel('QM - SWnet - Qprecip')
plt.ylabel('Vapour pressure (hPa)')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ea posQM.png')

plt.figure()
plt.hexbin(qm_wo_sw_prc[~(qm_wo_sw_prc == 0)], ta[~(qm_wo_sw_prc == 0)], cmap=plt.cm.inferno_r)
plt.plot(range(200), np.arange(200) / 14.7,'k')
plt.plot(range(100), np.arange(100) / 8.7,'r')
plt.xlabel('QM - SWnet - Qprecip')
plt.ylabel('Air temperature (C)')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ta.png')

plt.figure()
plt.hexbin(qm_wo_sw_prc[~(qm_wo_sw_prc == 0)], ea[~(qm_wo_sw_prc == 0)], cmap=plt.cm.inferno_r)
plt.plot(range(200), 6.112 + np.arange(200) / 42.0,'k')
plt.xlabel('QM - SWnet - Qprecip')
plt.ylabel('Vapour pressure (hPa)')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ea.png')

#plt.show()

print(
np.sum(qm_wo_sw_prc[qm>0])/sw_net.shape,# average positive melt energy from temp dep fluxes
np.sum(sw_net[qm>0])/sw_net.shape, # average melt energy from sw_net
np.sum(qprc[qm>0])/sw_net.shape # average melt energy from precipitation
)

qm_wo_sw_prc[qm_wo_sw_prc<0] = 0 # set all negative melt energy to zero

# find optimal parameters for ea and ta
from scipy.optimize import curve_fit

def f(x, A): # this is your 'straight line' y=f(x)
    return A*x

# sum melt energy from ea and ta
# melt factor was 0.025 mm w.e. per hour per hPa
ea_pos = ea-6.112
ea_pos[ea_pos<0] = 0
A = curve_fit(f,ea_pos, qm_wo_sw_prc)[0] # find optimal ea_q factor = 41.9
np.median(qm_wo_sw_prc[qm_wo_sw_prc>0]/ea_pos[qm_wo_sw_prc>0]) # median Wm^-2 per K = 41.7
ea_q = ea_pos * 42

# Wm^-2 per K (melt rate of 0.05 mm w.e. per hour per K = 4.6 Wm^-2 per K)
ta_pos = ta - 0.
ta_pos[ta_pos<0] = 0
A = curve_fit(f,ta_pos, qm_wo_sw_prc)[0]# find optimal ta_q factor = 8.7
np.median(qm_wo_sw_prc[qm_wo_sw_prc>0]/ta_pos[qm_wo_sw_prc>0]) # median Wm^-2 per K = 14.7
ta_q = ta_pos * 8.7

#K * / (mm w.e. W) *
print(
np.sum(qm_wo_sw_prc[qm>0])/sw_net.shape,# average positive melt energy from temp dep fluxes
np.sum(ea_q)/sw_net.shape, # average calculated melt energy from temp dep fluxes using ea
np.sum(ta_q)/sw_net.shape, # average calculated melt energy from temp dep fluxes using ta
np.sum(sw_net[qm>0])/sw_net.shape, # average melt energy from sw_net
np.sum(sw_net[np.logical_and(qm>0,ta<0)])/sw_net.shape, # average melt energy from sw_net when temperature below 0
np.sum(sw_net[np.logical_and(qm>0,ta>0)])/sw_net.shape, # average melt energy from sw_net when temperature above 0
np.sum(qprc[qm>0])/sw_net.shape # average melt energy from precipitation
)
plt.figure()
plt.hexbin(qm_wo_sw_prc[np.logical_and(ta_q>0,qm_wo_sw_prc>0)],ta_q[np.logical_and(ta_q>0,qm_wo_sw_prc>0)])
plt.plot(range(300),range(300),'b--')
plt.ylabel('mod'),plt.xlabel('obs'),plt.title('ta_q vs qm_wo_sw_prc')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ta_q.png')

plt.figure()
plt.hexbin(qm_wo_sw_prc[np.logical_and(ea_q>0,qm_wo_sw_prc>0)],ea_q[np.logical_and(ea_q>0,qm_wo_sw_prc>0)])
plt.ylabel('mod'),plt.xlabel('obs'),plt.title('ea_q vs qm_wo_sw_prc')
plt.plot(range(300),range(300),'b--')
plt.savefig(r'D:\Snow project\Oct2018 Results\qm_wo_sw_prc vs ea_q.png')

plt.figure()
plt.hist(qm_wo_sw_prc[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)]/ta_pos[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)],20)
plt.xlabel('ta_q_factor (W m-2 K-1)')
plt.savefig(r'D:\Snow project\Oct2018 Results\ta_q_factor_hist.png')
#plt.show()

print(
rmsd(qm_wo_sw_prc,ta_q),
rmsd(qm_wo_sw_prc,ea_q)
)

es = 6.1121 * np.exp(17.502*ta/(240.97+ta))
rh = (ea/es) * 100

plt.scatter(rh[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)]*ws[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)]/10.,qm_wo_sw_prc[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)]/ta_pos[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)],3)
plt.scatter(rh[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)],qm_wo_sw_prc[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)]/ta_pos[np.logical_and(ta_pos>0.5,qm_wo_sw_prc>0)])
plt.scatter(ql,qm_wo_sw_prc-ta_q)
plt.scatter(ta,qm_wo_sw_prc-ta_q)