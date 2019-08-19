"""
code to call the snow model for a simple test case using brewster glacier data

trying to find if there are any good values to use for TF and TT
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import datetime as dt
from scipy.optimize import curve_fit
from scipy.stats import linregress

from nz_snow_tools.util.utils import nash_sut, mean_bias, rmsd, mean_absolute_error, coef_determ


# define line with intercept at 0.
def f(x, A):  # this is your 'straight line' y=f(x)
    return A * x


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
# calculate the temperature dependent flux
t_dep_flux = lw_net + qs + ql + qc + qst
# calculate the melt energy - sw_net and precip energy (what the ETI TF accounts for)
qm_wo_sw_prc = qm - sw_net - qprc
# reset to 0 when no melt, as SWnet + qm_wo_sw_prc should never be negative.
qm_wo_sw_prc[(qm == 0)] = 0

ta = seb_dat[:, 8 - 1]
ea = seb_dat[:, 10 - 1]
ws = seb_dat[:, 7 - 1]

# plot for all melting periods
plt.figure()
plt.hexbin(ta[~(qm_wo_sw_prc == 0)], qm_wo_sw_prc[~(qm_wo_sw_prc == 0)], cmap=plt.cm.inferno_r)
# all periods with melt

a2 = linregress(ta[~(qm_wo_sw_prc == 0)], qm_wo_sw_prc[~(qm_wo_sw_prc == 0)])
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * a2[0] - a2[1], 'b')
a = curve_fit(f, ta[~(qm_wo_sw_prc == 0)], qm_wo_sw_prc[~(qm_wo_sw_prc == 0)])
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * a[0][0] + a[1][0], 'b--')

# all periods
b2 = linregress(ta, qm_wo_sw_prc)
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * b2[0] - b2[1], 'r')
b = curve_fit(f, ta, qm_wo_sw_prc)
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * b[0][0] - b[1][0], 'r--')

# all periods with positive melt energy from Ta
c2 = linregress(ta[(qm_wo_sw_prc > 0)], qm_wo_sw_prc[(qm_wo_sw_prc > 0)])
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * c2[0] - c2[1], 'g')
c = curve_fit(f, ta[qm_wo_sw_prc > 0], qm_wo_sw_prc[qm_wo_sw_prc > 0])
plt.plot(np.linspace(-5, 12.5, 100), np.linspace(-5, 12.5, 100) * c[0][0] - c[1][0], 'g--')

plt.axhline(0, color='k')
plt.ylim([-150, 300])
plt.legend(['melting periods', 'melting periods intercept=0','all periods', 'all periods intercept=0', 'positive T melt', 'positive T melt intercept=0'])
plt.ylabel('QM - SWnet - Qprecip')
plt.xlabel('Air temperature (C)')
plt.savefig(r'D:\Snow project\Oct2018 Results\tf_derive.png')
plt.show()
r2_ta_melt = coef_determ(qm_wo_sw_prc[~(qm_wo_sw_prc == 0)], ta[~(qm_wo_sw_prc == 0)])
r2_ta = coef_determ(qm_wo_sw_prc, ta)
r2_ta_pos = coef_determ(qm_wo_sw_prc[(qm_wo_sw_prc > 0)], ta[(qm_wo_sw_prc > 0)])
