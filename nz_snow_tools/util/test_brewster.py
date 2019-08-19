import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt

projects = 'P:/Projects'
# testing the brewster dataset

# dem is 208 by 175, If brewster point was 69,73 before, it is 139,73 with grid starting in north. slope = 7.13 and azimuth -157, elevation 1783.5814

output_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\test_brewster\snow_out_BrewsterGlacierHourly_Apr11_Mar12_origintopleft.nc")
nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\test_brewster\brewster_topo_origintopleft.nc")

met_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\test_brewster\BrewsterGlacierHourly_Apr11_Mar12.nc")
met_dt = nc.num2date(met_file.variables['time'][:], met_file.variables['time'].units)

sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]
albedo = output_file.variables['albedo'][:]
sw_inp = output_file.variables['incoming_solar_radiation_to_surface_from_input'][:]
cloud = output_file.variables['cloud'][:]

sw_in = sw_net / (1. - albedo)

sw_met = met_file.variables['surface_downwelling_shortwave_flux'][:]

plt.plot(met_dt[:96], sw_met[:96], 'o-', label='sw_input')
plt.plot(nc_dt[:96], sw_inp[:96, 139, 73], 'o-', label='sw_input')
plt.plot(nc_dt[:96], sw_in[:96, 139, 73], 'o-', label='sw_mod [139,73]')
plt.plot(nc_dt[:96], cloud[:96, 139, 73] * 100, 'o-', label='cloudiness (%)')
fig = plt.gcf()
fig.autofmt_xdate()
plt.legend()
plt.ylabel('Flux density (W m^-2) or cloudiness (%)')
plt.xlabel('Time (NZST)')
plt.savefig(projects + '/DSC-Snow/runs/test_brewster/time series sw.png', dpi=300)
plt.close()

var_id = ['sw_net','albedo','sw_in']
for j, var in enumerate([sw_net,albedo,sw_in]):
    for i in range(24):
        plt.imshow(var[i])
        plt.colorbar()
        plt.contour(topo_file.variables['DEM'][:], range(0, 4000, 50), colors='k', linewidths=0.5)
        plt.savefig(projects + '/DSC-Snow/runs/test_brewster/p_{}_{}.png'.format(var_id[j],i), dpi=300)
        # plt.show()
        plt.close()


grd_names=['DEM','ice','catchment','viewfield','debris','slope','aspect']
for gname in grd_names:
    plt.figure()
    plt.imshow(topo_file.variables[gname][:])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(gname)
    plt.tight_layout()
    plt.savefig(projects + '/DSC-Snow/runs/test_brewster/checking_dem_{}.png'.format(gname), dpi=300)
    plt.close()