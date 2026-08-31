import xarray as xr
import matplotlib.pylab as plt


infile = r'C:\Users\conwayjp\Downloads\nzra_solar_1990010100-utc.nc'
# infile = r'\\niwa.local\projects\christchurch\SSIFEG2704\Working\NZRA_solar\nzra_solar_1990010100-utc.nc'
infile_static = r'\\niwa.local\projects\christchurch\SSIFEG2704\Working\NZRA_solar\nzra_static.nc'
ds = xr.open_dataset(infile)
static = xr.open_dataset(infile_static)

# cloud = xr.open_dataset(r'\\niwa.local\projects\christchurch\SSIFEG2704\Working\NZRA_solar\nzra_cloud_2019010100-utc_ll.nc')
DW_SW_mean = xr.open_dataset( r'C:\Users\conwayjp\Downloads\nzra_DW_SW_mean_1990010100-utc_ll.nc')

print()

min_lon = ds.longitude.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).min()
max_lon = ds.longitude.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).max()
min_lat = ds.latitude.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).min()
max_lat = ds.latitude.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).max()
# ds.latitude.sel(rlat=slice(-4, -3), rlon=slice(178.3, 179.3)).diff(dim="rlat").mean() = 0.0134 degrees =  1490 m
# ds.longitude.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).diff(dim="rlon").mean() = 0.0187 degrees = 1500 at 44S

plt.figure(); static.orog_model.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).plot()

plt.figure(); ds.mean_sfc_direct_sw_flux.isel(time1=6).sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).plot()
plt.figure(); ds.mean_sfc_diffuse_sw_flux.isel(time1=6).sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).plot()
plt.figure(); ds.mean_sfc_net_dn_sw_flux.isel(time1=6).sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3)).plot()
# plt.figure(); cloud.total_cloud_max_rnd_ovrlap.isel(time=6).sel(latitude=slice(min_lat,max_lat),longitude=slice(min_lon,max_lon)).plot()
plt.figure(); DW_SW_mean.mean_sfc_dw_sw_flux.isel(time=6).sel(latitude=slice(min_lat,max_lat),longitude=slice(min_lon,max_lon)).plot()



elev = static.orog_model.sel(rlat=slice(-4,-3),rlon=slice(178.3,179.3))

import numpy as np
def calc_slope_aspect(dem,dxdy):
    # gridslo = 0
    # gridasp = 0
    # assume a square, axis-oriented grid
    gx, gy = np.gradient(dem, dxdy)
    gridslo = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    # if origin == 'topleft':
    #     data = - np.pi / 2. - np.arctan2(-gx, gy)
    # elif origin == 'bottomleft':
    # assume grid has origin in SW corner ('bottomleft'
    data = - np.pi / 2. - np.arctan2(gx, gy)
    data = np.where(data < -np.pi, data + 2 * np.pi, data)
    gridasp = np.mod(np.degrees(data), 360)
    return gridslo,gridasp

inp_gridslo, inp_gridasp  = calc_slope_aspect(elev.squeeze().values,1500.0)
plt.imshow(inp_gridslo,origin='lower')
plt.figure()
plt.imshow(inp_gridasp,origin='lower',cmap=plt.cm.gist_ncar)