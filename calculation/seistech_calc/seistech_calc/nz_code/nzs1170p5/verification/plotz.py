import pandas as pd

# points.xyz can be generated by the following commands
#
#  gmt grdmath -R166/180/-48/-34/ -I2k 1 1 NAN = tmp.grd
#  gmt grd2xyz tmp.grd > points.xyz
#
#  and plotted with
#  "/nesi/project/nesi00213/visualization/visualization/gmt/plot_items.py --xyz-grid --xyz-grid-search 1.1m \
#  --xyz-landmask --xyz-cpt rainbow --xyz-grid-contours --xyz-transparency 30 --xyz-size 1k --xyz-cpt-inc 0.02
#  --xyz-cpt-tick 0.1 --xyz-cpt-min 0.1 --xyz-cpt-max 0.6 --xyz z_factor.xyz -f "Z Factor" --xyz-cpt-labels ""
#  --xyz-cpt-fg red"

stations = pd.read_csv('points.xyz', header=None, names=["lon", "lat", "site"], delim_whitespace=True)

z_values = nz_code.nzs1170p5.nzs_zfactor_2016.ll2z.ll2z(stations[['lon', 'lat']].values)

stations['z'] = z_values

stations[['lon', 'lat', 'z']].to_csv('plot.xyz', header=None, sep=' ', index=False)



