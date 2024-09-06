import rioxarray as rio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp

region = 'sycan_marsh'

tree_population = pd.read_csv(f'data/results/{region}/tree_inventory.csv')
chm_raster = rio.open_rasterio(f'data/results/{region}/chm.tif')

plt.subplot(2,1,1)
chm_raster.plot()

plt.subplot(2,1,2)
chm_raster.plot()
plt.scatter( tree_population['X'].to_numpy(), tree_population['Y'].to_numpy(), s=1, color='red')

plt.show()

