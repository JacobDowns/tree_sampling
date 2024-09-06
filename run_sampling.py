import rioxarray as rio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import json
from sampling_functions import sample_trees
from scipy.ndimage import maximum_filter

#####################################################
# Prep inputs for the sampler
#####################################################

region = 'sycan_marsh'

tree_population = gp.read_file(f'data/regions/{region}/tree_population.shp')

with open('data/spcd_parameters.json') as f:
    spcd_params = json.load(f)

# Load CHM raster and tree inventory
chm_raster = rio.open_rasterio(f'data/regions/{region}/chm_raster.tif')
# Pixel size
dx = abs(chm_raster.x.data[1] - chm_raster.x.data[0])


def get_inputs(chm_raster, tree_population):
    
    tree_population = tree_population.dropna()

    # Get the trait score for each tree
    trait_score = np.array([spcd_params[spcd]['PURVES_TRAIT_SCORE'] for spcd in tree_population['SPCD'].to_numpy().astype(str)])
    tree_population['TRAIT_SCORE'] = trait_score

    # Compute a normalized tree height based on trees in each plot 
    tree_population['HT_NORMALIZED'] = tree_population.groupby('PLOT_ID')['HT'].transform(lambda x: x / (x.max() + 0.001))
    
    # Tree coordinates
    x = tree_population['X'].to_numpy()
    y = tree_population['Y'].to_numpy()

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    
    # Constrain to the tree inventory
    chm_raster = chm_raster.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
    chm = chm_raster.data[0].astype(np.float32)
    
    # Compute a normalized CHM
    normalized_chm = chm / (maximum_filter(chm, size=30) + 1e-3)
    normalized_chm[np.isnan(normalized_chm)] = 0.
        
    # Tree properties as numpy array
    tree_props = np.column_stack([
        tree_population['DIA'].to_numpy(),
        tree_population['HT_NORMALIZED'].to_numpy(),
        tree_population['CR'].to_numpy(),
        tree_population['TRAIT_SCORE'].to_numpy()
    ]).astype(np.float32)

   
    # Find nearest pixel in the CHM to each tree
    x = tree_population.geometry.x.to_numpy() - x_min
    y = tree_population.geometry.y.to_numpy() - y_min
    x = np.rint(x / dx).astype(int)
    y = np.rint(y / dx).astype(int)
    xi = chm.shape[0] - y 
    xj = x
    xi[xi < 0] = 0
    xi[xi > normalized_chm.shape[0] - 1] = normalized_chm.shape[0] - 1
    xj[xj < 0] = 0
    xj[xj > normalized_chm.shape[1] - 1] = normalized_chm.shape[1] - 1
    tree_coords = np.c_[xi, xj]
    
    return chm_raster, normalized_chm, tree_population, tree_props, tree_coords

chm_raster, normalized_chm, tree_population, tree_props, tree_coords = get_inputs(chm_raster, tree_population)
chm = chm_raster.data[0].astype(np.float32)


#####################################################
# Perform sampling 
#####################################################

iterations = 50

# Maximum distance a tree can be from its original spot per iteration 
max_dist = np.ones(iterations)*30.
# The acceptable difference in the normalized heights between the CHM and the tree inventory 
height_error = 0.5 + np.linspace(0., 1., iterations)
# Paramter that reduces the radius of trees over subsequent iterations
scale = np.linspace(1., 0., iterations)**2

# Tree grid is 0 if there's no tree, or a positive value that relates to a specific tree index if there's a tree on a given pixel
tree_grid = np.zeros_like(normalized_chm, dtype=np.int64)

# Do the sampling
tree_grid, trees_to_place = sample_trees(normalized_chm, tree_grid, tree_coords, tree_props, dx, scale, max_dist, height_error)
#print(trees_to_place)

#####################################################
# Plot the tree sample and the CHM
#####################################################

chm_raster.rio.to_raster(f'data/results/{region}/chm.tif')

height = np.zeros_like(normalized_chm)
indexes = (tree_grid >= 1)

tree_coords =  np.argwhere(tree_grid >= 1)
tree_indexes = tree_grid[indexes] - 1
height[indexes] = tree_population['HT'].to_numpy()[tree_indexes]
height1 = maximum_filter(height, 5)

# Convert back to map coordinates
coords = tree_coords[:,::-1] * dx 
tree_population['X'] = tree_population['X'].to_numpy().min() + coords[:,0]
tree_population['Y'] = tree_population['Y'].to_numpy().max() - coords[:,1]
tree_population = tree_population.drop(['geometry', 'TRAIT_SCORE', 'HT_NORMALIZED'], axis=1)
tree_population = pd.DataFrame(tree_population)
tree_population.to_csv(f'data/results/{region}/tree_inventory.csv')

plt.subplot(3,1,1)
chm_raster.plot()

plt.subplot(3,1,2)
chm_raster.plot()
plt.scatter( tree_population['X'].to_numpy(), tree_population['Y'].to_numpy(), s=1, color='red')

plt.subplot(3,1,3)
cmap = plt.get_cmap('viridis', 32)
plt.imshow(height1, cmap=cmap)
plt.colorbar()

plt.show()

