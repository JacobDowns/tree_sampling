import numpy as np
import xarray as xr
import rioxarray as rio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from chm_connection import CanopyHeightModelConnection
from treemap import TreeMapConnection
from scipy.interpolate import griddata, NearestNDInterpolator

region = 'sycan_marsh'

# Load the roi polygon
roi_gdf = gpd.read_file(f"data/regions/{region}/roi.geojson")
# Reproject to 5070
roi = roi_gdf.to_crs(5070)

# Initialize the TreeMapConnection connection
treemap_connection = TreeMapConnection(version="2016")

"""
# # Extract the window using the TreeMapConnection connection
treemap_raster = treemap_connection.extract_window(
     roi, projection_padding_meters=150, interpolation_padding_cells=1
)

treemap_raster.plot()
plt.show()

# # Save the treemap rasters
treemap_raster.rio.to_raster(f"data/regions/{region}/treemap_raster.tif")
"""

"""
# Load the Treemap rasters from file
treemap_raster = rio.open_rasterio(f"data/regions/{region}/treemap_raster.tif")

# Resample the treemap raster to 7.5m resolution
new_height = treemap_raster["y"].shape[0] * 4
new_width = treemap_raster["x"].shape[0] * 4
treemap_raster_resampled = treemap_raster.rio.reproject(
    treemap_raster.rio.crs, shape=(new_height, new_width), resampling=Resampling.nearest
)
treemap_raster_resampled.rio.to_raster(f"data/regions/{region}/treemap_raster_resampled.tif")
"""

# Load the resampled Treemap raster
treemap_raster_resampled = rio.open_rasterio(f"data/regions/{region}/treemap_raster_resampled.tif")

# # Extract the window using the CanopyHeightModelConnection connection
chm_connection = CanopyHeightModelConnection(roi=roi_gdf)
chm_raster = chm_connection.extract_window(
     roi, projection_padding_meters=1000, interpolation_padding_cells=100
)

# Save the chm raster
chm_raster.rio.to_raster(f"data/regions/{region}/chm_raster.tif")


# Load the CHM raster from file
chm_raster = rio.open_rasterio(f"data/regions/{region}/chm_raster.tif")

# Update the chm raster to have a nodata value of 0
chm_raster = chm_raster.fillna(0)
chm_raster.rio.write_nodata(0, inplace=True)

# Calculate new dimensions for 7.5m resolution for the CHM raster
original_height, original_width = chm_raster["y"].shape[0], chm_raster["x"].shape[0]
original_res_y, original_res_x = chm_raster.rio.resolution()
target_res = 7.5
new_height = int(round((original_height * abs(original_res_y)) / target_res))
new_width = int(round((original_width * abs(original_res_x)) / target_res))

horizontal_scale = original_width / new_width
vertical_scale = original_height / new_height
print(f"Horizontal scale: {horizontal_scale}")
print(f"Vertical scale: {vertical_scale}")


# Resample the CHM raster to 7.5m resolution by summing the values in each 7.5m cell
chm_raster_high_res_occupied_mask = xr.where(chm_raster > 1, 1.0, 0.0)
chm_raster_resampled = chm_raster_high_res_occupied_mask.rio.reproject(
    chm_raster.rio.crs,
    shape=(new_height, new_width),
    resampling=Resampling.sum,
)
chm_raster_resampled = chm_raster_resampled.fillna(0)
chm_raster_resampled.rio.write_nodata(0, inplace=True)
chm_raster_resampled /= horizontal_scale
chm_raster_resampled /= vertical_scale

chm_raster_resampled.rio.to_raster(f"data/regions/{region}/chm_raster_resampled.tif")

# Load the resampled CHM raster
chm_raster_resampled = rio.open_rasterio(f"data/regions/{region}/chm_raster_resampled.tif")


# Interpolate the 7.5m CHM raster to the 7.5m treemap raster using the nearest neighbor method
chm_x = chm_raster_resampled["x"].values
chm_y = chm_raster_resampled["y"].values
chm_xx, chm_yy = np.meshgrid(chm_x, chm_y)
treemap_x = treemap_raster_resampled["x"].values
treemap_y = treemap_raster_resampled["y"].values
treemap_xx, treemap_yy = np.meshgrid(treemap_x, treemap_y)
chm_values = chm_raster_resampled.values.ravel()
chm_interpolated = griddata(
    (chm_xx.ravel(), chm_yy.ravel()),
    chm_values,
    (treemap_xx, treemap_yy),
    method="nearest",
)
chm_raster_interpolated = xr.DataArray(
    chm_interpolated,
    coords={"y": treemap_y, "x": treemap_x},
    dims=["y", "x"],
    name="chm_interpolated",
)
chm_raster_interpolated.rio.write_crs(chm_raster_resampled.rio.crs, inplace=True)
chm_raster_interpolated.rio.write_nodata(chm_raster_resampled.rio.nodata, inplace=True)
chm_raster_interpolated.rio.to_raster(f"data/regions/{region}/chm_raster_interpolated.tif")

# Load the interpolated CHM raster
chm_raster_interpolated = rio.open_rasterio(f"data/regions/{region}/chm_raster_interpolated.tif")

"""
Create a new raster product that is the value of the nearest TreeMap
pixel where the CHM raster has a value greater than 0.25. Otherwise, the
value should be the nodata value of the TreeMap raster.
"""
# Create a mask where CHM > 0.25
chm_mask = chm_raster_interpolated > 0.25

# Get the coordinates of the valid TreeMap data points (non-nodata values)
valid_mask = (treemap_raster_resampled != treemap_raster_resampled.rio.nodata).values[0]
valid_coords = np.column_stack(np.where(valid_mask))
valid_values = treemap_raster_resampled.values[0][valid_mask]

# Create a NearestNDInterpolator object
interpolator = NearestNDInterpolator(valid_coords, valid_values)

# Create a grid of all coordinates
all_coords = np.column_stack(np.where(np.ones_like(treemap_raster_resampled.values[0])))

# Interpolate values for all coordinates
interpolated_values = interpolator(all_coords)

# Reshape the interpolated values to match the original raster shape
interpolated_raster = interpolated_values.reshape(treemap_raster_resampled.shape[1:])

# Create the final raster: use interpolated values where CHM > 0.25, and set to nodata elsewhere
treemap_chm_array = np.where(
    chm_mask.values[0], interpolated_raster, treemap_raster_resampled.rio.nodata
)

# Create an xarray DataArray for the result
treemap_chm_raster = xr.DataArray(
    treemap_chm_array[
        np.newaxis, :, :
    ],  # Add channel dimension to match original raster
    coords=treemap_raster_resampled.coords,
    dims=treemap_raster_resampled.dims,
    attrs=treemap_raster_resampled.attrs,
)
treemap_chm_raster.rio.write_crs(treemap_raster_resampled.rio.crs, inplace=True)
treemap_chm_raster.rio.write_nodata(treemap_raster_resampled.rio.nodata, inplace=True)

# Save the result
treemap_chm_raster.rio.to_raster(f"data/regions/{region}/treemap_chm_raster.tif")

# Load the treemap_chm_raster
treemap_chm_raster = rio.open_rasterio(f"data/regions/{region}/treemap_chm_raster.tif")

# Convert plots in the treemap raster to a geodataframe
treemap_plots = treemap_chm_raster.data.ravel()
treemap_x = treemap_chm_raster["x"].values
treemap_y = treemap_chm_raster["y"].values
treemap_xx, treemap_yy = np.meshgrid(treemap_x, treemap_y)
plots_gdf = gpd.GeoDataFrame(
    {
        "PLOT_ID": treemap_plots,
        "X": treemap_xx.ravel(),
        "Y": treemap_yy.ravel(),
        "geometry": gpd.points_from_xy(treemap_xx.ravel(), treemap_yy.ravel()),
    },
    crs=treemap_chm_raster.rio.crs,
)

# Get the trees in the plots
tree_sample = treemap_connection.query_trees_by_plots(plots_gdf)

# Expand the trees to a tree population
tree_population = tree_sample.expand_to_roi(
    "inhomogeneous_poisson", roi, plots=plots_gdf, intensity_resolution=7.5 / 2, seed=42
)

# Save the tree population as a shapefile
tree_population.to_file(f"data/regions/{region}/tree_population.shp")

tree_population["X"] = tree_population.geometry.x
tree_population["Y"] = tree_population.geometry.y 
tree_population.drop(columns=["geometry"], inplace=True)
tree_population.to_csv(f"data/regions/{region}/tree_population.csv", index=False)
