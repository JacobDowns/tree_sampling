import os
import geopandas as gpd
import rioxarray as rio
import matplotlib.pyplot as plt

from raster import RasterConnection


class CanopyHeightModelConnection(RasterConnection):
    """
    Creates a RasterConnection with a connection type of "rioxarray" for
    Canopy Height Model rasters. The URL for the raster is constructed from the
    version and product parameters.
    """

    def __init__(self, roi: gpd.GeoDataFrame, **kwargs):
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

        self.chm_tile_name = self.find_chm_tile(roi)
        self.bucket_url = (
            "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/"
        )
        self.url = self.bucket_url + self.chm_tile_name + ".tif"
        super().__init__(self.url, connection_type="rioxarray", **kwargs)

    @staticmethod
    def find_chm_tile(roi: gpd.GeoDataFrame) -> str:
        """
        Find the CHM tile that intersects the given ROI.
        """
        new_roi = roi.copy()
        new_roi = new_roi.to_crs("EPSG:4326")
        tile_gdf = gpd.read_file("data/tiles.geojson")
        map_tile_gdf = tile_gdf[tile_gdf.intersects(new_roi.union_all())]
        map_tile_row = map_tile_gdf.iloc[0]
        return map_tile_row["tile"]
