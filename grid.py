# Core imports
from __future__ import annotations
import json
from math import floor, ceil

# External imports
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import numpy as np

class Grid2D:
   def __init__(
       self,
       bounds: dict,
       hr: float,
       padding: float = 0,
       origin="upper",
       cell_centered=True,
   ):
       self.bounds = bounds
       self.hr = hr
       self.origin = origin

       cell_centered_adjustment = self.hr / 2 if cell_centered else 0

       # Create a mesh of cartesian coordinates.
       self.x_coords = np.arange(
           bounds["west"] + cell_centered_adjustment, bounds["east"], self.hr
       )

       if self.origin == "upper":
           self.y_coords = np.arange(
               bounds["north"] - cell_centered_adjustment, bounds["south"], -self.hr
           )
       elif self.origin == "lower":
           self.y_coords = np.arange(
               bounds["south"] + cell_centered_adjustment, bounds["north"], self.hr
           )
       else:
           raise ValueError("origin must be either 'upper' or 'lower'")

       # How many xy cells do we need to represent the 2D grid?
       self.nx = self.x_coords.shape[0]
       self.ny = self.y_coords.shape[0]
       self.shape = (self.ny, self.nx)
       self.size = self.nx * self.ny

       # Determine how many padding cells are needed to get 10 meters of
       # padding in each direction.
       self.padding_cells = ceil(padding / self.hr)



class Bounds:
   """
   A utility class to store and manipulate geographical bounding boxes.

   The Bounds class represents a geographical bounding box defined by its
   western, southern, eastern, and northern extents. It allows for easy
   creation, manipulation, and conversion of bounding boxes.

   The class supports creating bounding boxes directly through extents,
   or from shapely Polygons and geopandas GeoDataFrames. Moreover, it
   can convert these bounding boxes back into shapely Polygons or
   geopandas GeoDataFrames.

   Attributes
   ----------
   west : float
       The westernmost longitude of the bounding box.
   south : float
       The southernmost latitude of the bounding box.
   east : float
       The easternmost longitude of the bounding box.
   north : float
       The northernmost latitude of the bounding box.
   """

   def __init__(self, west: float, south: float, east: float, north: float):
       """Create a Bounds object from geographic extents.

       Parameters
       ----------
       west : float
           Western extent
       south : float
           Southern extent
       east : float
           Eastern extent
       north : float
           Northern extent
       """
       # Store instance variables
       self.west = west
       self.south = south
       self.east = east
       self.north = north

       # Store the extents as a tuple
       self._extents = (west, south, east, north)

   def __iter__(self):
       """Return an iterator for the object."""
       self._index = 0
       return self

   def __next__(self):
       """Return the next item in the iterator."""
       if self._index < len(self._extents):
           return self._extents[self._index]
       else:
           raise StopIteration

   def pad_to_resolution(self, res) -> Bounds:
       """
       Pad the bounds to a given resolution.
       """
       return Bounds(
           _round_down(self.west, res),
           _round_down(self.south, res),
           _round_up(self.east, res),
           _round_up(self.north, res),
       )

   def pad(self, pad) -> Bounds:
       """
       Pad the bounds by a given amount.
       """
       return Bounds(
           self.west - pad, self.south - pad, self.east + pad, self.north + pad
       )

   @classmethod
   def from_polygon(cls, polygon: Polygon) -> Bounds:
       """Create a Bounds object from a shapely Polygon

       Parameters
       ----------
       polygon : Polygon
           Shapely polygon object.

       Returns
       -------
       Bounds
           Object containing the bounds of the polygon.
       """
       return cls(*polygon.bounds)

   @classmethod
   def from_geodataframe(cls, gdf: GeoDataFrame) -> Bounds:
       """Create a Bounds object from a geopandas GeoDataFrame

       Parameters
       ----------
       gdf : gpd.GeoDataFrame
           Geopandas GeoDataFrame

       Returns
       -------
       Bounds
           Object containing the bounds of the GeoDataFrame.
       """
       return cls(*gdf.total_bounds)

   @classmethod
   def from_geojson(cls, geojson: str) -> Bounds:
       """

       Parameters
       ----------
       geojson

       Returns
       -------

       """
       gdf = GeoDataFrame.from_features(geojson)
       return cls.from_geodataframe(gdf)

   @classmethod
   def from_dict(cls, bounds_dict: dict) -> Bounds:
       """Create a Bounds object from a dictionary

       Parameters
       ----------
       bounds_dict : dict
           Dictionary containing the bounds with keys:
           ["west", "south", "east", "north"].

       Returns
       -------
       Bounds
           Object containing the bounds from the dictionary.
       """
       return cls(
           west=bounds_dict["west"],
           south=bounds_dict["south"],
           east=bounds_dict["east"],
           north=bounds_dict["north"],
       )

   def to_polygon(self) -> Polygon:
       """Convert the bounds to a shapely Polygon

       Returns
       -------
       Polygon
           Shapely polygon
       """
       return Polygon(
           [
               (self.west, self.south),
               (self.east, self.south),
               (self.east, self.north),
               (self.west, self.north),
           ]
       )

   def to_geodataframe(self, crs=None) -> GeoDataFrame:
       """Convert the bounds to a GeoDataFrame

       Parameters
       ----------
       crs: string, optional
           Coordinate reference system to set for the resulting GeoDataFrame.

       Returns
       -------
       GeoDataFrame
           Geopandas GeoDataFrame containing a geometry with the bounding
           box polygon.
       """
       polygon = self.to_polygon()
       gdf = GeoDataFrame(geometry=[polygon])
       if crs:
           gdf.set_crs(crs, inplace=True)

       return gdf

   def to_list(self):
       """Convert the bounds to a list. The List order is:
       [west, south, east, north].

       Returns
       -------
       list
           A list containing the bounds of the object with format:
           [west, south, east, north].
       """
       return [self.west, self.south, self.east, self.north]

   def to_dict(self):
       """Convert the bounds to a dictionary. The dictionary keys are:
       ["west", "south", "east", "north"].

       Returns
       -------
       dict
           A dictionary containing the bounds of the object with keys:
           ["west", "south", "east", "north"].
       """
       return {
           "west": self.west,
           "south": self.south,
           "east": self.east,
           "north": self.north,
       }


def get_geodataframe_from_domain_data(domain_data: dict) -> GeoDataFrame:
   coordinates = domain_data["geometry"]["coordinates"]
   if isinstance(coordinates, str):
       coordinates = json.loads(coordinates)
   domain_data["geometry"]["coordinates"] = coordinates
   domain_crs = domain_data["crs"]["properties"]["name"]
   gdf = GeoDataFrame.from_features([domain_data])
   gdf.crs = domain_crs if domain_crs != "local" else None

   return gdf


def _round_down(num: float, divisor: float) -> float:
   """Rounds a float down to the nearest divisor"""
   return floor(num / divisor) * divisor


def _round_up(num: float, divisor: float) -> float:
   """Rounds a float up to the nearest divisor"""
   return ceil(num / divisor) * divisor