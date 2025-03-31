import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import scale
from rasterio.features import rasterize, shapes

from dataclasses import dataclass
from abc import ABC, abstractmethod

from .topology import Topology

class Parameterization(ABC) :
    domain: Polygon # bounding box
    dimension: int

    @abstractmethod
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon : ...

    @staticmethod
    def compute_geometry_from_image(mask: np.ndarray) -> MultiPolygon :
        polygons = [
            (Polygon(poly['coordinates'][0]), bool(is_positive)) for (poly, is_positive) in 
            shapes(mask.astype(np.uint8))
        ]
        geo: MultiPolygon = MultiPolygon((poly for (poly, is_positive) in polygons if is_positive))
        for (poly, is_positive) in polygons:
            if not(is_positive) and geo.contains(poly) : geo = geo.difference(poly)
        return geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo])
    
    @staticmethod
    def rasterize_geometry(topology: Topology) -> np.ndarray:
        return rasterize(
            shapes=scale(topology.geometry, xfact=topology.density, yfact=topology.density, origin=(0,0)).geoms, 
            out_shape=(int(topology.density*topology.domain_size_y), int(topology.density*topology.domain_size_x))
        )

    def update_topology(self, topology: Topology, x: np.ndarray) -> None :
        geo = self.compute_geometry(x).intersection(topology.domain)
        topology.geometry = geo if isinstance(geo, MultiPolygon) else MultiPolygon([geo])
        topology.mask = Parameterization.rasterize_geometry(topology).astype(bool)

        if not(topology.continuous) : 
            topology.geometry = Parameterization.compute_geometry_from_image(topology.mask)