""" GIS MCP Server - Main entry point

This module implements an MCP server that connects LLMs to GIS operations using
Shapely and PyProj libraries, enabling AI assistants to perform geospatial operations
and transformations.
"""

import json
import logging
import os
import sys
import argparse
import tempfile
from typing import Any, Dict, List, Optional, Union

# MCP imports using the new SDK patterns
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gis-mcp")

# Create FastMCP instance
mcp = FastMCP("GIS MCP")

# Resource handlers
@mcp.resource("gis://operations/basic")
def get_basic_operations() -> Dict[str, List[str]]:
    """List available basic geometric operations."""
    return {
        "operations": [
            "buffer",
            "intersection",
            "union",
            "difference",
            "symmetric_difference"
        ]
    }

@mcp.resource("gis://operations/geometric")
def get_geometric_properties() -> Dict[str, List[str]]:
    """List available geometric property operations."""
    return {
        "operations": [
            "convex_hull",
            "envelope",
            "minimum_rotated_rectangle",
            "get_centroid",
            "get_bounds",
            "get_coordinates",
            "get_geometry_type"
        ]
    }

@mcp.resource("gis://operations/transformations")
def get_transformations() -> Dict[str, List[str]]:
    """List available geometric transformations."""
    return {
        "operations": [
            "rotate_geometry",
            "scale_geometry",
            "translate_geometry"
        ]
    }

@mcp.resource("gis://operations/advanced")
def get_advanced_operations() -> Dict[str, List[str]]:
    """List available advanced operations."""
    return {
        "operations": [
            "triangulate_geometry",
            "voronoi",
            "unary_union_geometries"
        ]
    }

@mcp.resource("gis://operations/measurements")
def get_measurements() -> Dict[str, List[str]]:
    """List available measurement operations."""
    return {
        "operations": [
            "get_length",
            "get_area"
        ]
    }

@mcp.resource("gis://operations/validation")
def get_validation_operations() -> Dict[str, List[str]]:
    """List available validation operations."""
    return {
        "operations": [
            "is_valid",
            "make_valid",
            "simplify"
        ]
    }

@mcp.resource("gis://crs/transformations")
def get_crs_transformations() -> Dict[str, List[str]]:
    """List available CRS transformation operations."""
    return {
        "operations": [
            "transform_coordinates",
            "project_geometry"
        ]
    }

@mcp.resource("gis://crs/info")
def get_crs_info_operations() -> Dict[str, List[str]]:
    """List available CRS information operations."""
    return {
        "operations": [
            "get_crs_info",
            "get_available_crs",
            "get_utm_zone",
            "get_utm_crs",
            "get_geocentric_crs"
        ]
    }

@mcp.resource("gis://crs/geodetic")
def get_geodetic_operations() -> Dict[str, List[str]]:
    """List available geodetic calculation operations."""
    return {
        "operations": [
            "get_geod_info",
            "calculate_geodetic_distance",
            "calculate_geodetic_point",
            "calculate_geodetic_area"
        ]
    }

@mcp.resource("gis://operation/rasterio")
def get_rasterio_operations() -> Dict[str, List[str]]:
    """List available rasterio operations."""
    return {
        "operations": [
            "metadata_raster",
            "get_raster_crs",
            "clip_raster_with_shapefile",
            "resample_raster"
        ]
    }

# Tool implementations
@mcp.tool()
def buffer(geometry: str, distance: float, resolution: int = 16, 
        join_style: int = 1, mitre_limit: float = 5.0, 
        single_sided: bool = False) -> Dict[str, Any]:
    """Create a buffer around a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        buffered = geom.buffer(
            distance=distance,
            resolution=resolution,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided
        )
        return {
            "status": "success",
            "geometry": buffered.wkt,
            "message": "Buffer created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating buffer: {str(e)}")
        raise ValueError(f"Failed to create buffer: {str(e)}")

@mcp.tool()
def intersection(geometry1: str, geometry2: str) -> Dict[str, Any]:
    """Find intersection of two geometries."""
    try:
        from shapely import wkt
        geom1 = wkt.loads(geometry1)
        geom2 = wkt.loads(geometry2)
        result = geom1.intersection(geom2)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Intersection created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating intersection: {str(e)}")
        raise ValueError(f"Failed to create intersection: {str(e)}")

@mcp.tool()
def union(geometry1: str, geometry2: str) -> Dict[str, Any]:
    """Combine two geometries."""
    try:
        from shapely import wkt
        geom1 = wkt.loads(geometry1)
        geom2 = wkt.loads(geometry2)
        result = geom1.union(geom2)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Union created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating union: {str(e)}")
        raise ValueError(f"Failed to create union: {str(e)}")

@mcp.tool()
def difference(geometry1: str, geometry2: str) -> Dict[str, Any]:
    """Find difference between geometries."""
    try:
        from shapely import wkt
        geom1 = wkt.loads(geometry1)
        geom2 = wkt.loads(geometry2)
        result = geom1.difference(geom2)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Difference created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating difference: {str(e)}")
        raise ValueError(f"Failed to create difference: {str(e)}")

@mcp.tool()
def symmetric_difference(geometry1: str, geometry2: str) -> Dict[str, Any]:
    """Find symmetric difference between geometries."""
    try:
        from shapely import wkt
        geom1 = wkt.loads(geometry1)
        geom2 = wkt.loads(geometry2)
        result = geom1.symmetric_difference(geom2)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Symmetric difference created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating symmetric difference: {str(e)}")
        raise ValueError(f"Failed to create symmetric difference: {str(e)}")

@mcp.tool()
def convex_hull(geometry: str) -> Dict[str, Any]:
    """Calculate convex hull of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.convex_hull
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Convex hull created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating convex hull: {str(e)}")
        raise ValueError(f"Failed to create convex hull: {str(e)}")

@mcp.tool()
def envelope(geometry: str) -> Dict[str, Any]:
    """Get bounding box of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.envelope
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Envelope created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating envelope: {str(e)}")
        raise ValueError(f"Failed to create envelope: {str(e)}")

@mcp.tool()
def minimum_rotated_rectangle(geometry: str) -> Dict[str, Any]:
    """Get minimum rotated rectangle of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.minimum_rotated_rectangle
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Minimum rotated rectangle created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating minimum rotated rectangle: {str(e)}")
        raise ValueError(f"Failed to create minimum rotated rectangle: {str(e)}")

@mcp.tool()
def rotate_geometry(geometry: str, angle: float, origin: str = "center", 
                use_radians: bool = False) -> Dict[str, Any]:
    """Rotate a geometry."""
    try:
        from shapely import wkt
        from shapely.affinity import rotate
        geom = wkt.loads(geometry)
        result = rotate(geom, angle=angle, origin=origin, use_radians=use_radians)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Geometry rotated successfully"
        }
    except Exception as e:
        logger.error(f"Error rotating geometry: {str(e)}")
        raise ValueError(f"Failed to rotate geometry: {str(e)}")

@mcp.tool()
def scale_geometry(geometry: str, xfact: float, yfact: float, 
                origin: str = "center") -> Dict[str, Any]:
    """Scale a geometry."""
    try:
        from shapely import wkt
        from shapely.affinity import scale
        geom = wkt.loads(geometry)
        result = scale(geom, xfact=xfact, yfact=yfact, origin=origin)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Geometry scaled successfully"
        }
    except Exception as e:
        logger.error(f"Error scaling geometry: {str(e)}")
        raise ValueError(f"Failed to scale geometry: {str(e)}")

@mcp.tool()
def translate_geometry(geometry: str, xoff: float, yoff: float, 
                    zoff: float = 0.0) -> Dict[str, Any]:
    """Translate a geometry."""
    try:
        from shapely import wkt
        from shapely.affinity import translate
        geom = wkt.loads(geometry)
        result = translate(geom, xoff=xoff, yoff=yoff, zoff=zoff)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Geometry translated successfully"
        }
    except Exception as e:
        logger.error(f"Error translating geometry: {str(e)}")
        raise ValueError(f"Failed to translate geometry: {str(e)}")

@mcp.tool()
def triangulate_geometry(geometry: str) -> Dict[str, Any]:
    """Create a triangulation of a geometry."""
    try:
        from shapely import wkt
        from shapely.ops import triangulate
        geom = wkt.loads(geometry)
        triangles = triangulate(geom)
        return {
            "status": "success",
            "geometries": [tri.wkt for tri in triangles],
            "message": "Triangulation created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating triangulation: {str(e)}")
        raise ValueError(f"Failed to create triangulation: {str(e)}")

@mcp.tool()
def voronoi(geometry: str) -> Dict[str, Any]:
    """Create a Voronoi diagram from points."""
    try:
        from shapely import wkt
        from shapely.ops import voronoi_diagram
        geom = wkt.loads(geometry)
        result = voronoi_diagram(geom)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Voronoi diagram created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating Voronoi diagram: {str(e)}")
        raise ValueError(f"Failed to create Voronoi diagram: {str(e)}")

@mcp.tool()
def unary_union_geometries(geometries: List[str]) -> Dict[str, Any]:
    """Create a union of multiple geometries."""
    try:
        from shapely import wkt
        from shapely.ops import unary_union
        geoms = [wkt.loads(g) for g in geometries]
        result = unary_union(geoms)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Union created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating union: {str(e)}")
        raise ValueError(f"Failed to create union: {str(e)}")

@mcp.tool()
def get_centroid(geometry: str) -> Dict[str, Any]:
    """Get the centroid of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.centroid
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Centroid calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating centroid: {str(e)}")
        raise ValueError(f"Failed to calculate centroid: {str(e)}")

@mcp.tool()
def get_length(geometry: str) -> Dict[str, Any]:
    """Get the length of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "length": float(geom.length),
            "message": "Length calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating length: {str(e)}")
        raise ValueError(f"Failed to calculate length: {str(e)}")

@mcp.tool()
def get_area(geometry: str) -> Dict[str, Any]:
    """Get the area of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "area": float(geom.area),
            "message": "Area calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating area: {str(e)}")
        raise ValueError(f"Failed to calculate area: {str(e)}")

@mcp.tool()
def get_bounds(geometry: str) -> Dict[str, Any]:
    """Get the bounds of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "bounds": list(geom.bounds),
            "message": "Bounds calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating bounds: {str(e)}")
        raise ValueError(f"Failed to calculate bounds: {str(e)}")

@mcp.tool()
def get_coordinates(geometry: str) -> Dict[str, Any]:
    """Get the coordinates of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "coordinates": [list(coord) for coord in geom.coords],
            "message": "Coordinates retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting coordinates: {str(e)}")
        raise ValueError(f"Failed to get coordinates: {str(e)}")

@mcp.tool()
def get_geometry_type(geometry: str) -> Dict[str, Any]:
    """Get the type of a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "type": geom.geom_type,
            "message": "Geometry type retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting geometry type: {str(e)}")
        raise ValueError(f"Failed to get geometry type: {str(e)}")

@mcp.tool()
def is_valid(geometry: str) -> Dict[str, Any]:
    """Check if a geometry is valid."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        return {
            "status": "success",
            "is_valid": bool(geom.is_valid),
            "message": "Geometry validation completed successfully"
        }
    except Exception as e:
        logger.error(f"Error validating geometry: {str(e)}")
        raise ValueError(f"Failed to validate geometry: {str(e)}")

@mcp.tool()
def make_valid(geometry: str) -> Dict[str, Any]:
    """Make a geometry valid."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.make_valid()
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Geometry made valid successfully"
        }
    except Exception as e:
        logger.error(f"Error making geometry valid: {str(e)}")
        raise ValueError(f"Failed to make geometry valid: {str(e)}")

@mcp.tool()
def simplify(geometry: str, tolerance: float, 
            preserve_topology: bool = True) -> Dict[str, Any]:
    """Simplify a geometry."""
    try:
        from shapely import wkt
        geom = wkt.loads(geometry)
        result = geom.simplify(tolerance=tolerance, preserve_topology=preserve_topology)
        return {
            "status": "success",
            "geometry": result.wkt,
            "message": "Geometry simplified successfully"
        }
    except Exception as e:
        logger.error(f"Error simplifying geometry: {str(e)}")
        raise ValueError(f"Failed to simplify geometry: {str(e)}")

@mcp.tool()
def transform_coordinates(coordinates: List[float], source_crs: str, 
                        target_crs: str) -> Dict[str, Any]:
    """Transform coordinates between CRS."""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        x, y = coordinates
        x_transformed, y_transformed = transformer.transform(x, y)
        return {
            "status": "success",
            "coordinates": [x_transformed, y_transformed],
            "source_crs": source_crs,
            "target_crs": target_crs,
            "message": "Coordinates transformed successfully"
        }
    except Exception as e:
        logger.error(f"Error transforming coordinates: {str(e)}")
        raise ValueError(f"Failed to transform coordinates: {str(e)}")

@mcp.tool()
def project_geometry(geometry: str, source_crs: str, 
                    target_crs: str) -> Dict[str, Any]:
    """Project a geometry between CRS."""
    try:
        from shapely import wkt
        from shapely.ops import transform
        from pyproj import Transformer
        geom = wkt.loads(geometry)
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        projected = transform(transformer.transform, geom)
        return {
            "status": "success",
            "geometry": projected.wkt,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "message": "Geometry projected successfully"
        }
    except Exception as e:
        logger.error(f"Error projecting geometry: {str(e)}")
        raise ValueError(f"Failed to project geometry: {str(e)}")

@mcp.tool()
def get_crs_info(crs: str) -> Dict[str, Any]:
    """Get information about a CRS."""
    try:
        import pyproj
        crs_obj = pyproj.CRS(crs)
        return {
            "status": "success",
            "name": crs_obj.name,
            "type": crs_obj.type_name,
            "axis_info": [axis.direction for axis in crs_obj.axis_info],
            "is_geographic": crs_obj.is_geographic,
            "is_projected": crs_obj.is_projected,
            "datum": str(crs_obj.datum),
            "ellipsoid": str(crs_obj.ellipsoid),
            "prime_meridian": str(crs_obj.prime_meridian),
            "area_of_use": str(crs_obj.area_of_use) if crs_obj.area_of_use else None,
            "message": "CRS information retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting CRS info: {str(e)}")
        raise ValueError(f"Failed to get CRS info: {str(e)}")

@mcp.tool()
def get_available_crs() -> Dict[str, Any]:
    """Get list of available CRS."""
    try:
        import pyproj
        crs_list = []
        for crs in pyproj.database.get_crs_list():
            try:
                crs_info = get_crs_info({"crs": crs})
                crs_list.append({
                    "auth_name": crs.auth_name,
                    "code": crs.code,
                    "name": crs_info["name"],
                    "type": crs_info["type"]
                })
            except:
                continue
        return {
            "status": "success",
            "crs_list": crs_list,
            "message": "Available CRS list retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting available CRS: {str(e)}")
        raise ValueError(f"Failed to get available CRS: {str(e)}")

@mcp.tool()
def get_geod_info(ellps: str = "WGS84", a: Optional[float] = None,
                b: Optional[float] = None, f: Optional[float] = None) -> Dict[str, Any]:
    """Get information about a geodetic calculation."""
    try:
        import pyproj
        geod = pyproj.Geod(ellps=ellps, a=a, b=b, f=f)
        return {
            "status": "success",
            "ellps": geod.ellps,
            "a": geod.a,
            "b": geod.b,
            "f": geod.f,
            "es": geod.es,
            "e": geod.e,
            "message": "Geodetic information retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting geodetic info: {str(e)}")
        raise ValueError(f"Failed to get geodetic info: {str(e)}")

@mcp.tool()
def calculate_geodetic_distance(point1: List[float], point2: List[float], 
                            ellps: str = "WGS84") -> Dict[str, Any]:
    """Calculate geodetic distance between points."""
    try:
        import pyproj
        geod = pyproj.Geod(ellps=ellps)
        lon1, lat1 = point1
        lon2, lat2 = point2
        forward_azimuth, back_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)
        return {
            "status": "success",
            "distance": distance,
            "forward_azimuth": forward_azimuth,
            "back_azimuth": back_azimuth,
            "ellps": ellps,
            "message": "Geodetic distance calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating geodetic distance: {str(e)}")
        raise ValueError(f"Failed to calculate geodetic distance: {str(e)}")

@mcp.tool()
def calculate_geodetic_point(start_point: List[float], azimuth: float, 
                        distance: float, ellps: str = "WGS84") -> Dict[str, Any]:
    """Calculate point at given distance and azimuth."""
    try:
        import pyproj
        geod = pyproj.Geod(ellps=ellps)
        lon, lat = start_point
        lon2, lat2, back_azimuth = geod.fwd(lon, lat, azimuth, distance)
        return {
            "status": "success",
            "point": [lon2, lat2],
            "back_azimuth": back_azimuth,
            "ellps": ellps,
            "message": "Geodetic point calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating geodetic point: {str(e)}")
        raise ValueError(f"Failed to calculate geodetic point: {str(e)}")

@mcp.tool()
def calculate_geodetic_area(geometry: str, ellps: str = "WGS84") -> Dict[str, Any]:
    """Calculate area of a polygon using geodetic calculations."""
    try:
        import pyproj
        from shapely import wkt
        geod = pyproj.Geod(ellps=ellps)
        polygon = wkt.loads(geometry)
        area = abs(geod.geometry_area_perimeter(polygon)[0])
        return {
            "status": "success",
            "area": float(area),
            "ellps": ellps,
            "message": "Geodetic area calculated successfully"
        }
    except Exception as e:
        logger.error(f"Error calculating geodetic area: {str(e)}")
        raise ValueError(f"Failed to calculate geodetic area: {str(e)}")

@mcp.tool()
def get_utm_zone(coordinates: List[float]) -> Dict[str, Any]:
    """Get UTM zone for given coordinates."""
    try:
        import pyproj
        lon, lat = coordinates
        zone = pyproj.database.query_utm_crs_info(
            datum_name="WGS84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=lon,
                south_lat_degree=lat,
                east_lon_degree=lon,
                north_lat_degree=lat
            )
        )[0].to_authority()[1]
        return {
            "status": "success",
            "zone": zone,
            "message": "UTM zone retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting UTM zone: {str(e)}")
        raise ValueError(f"Failed to get UTM zone: {str(e)}")

@mcp.tool()
def get_utm_crs(coordinates: List[float]) -> Dict[str, Any]:
    """Get UTM CRS for given coordinates."""
    try:
        import pyproj
        lon, lat = coordinates
        crs = pyproj.database.query_utm_crs_info(
            datum_name="WGS84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=lon,
                south_lat_degree=lat,
                east_lon_degree=lon,
                north_lat_degree=lat
            )
        )[0].to_wkt()
        return {
            "status": "success",
            "crs": crs,
            "message": "UTM CRS retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting UTM CRS: {str(e)}")
        raise ValueError(f"Failed to get UTM CRS: {str(e)}")

@mcp.tool()
def get_geocentric_crs(coordinates: List[float]) -> Dict[str, Any]:
    """Get geocentric CRS for given coordinates."""
    try:
        import pyproj
        lon, lat = coordinates
        crs = pyproj.database.query_geocentric_crs_info(
            datum_name="WGS84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=lon,
                south_lat_degree=lat,
                east_lon_degree=lon,
                north_lat_degree=lat
            )
        )[0].to_wkt()
        return {
            "status": "success",
            "crs": crs,
            "message": "Geocentric CRS retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting geocentric CRS: {str(e)}")
        raise ValueError(f"Failed to get geocentric CRS: {str(e)}")

@mcp.tool()
def metadata_raster(path_or_url: str) -> Dict[str, Any]:
    """
    Open a raster dataset in read-only mode and return metadata.
    
    This tool supports two modes based on the provided string:
    1. A local filesystem path (e.g., "D:\\Data\\my_raster.tif").
    2. An HTTPS URL (e.g., "https://example.com/my_raster.tif").
    
    The input must be a single string that is either a valid file path
    on the local machine or a valid HTTPS URL pointing to a raster.
    """
    try:
        # Import numpy first to ensure NumPy's C-API is initialized
        import numpy as np
        import rasterio

        # Remove any backticks (`) if the client wrapped the path_or_url in them
        cleaned = path_or_url.replace("`", "")

        # Determine if the string is an HTTPS URL or a local file path
        if cleaned.lower().startswith("https://"):
            # For HTTPS URLs, let Rasterio/GDAL handle remote access directly
            dataset = rasterio.open(cleaned)
        else:
            # Treat as local filesystem path
            local_path = os.path.expanduser(cleaned)

            # Verify that the file exists on disk
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f"Raster file not found at '{local_path}'.")

            # Open the local file in read-only mode
            dataset = rasterio.open(local_path)

        # Build a mapping from band index to its data type (dtype)
        band_dtypes = {i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}

        # Collect core metadata fields in simple Python types
        meta: Dict[str, Any] = {
            "name": dataset.name,                                       # Full URI or filesystem path
            "mode": dataset.mode,                                       # Mode should be 'r' for read                                 
            "driver": dataset.driver,                                   # GDAL driver, e.g. "GTiff"
            "width": dataset.width,                                     # Number of columns
            "height": dataset.height,                                   # Number of rows
            "count": dataset.count,                                     # Number of bands
            "bounds": dataset.bounds,                                   # Show bounding box
            "band_dtypes": band_dtypes,                                 # { band_index: dtype_string }
            "no_data": dataset.nodatavals,                              # Number of NoData values in each band
            "crs": dataset.crs.to_string() if dataset.crs else None,    # CRS as EPSG string or None
            "transform": list(dataset.transform),                       # Affine transform coefficients (6 floats)
        }

        # Return a success status along with metadata
        return {
            "status": "success",
            "metadata": meta,
            "message": f"Raster dataset opened successfully from '{cleaned}'."
        }

    except Exception as e:
        # Log the error for debugging purposes, then raise ValueError so MCP can relay it
        logger.error(f"Error opening raster '{path_or_url}': {str(e)}")
        raise ValueError(f"Failed to open raster '{path_or_url}': {str(e)}")

@mcp.tool()
def get_raster_crs(path_or_url: str) -> Dict[str, Any]:
    """
    Retrieve the Coordinate Reference System (CRS) of a raster dataset.
    
    Opens the raster (local path or HTTPS URL), reads its DatasetReader.crs
    attribute as a PROJ.4-style dict, and also returns the WKT representation.
    """
    try:
        import numpy as np
        import rasterio

        # Strip backticks if the client wrapped the input in them
        cleaned = path_or_url.replace("`", "")

        # Open remote or local dataset
        if cleaned.lower().startswith("https://"):
            src = rasterio.open(cleaned)
        else:
            local_path = os.path.expanduser(cleaned)
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f"Raster file not found at '{local_path}'.")
            src = rasterio.open(local_path)

        # Access the CRS object on the opened dataset
        crs_obj = src.crs
        src.close()

        if crs_obj is None:
            raise ValueError("No CRS defined for this dataset.")

        # Convert CRS to PROJ.4‐style dict and WKT string
        proj4_dict = crs_obj.to_dict()    # e.g., {'init': 'epsg:32618'}
        wkt_str    = crs_obj.to_wkt()     # full WKT representation

        return {
            "status":      "success",
            "proj4":       proj4_dict,
            "wkt":         wkt_str,
            "message":     "CRS retrieved successfully"
        }

    except Exception as e:
        # Log and re-raise as ValueError for MCP error propagation
        logger.error(f"Error retrieving CRS for '{path_or_url}': {e}")
        raise ValueError(f"Failed to retrieve CRS: {e}")

@mcp.tool()
def clip_raster_with_shapefile(
    raster_path_or_url: str,
    shapefile_path: str,
    destination: str
) -> Dict[str, Any]:
    """
    Clip a raster dataset using polygons from a shapefile and write the result.
    Converts the shapefile's CRS to match the raster's CRS if they are different.
    
    Parameters:
    - raster_path_or_url: local path or HTTPS URL of the source raster.
    - shapefile_path:     local filesystem path to a .shp file containing polygons.
    - destination:        local path where the masked raster will be written.
    """
    try:
        import numpy as np
        import rasterio
        import rasterio.mask
        from rasterio.warp import transform_geom
        import pyproj
        import fiona

        # Clean paths
        raster_clean = raster_path_or_url.replace("`", "")
        shp_clean = shapefile_path.replace("`", "")
        dst_clean = destination.replace("`", "")

        # Verify shapefile exists
        shp_path = os.path.expanduser(shp_clean)
        if not os.path.isfile(shp_path):
            raise FileNotFoundError(f"Shapefile not found at '{shp_path}'.")

        # Open the raster
        if raster_clean.lower().startswith("https://"):
            src = rasterio.open(raster_clean)
        else:
            src_path = os.path.expanduser(raster_clean)
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"Raster not found at '{src_path}'.")
            src = rasterio.open(src_path)

        raster_crs = src.crs  # Get raster CRS

        # Read geometries from shapefile and check CRS
        with fiona.open(shp_path, "r") as shp:
            shapefile_crs = pyproj.CRS(shp.crs)  # Get shapefile CRS
            shapes: List[Dict[str, Any]] = [feat["geometry"] for feat in shp]

            # Convert geometries to raster CRS if necessary
            if shapefile_crs != raster_crs:
                shapes = [transform_geom(str(shapefile_crs), str(raster_crs), shape) for shape in shapes]

        # Apply mask: crop to shapes and set outside pixels to zero
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        src.close()

        # Update metadata for the masked output
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Ensure destination directory exists
        dst_path = os.path.expanduser(dst_clean)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

        # Write the masked raster
        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)

        return {
            "status": "success",
            "destination": dst_path,
            "message": f"Raster masked and saved to '{dst_path}'."
        }

    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"Failed to mask raster: {e}")

def main():
    """Main entry point for the GIS MCP server."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GIS MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Start the MCP server
        print("Starting GIS MCP server...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 