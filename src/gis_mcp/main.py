"""Main MCP server implementation for GIS operations."""

from typing import Any, Dict, List, Optional
from mcp import MCPServer, Tool
from .tools import shapely_ops, pyproj_ops

class GISMCPServer(MCPServer):
    """GIS MCP Server implementation."""

    def __init__(self):
        """Initialize the GIS MCP server with available tools."""
        super().__init__()
        self.register_tools()

    def register_tools(self):
        """Register all available GIS tools."""
        # Shapely operations
        self.register_tool(Tool(
            name="buffer",
            description="Create a buffer around a geometry",
            parameters={
                "geometry": "WKT string of the input geometry",
                "distance": "Buffer distance in units of the geometry's CRS",
                "resolution": "Number of segments to use for buffer approximation",
                "join_style": "Join style (1=round, 2=mitre, 3=bevel)",
                "mitre_limit": "Mitre limit for join_style=2",
                "single_sided": "Whether to buffer on one side only"
            },
            handler=shapely_ops.buffer
        ))

        self.register_tool(Tool(
            name="intersection",
            description="Find intersection of two geometries",
            parameters={
                "geometry1": "WKT string of the first geometry",
                "geometry2": "WKT string of the second geometry"
            },
            handler=shapely_ops.intersection
        ))

        self.register_tool(Tool(
            name="union",
            description="Combine two geometries",
            parameters={
                "geometry1": "WKT string of the first geometry",
                "geometry2": "WKT string of the second geometry"
            },
            handler=shapely_ops.union
        ))

        self.register_tool(Tool(
            name="difference",
            description="Find difference between geometries",
            parameters={
                "geometry1": "WKT string of the first geometry",
                "geometry2": "WKT string of the second geometry"
            },
            handler=shapely_ops.difference
        ))

        self.register_tool(Tool(
            name="symmetric_difference",
            description="Find symmetric difference between geometries",
            parameters={
                "geometry1": "WKT string of the first geometry",
                "geometry2": "WKT string of the second geometry"
            },
            handler=shapely_ops.symmetric_difference
        ))

        self.register_tool(Tool(
            name="convex_hull",
            description="Calculate convex hull of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.convex_hull
        ))

        self.register_tool(Tool(
            name="envelope",
            description="Get bounding box of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.envelope
        ))

        self.register_tool(Tool(
            name="minimum_rotated_rectangle",
            description="Get minimum rotated rectangle of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.minimum_rotated_rectangle
        ))

        self.register_tool(Tool(
            name="rotate_geometry",
            description="Rotate a geometry",
            parameters={
                "geometry": "WKT string of the input geometry",
                "angle": "Rotation angle in degrees",
                "origin": "Rotation origin point or 'center'",
                "use_radians": "Whether angle is in radians"
            },
            handler=shapely_ops.rotate_geometry
        ))

        self.register_tool(Tool(
            name="scale_geometry",
            description="Scale a geometry",
            parameters={
                "geometry": "WKT string of the input geometry",
                "xfact": "X scale factor",
                "yfact": "Y scale factor",
                "origin": "Scale origin point or 'center'"
            },
            handler=shapely_ops.scale_geometry
        ))

        self.register_tool(Tool(
            name="translate_geometry",
            description="Translate a geometry",
            parameters={
                "geometry": "WKT string of the input geometry",
                "xoff": "X offset",
                "yoff": "Y offset",
                "zoff": "Z offset"
            },
            handler=shapely_ops.translate_geometry
        ))

        self.register_tool(Tool(
            name="triangulate_geometry",
            description="Create a triangulation of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.triangulate_geometry
        ))

        self.register_tool(Tool(
            name="voronoi",
            description="Create a Voronoi diagram from points",
            parameters={
                "geometry": "WKT string of the input points"
            },
            handler=shapely_ops.voronoi
        ))

        self.register_tool(Tool(
            name="unary_union_geometries",
            description="Create a union of multiple geometries",
            parameters={
                "geometries": "List of WKT strings"
            },
            handler=shapely_ops.unary_union_geometries
        ))

        self.register_tool(Tool(
            name="get_centroid",
            description="Get the centroid of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_centroid
        ))

        self.register_tool(Tool(
            name="get_length",
            description="Get the length of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_length
        ))

        self.register_tool(Tool(
            name="get_area",
            description="Get the area of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_area
        ))

        self.register_tool(Tool(
            name="get_bounds",
            description="Get the bounds of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_bounds
        ))

        self.register_tool(Tool(
            name="get_coordinates",
            description="Get the coordinates of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_coordinates
        ))

        self.register_tool(Tool(
            name="get_geometry_type",
            description="Get the type of a geometry",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.get_geometry_type
        ))

        self.register_tool(Tool(
            name="is_valid",
            description="Check if a geometry is valid",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.is_valid
        ))

        self.register_tool(Tool(
            name="make_valid",
            description="Make a geometry valid",
            parameters={
                "geometry": "WKT string of the input geometry"
            },
            handler=shapely_ops.make_valid
        ))

        self.register_tool(Tool(
            name="simplify",
            description="Simplify a geometry",
            parameters={
                "geometry": "WKT string of the input geometry",
                "tolerance": "Tolerance for simplification",
                "preserve_topology": "Whether to preserve topology"
            },
            handler=shapely_ops.simplify
        ))

        # PyProj operations
        self.register_tool(Tool(
            name="transform_coordinates",
            description="Transform coordinates between CRS",
            parameters={
                "coordinates": "List of [x, y] coordinates",
                "source_crs": "Source CRS (e.g., 'EPSG:4326')",
                "target_crs": "Target CRS (e.g., 'EPSG:3857')"
            },
            handler=pyproj_ops.transform_coordinates
        ))

        self.register_tool(Tool(
            name="project_geometry",
            description="Project a geometry between CRS",
            parameters={
                "geometry": "WKT string of the input geometry",
                "source_crs": "Source CRS (e.g., 'EPSG:4326')",
                "target_crs": "Target CRS (e.g., 'EPSG:3857')"
            },
            handler=pyproj_ops.project_geometry
        ))

        self.register_tool(Tool(
            name="get_crs_info",
            description="Get information about a CRS",
            parameters={
                "crs": "CRS identifier (e.g., 'EPSG:4326')"
            },
            handler=pyproj_ops.get_crs_info
        ))

        self.register_tool(Tool(
            name="get_available_crs",
            description="Get list of available CRS",
            parameters={},
            handler=pyproj_ops.get_available_crs
        ))

        self.register_tool(Tool(
            name="get_geod_info",
            description="Get information about a geodetic calculation",
            parameters={
                "ellps": "Ellipsoid name (default: 'WGS84')",
                "a": "Semi-major axis",
                "b": "Semi-minor axis",
                "f": "Flattening"
            },
            handler=pyproj_ops.get_geod_info
        ))

        self.register_tool(Tool(
            name="calculate_geodetic_distance",
            description="Calculate geodetic distance between points",
            parameters={
                "point1": "List of [lon, lat] coordinates",
                "point2": "List of [lon, lat] coordinates",
                "ellps": "Ellipsoid name (default: 'WGS84')"
            },
            handler=pyproj_ops.calculate_geodetic_distance
        ))

        self.register_tool(Tool(
            name="calculate_geodetic_point",
            description="Calculate point at given distance and azimuth",
            parameters={
                "start_point": "List of [lon, lat] coordinates",
                "azimuth": "Forward azimuth in degrees",
                "distance": "Distance in meters",
                "ellps": "Ellipsoid name (default: 'WGS84')"
            },
            handler=pyproj_ops.calculate_geodetic_point
        ))

        self.register_tool(Tool(
            name="calculate_geodetic_area",
            description="Calculate area of a polygon using geodetic calculations",
            parameters={
                "geometry": "WKT string of the input polygon",
                "ellps": "Ellipsoid name (default: 'WGS84')"
            },
            handler=pyproj_ops.calculate_geodetic_area
        ))

        self.register_tool(Tool(
            name="get_utm_zone",
            description="Get UTM zone for given coordinates",
            parameters={
                "coordinates": "List of [lon, lat] coordinates"
            },
            handler=pyproj_ops.get_utm_zone
        ))

        self.register_tool(Tool(
            name="get_utm_crs",
            description="Get UTM CRS for given coordinates",
            parameters={
                "coordinates": "List of [lon, lat] coordinates"
            },
            handler=pyproj_ops.get_utm_crs
        ))

        self.register_tool(Tool(
            name="get_geocentric_crs",
            description="Get geocentric CRS for given coordinates",
            parameters={
                "coordinates": "List of [lon, lat] coordinates"
            },
            handler=pyproj_ops.get_geocentric_crs
        ))

def main():
    """Run the GIS MCP server."""
    server = GISMCPServer()
    server.run()

if __name__ == "__main__":
    main() 