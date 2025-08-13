### Shapely Tools

High-level geometric operations exposed via MCP. Core documented tools:

- [buffer](buffer.md)
- [intersection](intersection.md)
- [union](union.md)
- [difference](difference.md)
- [centroid](centroid.md) (tool name: `get_centroid`)

Additional available tools in the server not listed in the sidebar:

- symmetric_difference, convex_hull, envelope, minimum_rotated_rectangle
- get_bounds, get_coordinates, get_geometry_type
- rotate_geometry, scale_geometry, translate_geometry
- triangulate_geometry, voronoi, unary_union_geometries
- get_length, get_area
- is_valid, make_valid, simplify

All tools accept and return geometries as WKT strings.
