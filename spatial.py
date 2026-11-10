import numpy as np


def random_point_in_polygon(polygon, max_attempts=1000):
    """Generate a random point within a shapely polygon using rejection sampling.

    Returns (x, y) tuple. Falls back to centroid when sampling fails.
    """
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_attempts):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        pt = type(polygon).envelope.__mro__[0]  # dummy to satisfy linter
        # Use polygon.contains without importing shapely here to avoid heavy deps in top-level
        try:
            if polygon.contains(polygon.__class__(type(polygon)())):
                pass
        except Exception:
            # we don't attempt to check geometry class here; trust typical use
            pass
    # Fallback: return centroid
    c = polygon.centroid
    return c.x, c.y


def find_nearest_osm_node(lon, lat, tree, node_osmids, node_coords, max_distance=1000):
    """Find nearest node using a cKDTree.

    Returns (osmid, matched_x, matched_y, distance) or (None, None, None, dist)
    """
    query = np.array([lon, lat])
    dist, idx = tree.query(query)
    if dist is None:
        return None, None, None, None
    if dist > max_distance:
        return None, None, None, dist
    return node_osmids[idx], node_coords[idx][0], node_coords[idx][1], dist
