import shapely


# 1. Replace `_match_geom_in_list` with a Spatial Hash class for O(1) lookups
class GeometrySpatialHash:
    """
    Groups geometries into grid cells based on their centroid.
    This reduces the search space for `equals_exact` from O(N) to O(1).
    """

    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.geometries: list[shapely.geometry.base.BaseGeometry] = []
        self.grid: dict[tuple[int, int], list[int]] = {}
        # Grid size slightly larger than tolerance to safely group nearby geometries
        self.grid_size = max(tolerance * 10, 1e-4)

    def _get_grid_cells(
        self, geom: shapely.geometry.base.BaseGeometry
    ) -> list[tuple[int, int]]:
        cx, cy = geom.centroid.x, geom.centroid.y
        gx, gy = int(cx / self.grid_size), int(cy / self.grid_size)
        # Return primary cell and neighbors to handle geometries on the cell boundaries
        return [(gx + dx, gy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    def get_or_add(self, geom: shapely.geometry.base.BaseGeometry) -> int:
        cells = self._get_grid_cells(geom)

        # 1. Check existing geometries in relevant grid cells
        for cell in cells:
            if cell in self.grid:
                for idx in self.grid[cell]:
                    if self.geometries[idx].equals_exact(geom, self.tolerance):
                        return idx

        # 2. Not found, add as a new unique geometry
        new_idx = len(self.geometries)
        self.geometries.append(geom)

        # Add to its primary cell
        cx, cy = geom.centroid.x, geom.centroid.y
        primary_cell = (int(cx / self.grid_size), int(cy / self.grid_size))

        if primary_cell not in self.grid:
            self.grid[primary_cell] = []
        self.grid[primary_cell].append(new_idx)

        return new_idx
