"""
Script to generate the embeddings.parquet geoparquet file
"""
import io

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point

# 1. Create the example data & GeoDataFrame
data = {
    "ID": [1, 2, 3, 4],
    "embedding": [
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),
        np.array([2.1, 2.2, 2.3, 2.4], dtype=np.float32),
        np.array([3.1, 3.2, 3.3, 3.4], dtype=np.float32),
    ],
    "geometry": [
        Point(405620, 5757180),
        Point(405780, 5757120),
        Point(404800, 5756300),
        Point(404750, 5757400),
    ],
}

gdf = gpd.GeoDataFrame(data, crs="EPSG:32632")
# Convert numpy arrays to lists
gdf["embedding"] = gdf["embedding"].apply(lambda x: x.tolist())

# 2. Write to an in-memory buffer to lock in the GeoPandas metadata
buf = io.BytesIO()
gdf.to_parquet(buf)
buf.seek(0)

# 3. Read it back as a strict PyArrow Table
table = pq.read_table(buf)

# 4. Define the FixedSizeList type
fixed_size_type = pa.list_(pa.float32(), 4)

# 5. Extract, cast, and replace the embedding column
col_idx = table.schema.get_field_index("embedding")
new_field = pa.field("embedding", fixed_size_type)

casted_column = table.column("embedding").cast(fixed_size_type)
table = table.set_column(col_idx, new_field, casted_column)

# 6. Write the final PyArrow table to disk
pq.write_table(table, "tests/data/embeddings.parquet")
