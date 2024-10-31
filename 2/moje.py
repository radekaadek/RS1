import geopandas
import rasterio
import numpy as np
from shapely.geometry import mapping
from rasterio.mask import mask


# load raster

raster_file = "/home/ard/prg/RS1/1/R_2023/N-34-111-A-a-1-1.tif"
raster_r = "/home/ard/prg/RS1/1/CIR_2023/78935_1207374_N-34-123-A-a-1-1.tif"

with rasterio.open(raster_file) as src:
    # vector_file = "features.fgb"

    profile = src.profile

    # gdf = geopandas.read_file(vector_file)
    # feature = gdf.iloc[0]

    band1 = src.read(1)

with rasterio.open(raster_r) as src:
    bandr = src.read(1)
    bandg = src.read(2)
    bandb = src.read(3)

#conver to floats
band1 = band1.astype(np.float64).copy()
print(band1)
bandr = bandr.astype(np.float64).copy()
bandg = bandg.astype(np.float64).copy()
bandb = bandb.astype(np.float64).copy()

nvdil = np.array(band1 - bandr)
nvdim = np.array(band1 + bandr)
print(nvdil, nvdim)

nvdim[(nvdim == 0) | np.isnan(nvdim)] = 0.001
print(len(nvdim == 0))
nvdi = nvdil / nvdim

# scale to 0-255 integers
nvdi = np.multiply(nvdi, 255)
nvdi = np.array(nvdi, dtype=np.uint8)

with rasterio.open("ok.tif", "w", **profile) as save:
    print(nvdi)
    save.write(nvdi, 1)





# cut the raster to the gdf bbox
