import rasterio
import numpy as np

# load raster

raster_file = "/home/ard/prg/RS1/1/IR_2023/N-34-111-C-c-3-3.tif"
raster_r = "/home/ard/prg/RS1/1/RGB_2023/78936_1204849_N-34-111-C-c-3-3.tif"

with rasterio.open(raster_file) as src:
    profile = src.profile
    band1 = src.read(1)

with rasterio.open(raster_r) as src:
    bandr = src.read(1)
    bandg = src.read(2)
    bandb = src.read(3)

#conver to floats
band1 = band1.astype(np.float32).copy()
bandr = bandr.astype(np.float32).copy()
bandg = bandg.astype(np.float32).copy()
bandb = bandb.astype(np.float32).copy()

nvdil = np.array(band1 - bandr)
nvdim = np.array(band1 + bandr)

nvdi = nvdil / nvdim
gdvi = band1 - bandg

nvdi[(nvdim == 0) | np.isnan(nvdim)] = 0

# scale to 0-1 floats 
nvdi = (nvdi - nvdi.min()) / (nvdi.max() - nvdi.min())
gdvi = (gdvi - gdvi.min()) / (gdvi.max() - gdvi.min())
nvdi[(nvdim == 0) | np.isnan(nvdim)] = 0

nvdi = np.multiply(nvdi, 255)
nvdi = np.round(nvdi).astype(np.uint8)
gdvi = np.multiply(gdvi, 255)
nvdi = np.round(nvdi).astype(np.uint8)


profile.update(
    dtype=rasterio.uint8
)
with rasterio.open("NDVI.tif", "w", **profile) as save:
    save.write(nvdi, 1)

with rasterio.open("GDVI.tif", "w", **profile) as save:
    save.write(gdvi, 1)


# save a new raster with 4 bands
profile.update(
    dtype=rasterio.uint8,
    count=4,
    compress="lzw",
    nodata=None,
    driver="GTiff",
)

with rasterio.open("RGBIR.tif", "w", **profile) as save:
    save.write(bandr, 1)
    save.write(bandg, 2)
    save.write(bandb, 3)
    save.write(band1, 4)

