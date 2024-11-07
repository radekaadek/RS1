import rasterio
import numpy as np

def load_raster(file_path, bands):
    with rasterio.open(file_path) as src:
        profile = src.profile
        data = [src.read(band) for band in bands]
    return profile, data

def calculate_indices(band1, bandr, bandg=None):
    band1 = band1.astype(np.float32).copy()
    bandr = bandr.astype(np.float32).copy()
    
    ndvil = np.array(band1 - bandr)
    ndvim = np.array(band1 + bandr)
    ndvi = ndvil / ndvim
    ndvi[(ndvim == 0) | np.isnan(ndvim)] = 0
    ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())
    ndvi = np.multiply(ndvi, 255)
    ndvi = np.round(ndvi).astype(np.uint8)
    
    gdvi = None
    if bandg is not None:
        bandg = bandg.astype(np.float32).copy()
        gdvi = band1 - bandg
        gdvi = (gdvi - gdvi.min()) / (gdvi.max() - gdvi.min())
        gdvi = np.multiply(gdvi, 255)
        gdvi = np.round(gdvi).astype(np.uint8)
    
    return ndvi, gdvi

def save_raster(file_path, profile, data):
    with rasterio.open(file_path, "w", **profile) as save:
        for i, band in enumerate(data, start=1):
            save.write(band, i)

# Load rasters
profile_2023, [band1_2023] = load_raster("/home/ard/prg/RS1/1/IR_2023/N-34-111-C-c-3-3.tif", [1])
_, [bandr_2023, bandg_2023, bandb_2023] = load_raster("/home/ard/prg/RS1/1/RGB_2023/78936_1204849_N-34-111-C-c-3-3.tif", [1, 2, 3])
_, [band1_2015] = load_raster("/home/ard/prg/RS1/1/IR_2015/N-34-C-c-3-3.tif", [1])
_, [bandr_2015] = load_raster("/home/ard/prg/RS1/1/R_2015/N-34-C-c-3-3.tif", [1])

# Calculate indices
ndvi_2023, gdvi_2023 = calculate_indices(band1_2023, bandr_2023, bandg_2023)
ndvi_2015, gdvi_2015 = calculate_indices(band1_2015, bandr_2015)

# Save NDVI and GDVI
profile_2023.update(dtype=rasterio.uint8)
save_raster("NDVI_2023.tif", profile_2023, [ndvi_2023])
save_raster("GDVI_2023.tif", profile_2023, [gdvi_2023])

profile_2015 = profile_2023.copy()
save_raster("NDVI_2015.tif", profile_2015, [ndvi_2015])
if gdvi_2015 is not None:
    save_raster("GDVI_2015.tif", profile_2015, [gdvi_2015])

# Save a new raster with 4 bands for 2023
profile_2023.update(
    dtype=rasterio.uint8,
    count=4,
    compress="lzw",
    nodata=None,
    driver="GTiff",
)
save_raster("RGBIR_2023.tif", profile_2023, [bandr_2023, bandg_2023, bandb_2023, band1_2023])
