import numpy as np
import rasterio
import os

def normalize(band):
    band = (band - band.min()) / (band.max() - band.min())
    return band

with rasterio.open('grupa_6.tif') as src:
    profile = src.profile
    bands = [src.read(band) for band in range(1, profile['count'] + 1)]

for i in range(len(bands)):
    # band = band.astype(np.float32)
    bands[i] = bands[i].astype(np.float32)
profile.update(dtype=rasterio.float32)

coastal_blue = bands[0]
blue = bands[1]
green_i = bands[2]
green = bands[3]
yellow = bands[4]
red = bands[5]
rededge = bands[6]
nir = bands[7]


# BAEI
baei = (red + 0.3 ) / (green + nir)
baei[(baei == 0) | np.isnan(baei) | np.isinf(baei)] = 0
baei = normalize(baei)
# np.savetxt('baei.txt', baei, fmt='%.2f')


profile.update(count=1)
with rasterio.open('baei.tif', 'w', **profile) as dst:
    dst.write(baei, 1)
# with rasterio.open('ndbi.tif', 'w', **profile) as dst:
#     dst.write(ndbi, 1)

def calculate_index(bands: list[np.ndarray], filename: str, threshold: float):
    band_nums = len(bands)
    index_bands = [[] for _ in range(band_nums)]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            band_number = int(line[5])
            index_bands[band_number - 1].append(float(line.split()[-1]))
    index_bands = np.array(index_bands)
    means = np.array([np.mean(b) for b in index_bands])
    # diffs = np.array([np.abs(band - mean) for band, mean in zip(bands, means)])
    diffs = np.zeros((np.shape(bands[0])[0], np.shape(bands[0])[1]))
    for i, band in enumerate(bands):
        diffs += np.abs(band - means[i])
    diffs_norm = normalize(diffs)
    # set larger than 0.04 to 1
    new_diffs = np.zeros_like(diffs_norm, dtype=np.float32)
    new_diffs[diffs_norm > threshold] = 1
    # set smaller than 0.04 to 0
    new_diffs[diffs_norm <= threshold] = 1/2
    return new_diffs


new_diffs = calculate_index(bands, 'drogi.txt', 0.035)

not_zero = (nir + red) != 0

# Identify water areas where nir < 900
water = np.zeros_like(nir, dtype=np.float32)
water[nir < 900] = 1
water[nir >= 900] = 0

idx_p = calculate_index(bands, 'pola.txt', 0.05)
idx_p1 = calculate_index(bands, 'pola1.txt', 0.05)


red_index = red / (green + blue)

# Create result_buildings raster
result_buildings = np.zeros_like(nir, dtype=np.float32)
result_buildings += 0  # Initialize all values to 1/2
result_buildings[np.logical_and.reduce((new_diffs < 1, baei > 0.355, water != 1, idx_p == 1, idx_p1 == 1))] = 1
result_buildings[red_index > 1] = 1
result_buildings[coastal_blue == np.nan] = np.nan


with rasterio.open('red_index.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(red_index, 1)


with rasterio.open('idx.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(new_diffs, 1)

with rasterio.open('water.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(water, 1)

with rasterio.open('result.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(result_buildings, 1)

# use gdal_sieve.py to remove small islands
gdal_command = f"gdal_sieve -st 4 result.tif result_sieved.tif"
os.system(gdal_command)

with rasterio.open('result_sieved.tif') as src:
    profile = src.profile
    bands = [src.read(band) for band in range(1, profile['count'] + 1)]
