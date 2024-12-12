import numpy as np
import rasterio

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


# ndbi = (nir - red) / (nir + red)
# ndbi[(ndbi == 0) | np.isnan(ndbi) | np.isinf(ndbi)] = 0
# ndbi = normalize(ndbi)
# #save to a file
# np.savetxt('ndbi.txt', ndbi, fmt='%.2f')

# BAEI
baei = (red + 0.3 ) / (green + nir)
baei[(baei == 0) | np.isnan(baei) | np.isinf(baei)] = 0
baei = normalize(baei)
# np.savetxt('baei.txt', baei, fmt='%.2f')

# load from baei.tif
# with rasterio.open('baei.tif') as src:
#     baei = src.read(1)

profile.update(count=1)
with rasterio.open('baei.tif', 'w', **profile) as dst:
    dst.write(baei, 1)
# with rasterio.open('ndbi.tif', 'w', **profile) as dst:
#     dst.write(ndbi, 1)

band_nums = 8
road_bands = [[] for _ in range(band_nums)]
with open('drogi.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        band_number = int(line[5])
        road_bands[band_number - 1].append(float(line.split()[-1]))
road_bands = np.array(road_bands)
means = np.array([np.mean(b) for b in road_bands])
# calculate the difference to the mean
print(np.shape(bands[0]), np.shape(means))
# diffs = np.array([np.abs(band - mean) for band, mean in zip(bands, means)])
diffs = np.zeros((np.shape(bands[0])[0], np.shape(bands[0])[1]))
for i, band in enumerate(bands):
    diffs += np.abs(band - means[i])
diffs_norm = normalize(diffs)
# set larger than 0.04 to 1
new_diffs = np.zeros_like(diffs_norm, dtype=np.float32)
threshold = 0.035
new_diffs[diffs_norm > threshold] = 1
# set smaller than 0.04 to 0
new_diffs[diffs_norm <= threshold] = 1/2

not_zero = (nir + red) != 0

ndvi = np.divide((nir - red), (nir + red), out=np.zeros_like(nir-red), where=not_zero)
# minmax scale
ndvi = normalize(ndvi)

# set ndvi > 0.8 to 1
ndvi_filtered = ndvi.copy()
ndvi_filtered[ndvi > 0.8] = 1
ndvi_filtered[ndvi <= 0.8] = 0

# Identify water areas where nir < 900
water = np.zeros_like(nir, dtype=np.float32)
water[nir < 900] = 1
water[nir >= 900] = 0

# Read idx_p and idx_p1 rasters
with rasterio.open('idx_p.tif') as src:
    idx_p = src.read(1)

with rasterio.open('idx_p1.tif') as src:
    idx_p1 = src.read(1)

# Create result_buildings raster
result_buildings = np.zeros_like(nir, dtype=np.float32)
result_buildings += 1/2  # Initialize all values to 1/2
result_buildings[np.logical_and.reduce((new_diffs < 1, baei > 0.355, water != 1, idx_p == 1, idx_p1 == 1))] = 1




# minmax normalize
with rasterio.open('idx.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(new_diffs, 1)

with rasterio.open('diffs.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(diffs_norm, 1)

with rasterio.open('ndvi.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(ndvi, 1)

with rasterio.open('ndvi_filtered.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(ndvi_filtered, 1)

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
