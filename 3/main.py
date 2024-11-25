import numpy as np
import rasterio

def normalize(band):
    band = (band - band.min()) / (band.max() - band.min())
    return band

with rasterio.open('grupa_6.tif') as src:
    profile = src.profile
    print(profile)
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
# baei = (red + 0.3 ) / (green + nir)
# baei[(baei == 0) | np.isnan(baei) | np.isinf(baei)] = 0
# baei = normalize(baei)
# # set >0.25 to 1 else 0
# baei[baei >= 0.25] = 1
# baei[baei < 0.25] = 0
# np.savetxt('baei.txt', baei, fmt='%.2f')

profile.update(count=1)
# with rasterio.open('baei.tif', 'w', **profile) as dst:
#     dst.write(baei, 1)
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
print("here")
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

nvdi = np.divide((nir - red), (nir + red), out=np.zeros_like(nir-red), where=not_zero)
print(nvdi)
# minmax scale
nvdi = normalize(nvdi)


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

with rasterio.open('nvdi.tif', 'w', **profile) as dst:
    profile.update(count=1)
    profile.update(dtype=rasterio.float32)
    profile.update(nodata=np.nan)
    dst.write(nvdi, 1)

