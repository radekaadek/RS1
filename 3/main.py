import numpy as np
import rasterio

def normalize(band):
    band = (band - band.min()) / (band.max() - band.min())
    return band

with rasterio.open('grupa_6.tif') as src:
    profile = src.profile
    print(profile)
    bands = [src.read(band) for band in range(1, profile['count'] + 1)]

for band in bands:
    band = band.astype(np.float32)
profile.update(dtype=rasterio.float32)

coastal_blue = bands[0]
blue = bands[1]
green_i = bands[2]
green = bands[3]
yellow = bands[4]
red = bands[5]
rededge = bands[6]
nir = bands[7]


ndbi = (nir - red) / (nir + red)
ndbi[(ndbi == 0) | np.isnan(ndbi) | np.isinf(ndbi)] = 0
ndbi = normalize(ndbi)
#save to a file
np.savetxt('ndbi.txt', ndbi, fmt='%.2f')

# BAEI
baei = (red + 0.3 ) / (green + nir)
baei[(baei == 0) | np.isnan(baei) | np.isinf(baei)] = 0
baei = normalize(baei)
# set >0.25 to 1 else 0
baei[baei >= 0.25] = 1
baei[baei < 0.25] = 0
np.savetxt('baei.txt', baei, fmt='%.2f')

profile.update(count=1)
with rasterio.open('baei.tif', 'w', **profile) as dst:
    dst.write(baei, 1)
with rasterio.open('ndbi.tif', 'w', **profile) as dst:
    dst.write(ndbi, 1)

