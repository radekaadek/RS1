import rasterio
import numpy as np
import time
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with rasterio.open('test.tif') as src:
    building_values = src.read(1)
    print(src.profile)

with rasterio.open('grupa_6_2180.tif') as src:
    # get number of bands
    nbands = src.count
    bands = [src.read(i) for i in range(1, nbands + 1)]
    bands = np.array(bands)

# trim arrays to the same sizes
building_values = building_values[:bands[0].shape[0], :bands[0].shape[1]]
bands = bands[:, :building_values[0].shape[0], :building_values.shape[1]]
print(f"building_values.shape: {building_values.shape}")
print(f"bands.shape: {bands.shape}")


# flatten building_values array to 1D
building_values = building_values.flatten()

df = pandas.DataFrame(building_values, columns=['building_values'])
# convert to int
df['building_values'] = df['building_values'].astype(int)
for i in range(len(bands)):
    df[f"band{i}"] = bands[i].flatten()
# remove row with all zeros
df = df[df.sum(axis=1) != 0]

# save df to csv
df.to_csv('df.csv', index=False)

# take the top 10000 rows
df = df.head(100)
df = df.sample(frac=0.5)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df, df['building_values'], test_size=0.2, random_state=42)

# create a random forest classifier
t1 = time.time()
clf = RandomForestClassifier(random_state=42)
print(f"Time to create classifier: {time.time() - t1}")

# train the classifier
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
