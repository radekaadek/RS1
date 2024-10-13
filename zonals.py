import pandas as pd
import matplotlib.pyplot as plt

# Assuming the file 'zonal_statistics.csv' is in the correct path
file = "zonal_statistics.csv"

# Load the CSV file
df = pd.read_csv(file)

pref2name = {'_': 'IR_2015', '3': 'IR_2023', '4': 'R_2015', '5': 'R_2023'}

for prefix, name in pref2name.items():
    mean_name = prefix + 'mean'
    stdev_name = prefix + 'stdev'
    min_name = prefix + 'min'
    max_name = prefix + 'max'

# Calculate Q1 and Q3 for each row in the DataFrame
    q1 = df[mean_name] - 1.5 * df[stdev_name]  # 25th percentile (approx.)
    q3 = df[mean_name] + 1.5 * df[stdev_name]  # 75th percentile (approx.)

# Prepare a list to hold boxplot data
    boxplot_data_list = []

# Iterate over the DataFrame rows and create boxplot data dictionaries
    for idx, row in df.iterrows():
        boxplot_data = {
            'whislo': row[min_name],          # Minimum value
            'q1': q1[idx],                     # 25th percentile (Q1)
            'med': row[mean_name],             # Median (using 'Średnia' as median)
            'q3': q3[idx],                     # 75th percentile (Q3)
            'whishi': row[max_name],         # Maximum value
            'mean': row[mean_name]             # Mean (Średnia) is shown as the mean
        }
        boxplot_data_list.append(boxplot_data)

# Create the figure and axis
    fig, ax = plt.subplots()

# Draw the boxplots for all rows
    ax.bxp(boxplot_data_list, showfliers=False)

# Set the title
    ax.set_title(f"Wykres wartości zonal dla {name}")
    ax.set_xlabel("Numer drogi")
    ax.set_ylabel("Wartośći pikseli")

# Show the plot
    plt.savefig(f"zonal_{prefix}.png")

