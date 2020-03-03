import numpy as np
import pandas as pd
import sys

# Take path of original data and location for new file data set from arguments
original_data_path = sys.argv[1]
clean_data_path = sys.argv[2]

df = pd.read_csv(original_data_path, header="infer")
# Drop the time stamp of sample data set
data = df.drop(df.columns[0], axis=1)
# Drop sample which contains missing value
data = data.replace('?', np.NaN)
processed_data = data.dropna()
# Save clean data to new file
np.savetxt(clean_data_path, processed_data, delimiter=',', fmt='%s')
