#  Copyright (c) 2020. Tuan-Hiep TRAN

import numpy as np
import pandas as pd
import sys

# Take path of original data and location for new file data set from arguments
original_data_path = sys.argv[1]
clean_data_path = sys.argv[2]

df = pd.read_csv(original_data_path, header="infer")
# Drop the time stamp of sample data set
print(df)
data = df.drop(["TIMESTAMP", "Hour"], axis=1)
# Replace sample missing value by zero
data.fillna(0, inplace=True)
print(data.head())
df_daily = data.groupby(data.index // 24).sum()
df_weekly = df_daily.groupby(df_daily.index // 7).sum()
print("Total weeks:", df_weekly.shape)
print(df_weekly)
# Save clean data to new file
np.savetxt(clean_data_path, df_weekly, delimiter=',', fmt='%s')
