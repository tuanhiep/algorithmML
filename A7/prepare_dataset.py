#  Copyright (c) 2020. Tuan-Hiep TRAN

import numpy as np
import pandas as pd
import sys

# Take path of original data and location for new file data set from arguments
original_data_path = sys.argv[1]
clean_data_path = sys.argv[2]

df = pd.read_csv(original_data_path, header="infer")
label = np.zeros(df.shape[0])
# Drop the time stamp of sample data set
for i in range(1, 7 + 1):
    print(i)
    label = label + (df.iloc[:, -i].replace(1, i)).to_numpy()

print(label)

X = df.iloc[:, :27]
Y = pd.DataFrame(label)
df = pd.concat([X, Y], axis=1)
# Save clean data to new file
np.savetxt(clean_data_path, df, delimiter=',', fmt='%s')
