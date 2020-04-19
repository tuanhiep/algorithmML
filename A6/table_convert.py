# https://www.kaggle.com/cheedcheed/california-wind-power-generation-forecasting/data
import pandas as pd

# PRE-PROCESSING the data
# 1 load X, y
df_crp = pd.read_csv('../csv/all_breakdown.csv', header=0, low_memory=False)
print(df_crp.head(8))
rows, _ = df_crp.shape
print("rows", rows)
df_crp = df_crp.fillna(0)
print(df_crp.head())
df_crp = df_crp[["SOLAR", "SOLAR PV", "SOLAR THERMAL"]]
df_crp["SOLAR_TOTAL"] = df_crp["SOLAR"] + df_crp["SOLAR PV"] + df_crp["SOLAR THERMAL"]
print(df_crp.head(24))
df_crp_solar = df_crp["SOLAR_TOTAL"]
print(df_crp_solar.head(10))
print(df_crp_solar.shape)

df_daily = df_crp_solar.groupby(df_crp_solar.index // 24).sum()
print("Total days:", df_daily.shape)
print(df_daily)
num_rows = df_daily.shape[0]

weekly = []
prev_i = 0
num_days = 7
for i in range(num_days, num_rows, num_days):
    week = []
    print("week", i / num_days)
    first = i - num_days
    print("First day of week:", first)
    last = i
    print("Last day of week:", last)
    for day in range(first, last):
        print("day:", day)
        week.append(df_daily.iloc[day])
    weekly.append(week)
print(weekly)

weekly_columns = ["1", "2", "3", "4", "5", "6", "7"]
weekly_df = pd.DataFrame(weekly, columns=weekly_columns)
print(weekly_df)

# Get the first column
weekly_first_column_df = weekly_df["1"]
print(weekly_first_column_df)

# Shift to the top
weekly_first_column_df = weekly_first_column_df.shift(-1)
weekly_first_column_df = weekly_first_column_df.rename("Target")
print(weekly_first_column_df)
# print(type(weekly_first_column_df))

# Cat to the weekly df
weekly_target_df = pd.concat([weekly_df, weekly_first_column_df], axis=1)
print(weekly_target_df)
