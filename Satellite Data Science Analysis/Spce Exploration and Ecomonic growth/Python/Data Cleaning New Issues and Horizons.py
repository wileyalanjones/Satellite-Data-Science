#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:09:05 2025

@author: wileyjones
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

filename="/Users/wileyjones/Desktop/CS332/Project Files/launch_df_base.csv"
df = pd.read_csv(filename)

print(df.head(10))
print(df.columns)


##### DELETE UNNEEDED COLUMNS #############################################
columns_to_keep = ["Launch_Date", "LV_Type", "price_per_kg", "max_mass"]
columns_to_delete = [col for col in df.columns if col not in columns_to_keep]
print(columns_to_delete)

df = df.drop(columns=columns_to_delete)
print(df.columns)

##### FIX DATATYPES #######################################################
print(df.dtypes)

print(df["Launch_Date"].head(10))

date_list = df["Launch_Date"].tolist()
date = date_list[0].strip()
print(len(date))

check_list = []

def changemonth(mon: str):
    mon_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06", 
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12"
        }
    return mon_dict[mon]

for date in date_list:
    year = date[:4]
    mon = changemonth(date[5:8])
    day = date[9:11]
    check_list.append(f"{year}-{mon}-{day}")
    
print(check_list)

for i in range(len(check_list)):
    recheck = check_list[i].replace(" ", "0")
    check_list[i] = recheck
    
print(check_list[0:200])

df["Launch_Date"] = check_list

print(df.head(20))

df["Launch_Date"] = pd.to_datetime(df["Launch_Date"])
print(df.dtypes)

print(df["Launch_Date"].head(20))

##### MISSING VALUES ########################################################

columns = df.columns
for col in columns:
    print(f"{col} has {df[col].isna().sum()} missing values")
    
missing = df["price_per_kg"].isna().sum()
print((missing / len(df["price_per_kg"]) * 100))

len(df["price_per_kg"])

df = df.dropna()

##### INCORRECT VALUES #####################################################

### LV TYPE 
LV_count = df["LV_Type"].value_counts()
print(LV_count[68]) 

for count in LV_count:
    print(count)

print(LV_count[0:68])

df = df[df["LV_Type"] != "Space Shuttle"]
print(df.columns)

df = df[df["LV_Type"] != "Saturn V"]

LV_count = df["LV_Type"].value_counts()
print(LV_count[:67]) 

LV_type = sns.histplot(df, y="LV_Type")
plt.show()

### Price

price_check = sns.boxplot(df, x="price_per_kg")
plt.show()

df = df[df["price_per_kg"] < 800000]

price_check = sns.boxplot(df, x="price_per_kg")
plt.show()

### Mass 

mass_check = sns.boxplot(df, x="max_mass")
plt.show()

#### Add Label 

filename="/Users/wileyjones/Desktop/CS332/Project Files/VehicleFamilies.csv"
df_veh = pd.read_csv(filename)

columns = df_veh.columns

columns_keep = ["Family", "Class"]
columns_drop = [col for col in columns if col not in columns_keep]

df_veh = df_veh.drop(columns = columns_drop)

df_veh.isna().sum()

df_veh = df_veh.dropna()
df_veh = df_veh.drop_duplicates()

family = df_veh["Family"].tolist()
veh_class = df_veh["Class"].tolist()

class_dict = {}

for i in range(len(family)):
    class_dict[family[i]] = veh_class[i]

print(class_dict)

lv = df["LV_Type"].tolist()
class_list = []

for vehicle in lv:
    if vehicle in class_dict:
        class_list.append(class_dict[vehicle])
    else:
        class_list.append(None)
        
print(class_list)

len(class_list)
len(df)

df.insert(0, "Class", class_list)
print(df)

class_count = df["Class"].value_counts()
print(class_count)

df.to_csv("Satelitte_Cost_Clean.csv")

###### MIN/MAX ##############################################################

df_2 = df.copy()
print(df_2)

print(df)

df_save = df.copy()

scaler = MinMaxScaler()
df_2[["price_per_kg", "max_mass"]] = scaler.fit_transform(df_2[["price_per_kg", "max_mass"]])
print(df_2)

print(df_2["price_per_kg"])

df_2.to_csv("Satellite_Cost_Min_Max.csv")

line = sns.relplot(df, x="Launch_Date", y="price_per_kg", hue="Class")
plt.show()

df_quant = df_save.drop(columns=["Class", "Launch_Date", "LV_Type"])
print(df_quant)

df_quant_min_max = df_2.drop(columns=["Class", "Launch_Date", "LV_Type"])
print(df_quant_min_max)

df_quant_min_max.to_csv("PriceVsPayloadMassMinMax.csv")

df_save["Max Cost"] = df_save["price_per_kg"] * df_save["max_mass"]

print(df_save)
df_save["Max Cost"].describe()

df_save[["price_per_kg", "max_mass", "Max Cost"]] = scaler.fit_transform(df_save[["price_per_kg", "max_mass", "Max Cost"]])
print(df_save)

df_quant = df_save.drop(columns=["Class", "Launch_Date", "LV_Type"])
print(df_quant)

df_quant.to_csv("PriceVsPayloadMassMinMax.csv")

df = df.dropna()

df_quant = df.select_dtypes(include="number")

df_quant.to_csv("EmissionsDataQuant.csv")

