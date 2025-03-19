#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:28:13 2025

@author: wileyjones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

filename = "/Users/wileyjones/Desktop/CS332/Project Files/UCS-Satellite-Database 5-1-2023 (1).csv"
UCS_db = pd.read_csv(filename)
pd.set_option('display.max_columns', 10)

### UCS DataBase ############################################################
print(UCS_db)
print(UCS_db.columns)

## Delete Empty Columns
UCS_db = UCS_db.iloc[:,:27]
print(UCS_db.columns)

## Delete unneeded Columns
UCS_db = UCS_db.iloc[:,3:25]
print(UCS_db.columns)

## Remove bottom two rows
UCS_db = UCS_db.drop(UCS_db.tail(2).index)
print(UCS_db)

## Check dtypes
print(UCS_db.dtypes)

## Convert all Rows to correct dtype
columns_to_numeric = ["Perigee (km)", 
                      "Apogee (km)",
                     "Period (minutes)",
                     "Launch Mass (kg.)",
                     " Dry Mass (kg.) ",
                     "Power (watts)"]

## Perigee Conversion
perigee_list = UCS_db["Perigee (km)"].tolist()
print(perigee_list)

# Remove commas
for i in range(len(perigee_list)):
    perigee_list[i] = str(perigee_list[i]).replace(",", "")

print(len(perigee_list))

# Replace old list with new list
UCS_db["Perigee (km)"] = perigee_list
print(UCS_db["Perigee (km)"][:15])

UCS_db["Perigee (km)"] = pd.to_numeric(UCS_db["Perigee (km)"], errors="coerce")
print(UCS_db['Perigee (km)'])


## Apogee Conversion ##

apogee_list = UCS_db["Apogee (km)"].tolist()
print(apogee_list)
print(len(apogee_list))

# Remove commas
for i in range(len(apogee_list)):
    apogee_list[i] = str(apogee_list[i]).replace(",", "")


# Replace old list with new list
UCS_db["Apogee (km)"] = apogee_list
print(UCS_db["Apogee (km)"][:15])

UCS_db["Apogee (km)"] = pd.to_numeric(UCS_db["Apogee (km)"], errors="coerce")
print(UCS_db['Apogee (km)'])

print(UCS_db.dtypes)


## Launch Mass ##

mass_list = UCS_db["Launch Mass (kg.)"].tolist()
print(mass_list)
print(len(mass_list))

# Remove commas
for i in range(len(mass_list)):
    mass_list[i] = str(mass_list[i]).replace(",", "")


# Replace old list with new list
UCS_db["Launch Mass (kg.)"] = mass_list
print(UCS_db["Launch Mass (kg.)"][:15])

UCS_db["Launch Mass (kg.)"] = pd.to_numeric(UCS_db["Launch Mass (kg.)"], errors="coerce")
print(UCS_db["Launch Mass (kg.)"])

print(UCS_db.dtypes)


## Period Converstion ##

period_list = UCS_db["Period (minutes)"].tolist()
print(period_list)
print(len(period_list))

# Remove commas
for i in range(len(period_list)):
    period_list[i] = str(period_list[i]).replace(",", "")

# Replace old list with new list
UCS_db["Period (minutes)"] = period_list
print(UCS_db["Period (minutes)"][:15])

UCS_db["Period (minutes)"] = pd.to_numeric(UCS_db["Period (minutes)"], errors="coerce")
print(UCS_db["Period (minutes)"])

print(UCS_db.dtypes)


## Dry Mass and Power ##

print("Dry Mass has this many missing values: ")
print(UCS_db[" Dry Mass (kg.) "].isna().sum())

print("Power has this many missing values: ")
print(UCS_db[" Dry Mass (kg.) "].isna().sum())

print(f"These variables only have {((7558 - 6791)/7558) * 100}% of data")

UCS_db = UCS_db.drop(columns=[" Dry Mass (kg.) ", "Power (watts)"])
print(UCS_db.columns)

print("Detailed has this many missing values: ")
print(UCS_db["Detailed Purpose"].isna().sum())

print("Type of Orbit has this many missing values: ")
print(UCS_db["Type of Orbit"].isna().sum())

UCS_db = UCS_db.drop(columns=["Detailed Purpose", "Type of Orbit"])
print(UCS_db.columns)

## Convert Date into year

print(UCS_db.dtypes)

print(UCS_db["Date of Launch"].isna().sum())
UCS_db = UCS_db.dropna(subset=["Date of Launch"])
print(UCS_db["Date of Launch"].isna().sum())

launch_year = UCS_db["Date of Launch"].tolist()
print(type(launch_year[15]))

print(UCS_db["Date of Launch"].describe())


for i in range(len(launch_year)):
    year = str(launch_year[i])
    year = year[-2:]
    if int(year) < 25:
        year = "20" + year
    else:
        year = "19" + year
    launch_year[i] = year    
    
    
print(launch_year)
UCS_db["Date of Launch"] = launch_year

UCS_db = UCS_db.rename(columns={'Date of Launch': "Year of Launch"})

## Convert Class of Orbit to Category ##

UCS_db = UCS_db.astype({"Class of Orbit": "category"})
print(UCS_db.dtypes)

orbit_class = sns.catplot(UCS_db, x="Class of Orbit")
plt.show()


### CLEANING MISSING VALUES #################################################

columns = UCS_db.columns
for col in columns:
    print(f"{col} has {UCS_db[col].isna().sum()} missing values")

print(f"The mean of Expected Lifetime is {UCS_db['Expected Lifetime (yrs.)'].mean()}")
print(f"The median of Expected Lifetime is {UCS_db['Expected Lifetime (yrs.)'].median()}")
expected_lifetime = UCS_db["Expected Lifetime (yrs.)"].value_counts()
print(expected_lifetime)

print(UCS_db["Expected Lifetime (yrs.)"].describe())

expected_count_plot = sns.boxplot(UCS_db, x="Expected Lifetime (yrs.)")
plt.show()

UCS_db["Expected Lifetime (yrs.)"] = UCS_db["Expected Lifetime (yrs.)"].fillna(UCS_db["Expected Lifetime (yrs.)"].mean())
expected_lifetime = UCS_db["Expected Lifetime (yrs.)"].value_counts()
print(expected_lifetime)
print(UCS_db["Expected Lifetime (yrs.)"].median())

columns = UCS_db.columns
for col in columns:
    print(f"{col} has {UCS_db[col].isna().sum()} missing values")
    
UCS_db = UCS_db.dropna()

print(UCS_db)

## CLEANING MISTAKES ########################################################

## Fixing Orbit Mistake 
row = UCS_db.loc[UCS_db["Class of Orbit"] == "LEo"]
print(row)
UCS_db.loc[UCS_db["Class of Orbit"] == "LEo", "Class of Orbit"] = "LEO" 

UCS_db = UCS_db.astype({"Class of Orbit": str})
orbits=UCS_db["Class of Orbit"].value_counts()
print(orbits)

UCS_db = UCS_db.astype({"Class of Orbit": 'category'})
orbits=UCS_db["Class of Orbit"].value_counts()
print(orbits)

orbit_class = sns.catplot(UCS_db, x="Class of Orbit")
plt.show()

### CHECKING FOR MISTAKES AND MISFORMATS ###################################


counts_list = []
columns = UCS_db.columns

for col in columns:
    counts_list.append(UCS_db[col].value_counts())
    
print(counts_list[2])

#### USERS ######
users_plot = sns.histplot(UCS_db, y="Users")
plt.show()

UCS_db["Users"] = UCS_db["Users"].str.strip()

users_plot = sns.histplot(UCS_db, y="Users")
plt.show()


#### PURPOSE #######
print(counts_list[3])

purpose_plot = sns.histplot(UCS_db, y="Purpose")
plt.show()

UCS_db["Purpose"] = UCS_db["Purpose"].str.strip()

##### USERS CONTINUED ########################
purpose_plot = sns.histplot(UCS_db, y="Purpose")
plt.show()

new_user_count = UCS_db["Users"].value_counts()
print(new_user_count)

outlier_check = sns.boxenplot(UCS_db, x="Perigee (km)", hue="Class of Orbit")
plt.show()

UCS_db.loc[(UCS_db["Class of Orbit"] == "LEO") & (UCS_db["Perigee (km)"] > 10000), ["Class of Orbit"]] = "GEO"
row = UCS_db[(UCS_db["Class of Orbit"] == "GEO") & (UCS_db["Perigee (km)"] < 10000)].index[0]
print(row)

UCS_db = UCS_db.drop(index=2866)

print(UCS_db)

outlier_check = sns.boxenplot(UCS_db, x="Perigee (km)", hue="Class of Orbit")
plt.show()

row = UCS_db.loc[(UCS_db["Class of Orbit"] == "GEO") & (UCS_db["Perigee (km)"] < 10000)]
print(row) 

outlier_check = sns.relplot(UCS_db, x="Perigee (km)", y="Apogee (km)", hue="Class of Orbit")
plt.show()

### Apogee Check
apogee_plot = sns.boxenplot(UCS_db, x="Apogee (km)", hue="Class of Orbit")
plt.show()
row = UCS_db[(UCS_db["Class of Orbit"] == "GEO") & (UCS_db["Apogee (km)"] > 100000)].index[0]
print(row)
UCS_db = UCS_db.drop(index=756)

apogee_plot = sns.boxenplot(UCS_db, x="Apogee (km)", hue="Class of Orbit")
plt.show()


UCS_db = UCS_db.drop(columns=["Longitude of GEO (degrees)"])
print(UCS_db.columns)

### EXPECTED LIFETIME ##########################
check = sns.boxenplot(UCS_db, x="Expected Lifetime (yrs.)", hue="Class of Orbit")
plt.show()

UCS_db = UCS_db.round({"Expected Lifetime (yrs.)": 1})
print(UCS_db["Expected Lifetime (yrs.)"])

label = UCS_db["Class of Orbit"].tolist()
print(label[:14])

UCS_db = UCS_db.drop(columns="Class of Orbit")
UCS_db.insert(0, "Class of Orbit", label)
print(UCS_db.columns)

UCS_db.to_csv("UCS_DB_Cleaned.csv")

###### CREATING NEW DATA FRAME #############################################

UCS_Quant = UCS_db.select_dtypes(include=["number"])
print(UCS_Quant)

UCS_Quant.to_csv("UCS_Quant.csv")
