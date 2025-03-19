#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:29:20 2025

@author: wileyjones
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "/Users/wileyjones/Downloads/cbarker211-Satellite-Megaconstellation-Emission-Inventory-Development-65ccfb3/databases/launch_activity_data_2020-2022.nc"
dataset = xr.open_dataset(filename)
print(dataset)
df = dataset.to_dataframe().reset_index()
print(df)
rocket_names = df["Rocket_Name"]
print(rocket_names)
reduced_rocket_names = rocket_names.drop_duplicates()
print(df.columns)
print(len(df))

filename2 = "/Users/wileyjones/Downloads/cbarker211-Satellite-Megaconstellation-Emission-Inventory-Development-65ccfb3/databases/rocket_attributes_2020-2022.nc"
data = xr.open_dataset(filename2)
df2 = data.to_dataframe().reset_index()
print(df2)
print(df2.columns)
print(df2["Stage1_PropMass"], df2["Stage1_Fuel_Type"])
print(df2["Stage2_PropMass"], df2["Stage2_Fuel_Type"])
print(df2["Stage3_PropMass"], df2["Stage3_Fuel_Type"])
print(df2["Stage4_PropMass"], df2["Stage4_Fuel_Type"])
print(df2["Rocket_Name"])

df2.to_csv("Rocket_Attributes.csv")

filename3 = "/Users/wileyjones/Downloads/16974166/rocketresults_r10_radiativeforcing.nc"
data2 = xr.open_dataset(filename3)
df3 = data2.to_dataframe().reset_index()
print(df3)
print(df3.columns)


#############################################################################

### Get Rocket emission into Map with values
rocket_att_dict = {}

for value in df2["Rocket_Name"]:
    rocket_att_dict[value] = 0
    
for i, key in enumerate(rocket_att_dict.keys()):
    att = []
    for entry in df2.iloc[i]:
        att.append(entry)
    rocket_att_dict[key] = att[2:]
 
### Get list of list where each inner list will become a new column in df   
series_to_add = []   
for i in range(18):
    series_column = []
    for value in df["Rocket_Name"]:
        series_column.append(rocket_att_dict[value][i]) 
    series_to_add.append(series_column)

## ADD the new columns to dataframe
columns_to_add_to_df = df2.columns[2:]
print(len(columns_to_add_to_df))

for i in range(len(columns_to_add_to_df)):
    df[columns_to_add_to_df[i]] = series_to_add[i]

df.to_csv("working_rocket_emissions.csv")

### Drop unneeded columns
df_drops = ["launches", "COSPAR_ID", 'Time(UTC)', 
            'Date', 'DISCOSweb_Rocket_ID', "Proxy_Rocket"]

df = df.drop(columns=df_drops)

LABEL = df["Megaconstellation_Flag"]
df.insert(0, "LABEL", LABEL)
df.columns
df = df.drop(columns=["Megaconstellation_Flag"])
df.columns

names_of_rockets= df.pop("Rocket_Name")
df.insert(1, "Rocket_Name", names_of_rockets)
df.columns

## Add Fuel Columns together into one

index_dict = {}

for i, col in enumerate(df.columns):
    index_dict[col] = i + 1
    
print(index_dict)    
    
print(df["Stage1_Fuel_Type"].value_counts())

kerosene_total = []
hypergolic_total = []
solid_total = []
hydrogen_total = []
methane_total = []

fuel_type_check = ["Booster_Fuel_Type", "Stage1_Fuel_Type",
                   "Stage2_Fuel_Type", "Stage3_Fuel_Type",
                   "Stage4_Fuel_Type"]

prop_mass = ["Booster_PropMass", "Stage1_PropMass", "Stage2_PropMass",
             "Stage3_PropMass", "Stage4_PropMass"]

for row in df.itertuples():
    kerosene = 0
    hypergolic = 0
    solid = 0
    hydrogen = 0
    methane = 0
    for i in range(len(fuel_type_check)):
        if row[index_dict[fuel_type_check[i]]] == "Kerosene":
            kerosene += row[index_dict[prop_mass[i]]]
        elif row[index_dict[fuel_type_check[i]]] == "Hypergolic":
            hypergolic += row[index_dict[prop_mass[i]]]
        elif row[index_dict[fuel_type_check[i]]] == "Solid":
            solid += row[index_dict[prop_mass[i]]]
        elif row[index_dict[fuel_type_check[i]]] == "Hydrogen":
            hydrogen += row[index_dict[prop_mass[i]]]
        elif row[index_dict[fuel_type_check[i]]] == "Methane":
            methane += row[index_dict[prop_mass[i]]]
    kerosene_total.append(kerosene)
    hypergolic_total.append(hypergolic)
    solid_total.append(solid)
    hydrogen_total.append(hydrogen)
    methane_total.append(methane)
    
print(methane_total)  
len(df)    
type(index_dict[fuel_type_check[0]])
index_dict[prop_mass[0]]
  
emissions_data = "/Users/wileyjones/Desktop/CS332/Project Files/primary_emission_indices.csv"      
emissions_df = pd.read_csv(emissions_data)

emission_total = [hypergolic_total, kerosene_total, solid_total, 
                  hydrogen_total, methane_total]

PEI = emissions_df.values.tolist()

emissions_df.columns
H2O = emissions_df["H2O"].tolist()
print(H2O)

print(len(PEI))

for i in range(len(PEI)):
    PEI[i] = PEI[i][2:]
    
print(len(PEI))

gas_total = []

for k in range(len(PEI[0])):
    total_per_gas = []
    for i in range(len(emission_total[0])):
        total = 0
        for j in range(len(emission_total)):
            total += (emission_total[j][i] * PEI[j][k])
        total_per_gas.append(total)
    gas_total.append(total_per_gas)
 
print(gas_total)
gas_total[7]

gases = ["H2O", "H2", "CO", "CO2", "BC", "NOx", "AI203", "Cly"]
len(gas_total)

for i in range(len(gases)):
    df[gases[i]] = gas_total[i]

df.columns

df.to_csv("2.0_emissions.csv")

columns = df.columns

for col in columns:
    print(f"{col} has {df[col].isna().sum()} missing values")

booster = sns.countplot(df, x="Stage1_Fuel_Type")
plt.show()

df = df.astype({"Booster_No": int})
df.dtypes
