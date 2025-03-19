#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:35:18 2025

@author: wileyjones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl

filename = "/Users/wileyjones/Desktop/CS332/Project Files/Emissions/EmissionsDataQuant.csv"
df = pd.read_csv(filename)

print(df.head(10))
print(df.columns)

df = df.drop(columns="Unnamed: 0")
df = df[df["CO2"] < 500000000]
df = df.drop(columns=["Longitude", "Latitude"])

print(df["Fairing_Mass"])

columns = df.columns
print(columns)
for col in columns:
    print(col)

## Save Variables for Graph
CO2 = np.array(df["CO2"].tolist())
BC  = np.array(df["BC"].tolist())
CO = np.array(df["CO"].tolist())

### 3D representation 
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(CO, CO2, BC, marker="v", color='red', alpha=1, facecolor="none")

ax.set_xlabel('CO')
ax.set_ylabel("CO2")
ax.set_title("Satellite Emissons Chart")
ax.invert_xaxis()

plt.show()

carbon = sns.scatterplot(df, x="NOx", y="CO2", hue="Booster_No")
plt.show()

## Set up KMeans Object
kmeans_2_object = KMeans(n_clusters=2, max_iter=100)

## Run KMeans on the clean quant data frame
kmeans = kmeans_2_object.fit(df)
kmeans_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

print(kmeans_labels[:1000])

## Make Data Float type
for j in range(len(cluster_centers)):
    for i in range(len(cluster_centers[0])):
        print(f"{columns[i]} :", float(cluster_centers[j][i]))
    print("\n")
    
for col in columns:
    print(df[col].describe())
    print("\n")
    
new_emmissions = [[0,0,0,90000, 6500, 20000, 2400, 0,0,0,0,1000,90000000,
                  1000000, 15000000, 30000000, 4000000,0,0,0], 
                  [4, 150000, 14000, 400000, 25000, 100000, 4000, 5000,
                   900, 0,0, 1800, 150000000, 5000000, 250000000,
                   300000000, 12000000, 700000,0,0]]

pred_kmeans = kmeans_2_object.predict(new_emmissions)
print(pred_kmeans)

### 3D representation with two
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(CO, CO2, BC, marker="v", color='red', alpha=.25, facecolor="none")

ax.set_xlabel('CO')
ax.set_ylabel("CO2")
ax.set_title("Satellite Emissons Chart")
ax.invert_xaxis()

for j in range(len(cluster_centers)):
    cluster_CO   = cluster_centers[j][14]  # Mass
    cluster_CO2  = cluster_centers[j][15]  # Price
    cluster_BC   = cluster_centers[j][16]  # Cost
    ax.scatter(cluster_CO, cluster_CO2, cluster_BC, 
               marker="X", color="black", s=50)

plt.show()

#### THREE CLUSTERS ####

## Set up KMeans Object
kmeans_3_object = KMeans(n_clusters=3, max_iter=100)

## Run KMeans on the clean quant data frame
kmeans = kmeans_3_object.fit(df)
kmeans_labels_3 = kmeans.labels_
cluster_centers_3 = kmeans.cluster_centers_

print(kmeans_labels_3[:1000])

## Make Data Float type
for j in range(len(cluster_centers_3)):
    for i in range(len(cluster_centers_3[0])):
        print(f"{columns[i]} :", float(cluster_centers_3[j][i]))
    print("\n")
    
    
pred_kmeans = kmeans_3_object.predict(new_emmissions)
print(pred_kmeans)

  
#### FOUR CLUSTERS ####

## Set up KMeans Object
kmeans_4_object = KMeans(n_clusters=4, max_iter=100)  
  
## Run KMeans on the clean quant data frame
kmeans = kmeans_4_object.fit(df)
kmeans_labels_4 = kmeans.labels_
cluster_centers_4 = kmeans.cluster_centers_

print(kmeans_labels_4[:1000])

## Make Data Float type
for j in range(len(cluster_centers_4)):
    for i in range(len(cluster_centers_4[0])):
        print(f"{columns[i]} :", float(cluster_centers_4[j][i]))
    print("\n")
    
pred_kmeans = kmeans_4_object.predict(new_emmissions)
print(pred_kmeans)

def createscatter(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(CO, CO2, BC, marker="v", color='red', alpha=.25, facecolor="none")

    ax.set_xlabel('CO')
    ax.set_ylabel("CO2")
    ax.set_title("Satellite Emissons Chart")
    ax.invert_xaxis()
    
    for j in range(len(clusters)):
        cluster_CO   = clusters[j][14]  
        cluster_CO2  = clusters[j][15]  
        cluster_BC   = clusters[j][16]  
        ax.scatter(cluster_CO, cluster_CO2, cluster_BC, 
                   marker="X", color="black", s=50)
    plt.show()

createscatter(cluster_centers_3)
createscatter(cluster_centers_4)

## 3D Visualizationwith 4 clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for i in range(len(df)):
    if kmeans_labels_4[i] == 3:
        ax.scatter(CO[i], CO2[i], BC[i], marker='s', color='blue', 
                     facecolors='none', alpha=.25)
    elif kmeans_labels_4[i] == 2:
        ax.scatter(CO[i], CO2[i], BC[i], marker='d', color='red', 
                     facecolors='none', alpha=.25)
    elif kmeans_labels_4[i] == 1:
        ax.scatter(CO[i], CO2[i], BC[i], marker='v', color='yellow', 
                     facecolors='none', alpha=.25)
    else:
        ax.scatter(CO[i], CO2[i], BC[i], marker='^', color='purple', 
                     facecolors='none', alpha=.25)
    

for j in range(len(cluster_centers_4)):
    cluster_mass   = cluster_centers_4[j][14]  # Mass
    cluster_price  = cluster_centers_4[j][15]  # Price
    cluster_cost   = cluster_centers_4[j][16]  # Cost
    ax.scatter(cluster_mass, cluster_price, cluster_cost, 
               marker="X", color="black", s=50)
ax.invert_xaxis()
#ax.view_init(elev=15, azim=75, roll=0)
mpl.rcParams["axes3d.mouserotationstyle"] = 'arcball'
plt.show()
