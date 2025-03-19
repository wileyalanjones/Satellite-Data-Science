#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:54:10 2025

@author: wileyjones
"""


##### LIBARIES #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
import seaborn as sns


#### KMEANS Satellite Costs ####

filename = "/Users/wileyjones/Desktop/CS332/Project Files/Satelitte_Cost_Clean.csv"
df = pd.read_csv(filename)

df = df.select_dtypes(include='number')
df = df.drop(columns="Unnamed: 0")

### Create new  
df["Cost"] = df["price_per_kg"] * df["max_mass"]

## Save for plots
Mass  = np.array(df["max_mass"].tolist())
Price = np.array(df['price_per_kg'].tolist())
Cost  = np.array(df['Cost'].tolist())

## 3D Visualization 
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Mass, Price, Cost, marker='^', color='green', 
                     facecolors='none', alpha=.25)
ax.invert_xaxis()

ax.set_xlabel('Mass')
ax.set_ylabel("Price Per KG")
ax.set_zlabel("Max Cost")
ax.set_title("Satellite Cost Chart")

plt.show()

## 2D Representation of Cost and Mass 
scatter = sns.regplot(df, x="max_mass", y="Cost")
plt.show()

##### 2 Clusters ######

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
        print(float(cluster_centers[j][i]))
        
## 3D Visualizationwith 2 clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Mass, Price, Cost, marker='^', color='green', 
                     facecolors='none', alpha=.1)
ax.invert_xaxis()

for j in range(len(cluster_centers)):
    cluster_mass   = cluster_centers[j][1]  # Mass
    cluster_price  = cluster_centers[j][0]  # Price
    cluster_cost   = cluster_centers[j][2]  # Cost
    ax.scatter(cluster_mass, cluster_price, cluster_cost, 
               marker="X", color="black", s=50)

ax.set_xlabel('Mass')
ax.set_ylabel("Price Per KG")
ax.set_zlabel("Max Cost")
ax.set_title("Satellite Cost Chart")

plt.show()

## Predict 2 KMeans
new_launches = [[8000, 3000, 24000000],
                [3000, 20000, 60000000]]
pred_kmeans = kmeans_2_object.predict(new_launches)
print(pred_kmeans)

##### 3 Clusters ######

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
        print(float(cluster_centers_3[j][i]))

## 3D Visualizationwith 3 clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Mass, Price, Cost, marker='^', color='green', 
                     facecolors='none', alpha=.1)

for j in range(len(cluster_centers_3)):
    cluster_mass   = cluster_centers_3[j][1]  # Mass
    cluster_price  = cluster_centers_3[j][0]  # Price
    cluster_cost   = cluster_centers_3[j][2]  # Cost
    ax.scatter(cluster_mass, cluster_price, cluster_cost, 
               marker="X", color="black", s=50)

ax.invert_xaxis()
ax.set_xlabel('Mass')
ax.set_ylabel("Price Per KG")
ax.set_zlabel("Max Cost")
ax.set_title("Satellite Cost Chart")

plt.show()

pred_kmeans = kmeans_3_object.predict(new_launches)
print(pred_kmeans)

##### 4 Clusters ######

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
        print(float(cluster_centers_4[j][i]))
        
## 3D Visualizationwith 4 clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for i in range(len(df)):
    if kmeans_labels_4[i] == 3:
        ax.scatter(Mass[i], Price[i], Cost[i], marker='s', color='blue', 
                     facecolors='none', alpha=.1)
    elif kmeans_labels_4[i] == 2:
        ax.scatter(Mass[i], Price[i], Cost[i], marker='d', color='red', 
                     facecolors='none', alpha=.1)
    elif kmeans_labels_4[i] == 1:
        ax.scatter(Mass[i], Price[i], Cost[i], marker='v', color='yellow', 
                     facecolors='none', alpha=.1)
    else:
        ax.scatter(Mass[i], Price[i], Cost[i], marker='^', color='purple', 
                     facecolors='none', alpha=.1)

for j in range(len(cluster_centers_4)):
    cluster_mass   = cluster_centers_4[j][1]  # Mass
    cluster_price  = cluster_centers_4[j][0]  # Price
    cluster_cost   = cluster_centers_4[j][2]  # Cost
    ax.scatter(cluster_mass, cluster_price, cluster_cost, 
               marker="X", color="black", s=50)
    
ax.invert_yaxis()
ax.invert_zaxis()
ax.set_xlabel('Mass')
ax.set_ylabel("Price Per KG")
ax.set_zlabel("Max Cost")
ax.set_title("Satellite Cost Chart")

plt.show()

## 4 Clusters Predictions
pred_kmeans = kmeans_4_object.predict(new_launches)
print(pred_kmeans)
