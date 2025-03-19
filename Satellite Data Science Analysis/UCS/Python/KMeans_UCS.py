#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:58:38 2025

@author: wileyjones
"""

##### LIBARIES #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.ticker import LinearLocator


#### KMEANS UCS DATABASE ####

## Reading file to Pandas 
filename = "/Users/wileyjones/Desktop/CS332/Project Files/UCS/UCS_DB_Cleaned.csv"
UCS = pd.read_csv(filename)

## Saving Label of Data
LABEL = UCS["Class of Orbit"].tolist()

## Get only quantitative data
UCS = UCS.select_dtypes(include="number")
UCS = UCS.drop(columns="Unnamed: 0")
UCS = UCS.drop(columns='Year of Launch')
print(UCS.head(10))

## 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

Period = np.array(UCS["Period (minutes)"].tolist())
Mass = np.array(UCS["Launch Mass (kg.)"].tolist())
Lifetime = np.array(UCS["Expected Lifetime (yrs.)"].tolist())

surface = ax.scatter(Mass, Period, Lifetime, marker='o', color='blue', 
                     facecolors='none', alpha=.25)

ax.set_xlabel('Mass')
ax.set_ylabel("Period")
ax.set_zlabel("Lifetime")
ax.set_title("UCS 3D Visualization")

plt.show()

## 3D Inverted Axis
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

Period = np.array(UCS["Period (minutes)"].tolist())
Mass = np.array(UCS["Launch Mass (kg.)"].tolist())
Lifetime = np.array(UCS["Expected Lifetime (yrs.)"].tolist())

surface = ax.scatter(Mass, Period, Lifetime, marker='o', color='blue', 
                     facecolors='none', alpha=.25)

ax.set_xlabel('Mass')
ax.set_ylabel("Period")
ax.set_zlabel("Lifetime")
ax.set_title("UCS 3D Visualization")

ax.invert_xaxis()

plt.show()

##### 2 Clusters ######

## Set up KMeans Object
kmeans_2_object = KMeans(n_clusters=2, max_iter=100)

## Run KMeans on the clean quant data frame
kmeans = kmeans_2_object.fit(UCS)
kmeans_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

## Run through predictions now
prediction_kmeans = kmeans_2_object.predict(UCS)
print(prediction_kmeans[0:1000])

## Make Data Float type
for j in range(2):
    for i in range(len(cluster_centers[0])):
        print(float(cluster_centers[j][i]))

## 3D Visualizationwoth 2 Clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

Period = np.array(UCS["Period (minutes)"].tolist())
Mass = np.array(UCS["Launch Mass (kg.)"].tolist())
Lifetime = np.array(UCS["Expected Lifetime (yrs.)"].tolist())

surface = ax.scatter(Mass, Period, Lifetime, marker='o', color='blue', 
                     facecolors='none', alpha=.01)

for j in range(len(cluster_centers)):
    cluster_period   = cluster_centers[j][4]  # Period
    cluster_mass     = cluster_centers[j][5]  # Mass
    cluster_lifetime = cluster_centers[j][6]  # Lifetime
    ax.scatter(cluster_mass, cluster_period, cluster_lifetime, 
               marker="X", color="black", s=50)

ax.set_xlabel('Mass')
ax.set_ylabel("Period")
ax.set_zlabel("Lifetime")
ax.set_title("UCS 3D Visualization")

ax.invert_xaxis()

plt.show()

## Predict 2 KMeans
new_satelite = [[1000, 1050, .1, 35, 5000, 1000, 6],
                [100, 150, .6, 57, 90, 200, 2]]
pred_kmeans = kmeans_2_object.predict(new_satelite)
print(pred_kmeans)

#### THREE CLUSTERS ####

## Set up KMeans Object
kmeans_3_object = KMeans(n_clusters=3, max_iter=100)

## Run KMeans on the clean quant data frame
kmeans_3 = kmeans_3_object.fit(UCS)
kmeans_labels_3 = kmeans_3.labels_
cluster_centers_3 = kmeans_3.cluster_centers_

## Run through predictions now
prediction_kmeans_3 = kmeans_2_object.predict(UCS)

print(kmeans_labels_3[0:1000])

## Make Data Float type
for j in range(len(cluster_centers_3)):
    for i in range(len(cluster_centers_3[0])):
        print(float(cluster_centers_3[j][i]))

## 3D Visualizationwoth 3 Clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

Period = np.array(UCS["Period (minutes)"].tolist())
Mass = np.array(UCS["Launch Mass (kg.)"].tolist())
Lifetime = np.array(UCS["Expected Lifetime (yrs.)"].tolist())

surface = ax.scatter(Mass, Period, Lifetime, marker='o', color='blue', 
                     facecolors='none', alpha=.1)

for j in range(len(cluster_centers_3)):
    cluster_period   = cluster_centers_3[j][4]  # Period
    cluster_mass     = cluster_centers_3[j][5]  # Mass
    cluster_lifetime = cluster_centers_3[j][6]  # Lifetime
    ax.scatter(cluster_mass, cluster_period, cluster_lifetime, 
               marker="X", color="black", s=50)

ax.set_xlabel('Mass')
ax.set_ylabel("Period")
ax.set_zlabel("Lifetime")
ax.set_title("UCS 3D Visualization")

ax.invert_xaxis()

plt.show()

## Predict 2 KMeans
new_satelite = [[1000, 1050, .1, 35, 5000, 1000, 6],
                [100, 150, .6, 57, 90, 200, 2]]
pred_kmeans = kmeans_3_object.predict(new_satelite)
print(pred_kmeans)

#### FOUR CLUSTERS ####

## Set up KMeans Object
kmeans_4_object = KMeans(n_clusters=4, max_iter=100)

## Run KMeans on the clean quant data frame
kmeans_4 = kmeans_4_object.fit(UCS)
kmeans_labels_4 = kmeans_4.labels_
cluster_centers_4 = kmeans_4.cluster_centers_

## Run through predictions now
prediction_kmeans_4 = kmeans_2_object.predict(UCS)

print(kmeans_labels_4[0:1000])

## Make Data Float type
for j in range(len(cluster_centers_4)):
    for i in range(len(cluster_centers_4[0])):
        print(float(cluster_centers_4[j][i]))
    print("\n")
    
## 3D Visualizationwoth 4 Clusters
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

Period = np.array(UCS["Period (minutes)"].tolist())
Mass = np.array(UCS["Launch Mass (kg.)"].tolist())
Lifetime = np.array(UCS["Expected Lifetime (yrs.)"].tolist())

surface = ax.scatter(Mass, Period, Lifetime, marker='o', color='blue', 
                     facecolors='none', alpha=.01)

for j in range(len(cluster_centers_4)):
    cluster_period   = cluster_centers_4[j][4]  # Period
    cluster_mass     = cluster_centers_4[j][5]  # Mass
    cluster_lifetime = cluster_centers_4[j][6]  # Lifetime
    ax.scatter(cluster_mass, cluster_period, cluster_lifetime, 
               marker="X", color="black", s=50)

ax.set_xlabel('Mass')
ax.set_ylabel("Period")
ax.set_zlabel("Lifetime")
ax.set_title("UCS 3D Visualization")

ax.invert_xaxis()

plt.show()

## Predict 4 KMeans
new_satelite = [[1000, 1050, .1, 35, 5000, 1000, 6],
                [100, 150, .6, 57, 90, 200, 2]]
pred_kmeans = kmeans_4_object.predict(new_satelite)
print(pred_kmeans)