#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:11:15 2025

@author: wileyjones
"""

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

####### IMPORT FILE #######################################################
filename = "/Users/wileyjones/Desktop/CS332/Project Files/Emissions/2.0_emissions.csv"
df = pd.read_csv(filename)

## Save Label
label = df["LABEL"]

## Only numeric types
df = df.select_dtypes(include="number")
df = df.drop(columns=["Unnamed: 0", "Longitude", "Latitude"])
print(df.columns)

df.insert(0, "LABEL", label)
print(df.head(20))

print(df["LABEL"].value_counts())

## Show Data Imbalance
ax = sns.countplot(df, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Label Entries in Dataset")
plt.show()

####### OVERSAMPLING ########################################################

## Create Over Sampler Object
ros = RandomOverSampler()

## Save Labels
labels = df["LABEL"].tolist()
df = df.drop(columns=["LABEL"])
print(df)

## Over Sample 
resam, re_labels = ros.fit_resample(df, labels)
resam.insert(0, "LABEL", re_labels)
print(resam)

## Show Balanced data
ax = sns.countplot(resam, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

###### TRAIN TEST SPLIT #####################################################

train, test = train_test_split(resam)
print("The Training Data is \n", train)
print("The Testing Data is \n ", test)

train_labels = train["LABEL"]
test_labels = test["LABEL"]

print("The Training Labels are \n", train_labels)
print("The Testing Labels are \n", test_labels)

train = train.drop(columns=["LABEL"])
test = test.drop(columns=["LABEL"])

train_labels.value_counts()

print(train, test)

####### VISUALIZATION #######################################################

corr_train = train.corr()
print(corr_train)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_train, dtype=bool))

# Color Pallette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

## Heat Map for correlation chart
sns.heatmap(corr_train, mask=mask, cmap=cmap)
plt.title("Correlation Matrix for Training Data")
plt.show()

sns.displot(train, 
            x="CO2", 
            hue="Booster_No",
            kind="kde",
            fill=True)
plt.title("Kernel Density Estimation for CO2")
plt.show()

ax = sns.displot(train, x="CO", hue="Booster_No", stat="density", kde=True)
plt.title("Density Estimation for CO")
plt.show()

####### DECISION TREES ########################################################

### TRAINING ######
DT = DecisionTreeClassifier(max_depth=4)
DT = DT.fit(train, train_labels)

print(type(DT.classes_[0]))
print(DT)
names = DT.classes_
print(type(names))

names = [str(DT.classes_[i] for i in range(2))]

fig, ax = plt.subplots(figsize=(13,8))
annotations = tree.plot_tree(
    DT,
    feature_names=train.columns,
    class_names=names,
    filled=True,
    ax=ax,
    fontsize=11)
plt.show()
    
plt.savefig("Emission_Tree", dpi=500)
plt.close()

## PREDICTION 
pred = DT.predict(test)
print(pred)

## Confustion Matrix ##

act = test_labels.tolist()
conf_mat = confusion_matrix(act, pred)
print(conf_mat)

##Create the fancy CM using Seaborn
sns.heatmap(conf_mat, annot=True, cmap='Reds', xticklabels=names, yticklabels=names, cbar=False, fmt='d')
plt.title("Confusion Matrix For Rocket Emissions",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()
