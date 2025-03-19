#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:20:44 2025

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

## Read File into Dataframe 
filename = "/Users/wileyjones/Desktop/CS332/Project Files/UCS/UCS_DB_Cleaned.csv"
df = pd.read_csv(filename)

## Save Label
LABEL = df["Class of Orbit"].tolist()
print(LABEL)

## Drop all non-numeric columns
df = df.select_dtypes(include="number")
df = df.drop(columns=["Unnamed: 0", "Year of Launch"])
print(df)

## Add Label back on
df.insert(0, "LABEL", LABEL)
print(df)

########### UNDER SAMPLING ##################################################

## Show Data Imbalance
ax = sns.countplot(df, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

## Make DF with only LEO and GRO 
df_under = df[df["LABEL"].isin(["LEO", "GEO"])].reset_index(drop=True)
print(df_under)
df_under["LABEL"].value_counts()

## Make DF with only MEO and Ellipitcal
df_over = df[df["LABEL"].isin(["MEO", "Elliptical"])].reset_index(drop=True)
print(df_over)

## Undersample Majority Class
rus = RandomUnderSampler()

# Save Label and drop it
labels_under = df_under["LABEL"].tolist()
df_under = df_under.drop(columns="LABEL")
print(df_under)

df_resam, label_resam = rus.fit_resample(df_under, labels_under)

df_resam.insert(0, "LABEL", label_resam)
print(df_resam)

## Remerge the data 
df_merged = pd.concat([df_resam, df_over], ignore_index=True)
print(df_merged)

## Show new bar plot to get ready for over sampling
ax = sns.countplot(df_merged, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
plt.show()
#############################################################################

########### OVER SAMPLING ####################################################

## Create Over Sampler Object
ros = RandomOverSampler()

## Save Labels
labels_merged = df_merged["LABEL"].tolist()
df_merged = df_merged.drop(columns=["LABEL"])
print(df_merged)

## Over Sample 
merged_resam, merged_labels = ros.fit_resample(df_merged, labels_merged)
merged_resam.insert(0, "LABEL", merged_labels)
print(merged_resam)

## Show Balanced data
ax = sns.countplot(merged_resam, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

#############################################################################

########## TRAIN TEST SPLIT #################################################

train, test = train_test_split(merged_resam)
print("The Training Data is \n", train)
print("The Testing Data is \n ", test)

train_labels = train["LABEL"]
test_labels = test["LABEL"]

print(train_labels)
print(test_labels)

train = train.drop(columns=["LABEL"])
test = test.drop(columns=["LABEL"])

print(train, test)

#############################################################################

########## VISUALIZE DATA ###################################################

lm = sns.pairplot(
    train, 
    x="Expected Lifetime (yrs.)", 
    y="Launch Mass (kg.)",
    kind="reg"
    )
plt.show()

corr_train = train.corr()
print(corr_train)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_train, dtype=bool))

# Color Pallette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

## Heat Map for correlation chart
sns.heatmap(corr_train, mask=mask, cmap=cmap)
plt.show()

## Kernel Density Estimation 
sns.displot(
    train, 
    x="Period (minutes)",
    hue="",       
    kind='kde')
plt.show()

train.columns

life = sns.regplot(
    train,
    x="Period (minutes)",
    y="Perigee (km)",
    )
plt.show()

######## DECISION TREES #####################################################

### TRAINING ######
DT = DecisionTreeClassifier(max_depth=3)
DT = DT.fit(train, train_labels)

print(DT.classes_)

names = DT.classes_
print(type(names))

fig, ax = plt.subplots(figsize=(12,8))
annotations = tree.plot_tree(
    DT,
    feature_names=train.columns,
    class_names=DT.classes_,
    filled=True,
    ax=ax,
    fontsize=10)
plt.show()
    
plt.savefig("UCS_Tree", dpi=500)
plt.close()

### TESTING ###

pred = DT.predict(test)
print(pred)

## Confustion Matrix ##

act = test_labels.tolist()
conf_mat = confusion_matrix(act, pred)
print(conf_mat)

##Create the fancy CM using Seaborn
sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=names, yticklabels=names, cbar=False, fmt='d')
plt.title("Confusion Matrix For Satellite Orbit Classification",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

















