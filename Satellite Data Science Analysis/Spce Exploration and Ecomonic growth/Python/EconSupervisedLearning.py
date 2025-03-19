#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:27:51 2025

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

filename = "/Users/wileyjones/Desktop/CS332/Project Files/Spce Exploration and Ecomonic growth/Satelitte_Cost_Clean.csv"
df = pd.read_csv(filename)

df["max_cost"] = df["price_per_kg"] * df["max_mass"]
print(df)

df = df.dropna()

label = df["Class"]

## Drop all non-numeric columns
df = df.select_dtypes(include="number")
df = df.drop(columns=["Unnamed: 0"])
print(df)

df.insert(0, "LABEL", label)
print(df.head(20))

print(df["LABEL"].value_counts())

## Show Data Imbalance
ax = sns.countplot(df, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Label Entries in Dataset")
plt.show()


######## UNDER SAMPLING #####################################################

## Undersample Majority Class
rus = RandomUnderSampler()

label = df["LABEL"].tolist()
df = df.drop(columns="LABEL")

df_resam, label_resam = rus.fit_resample(df, label)
print(df_resam)

df_resam.insert(0, "LABEL", label_resam)

## Show Data Imbalance
ax = sns.countplot(df_resam, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Label Entries in Dataset")
plt.show()

###### TRAIN TEST SPLIT #####################################################

train, test = train_test_split(df_resam)
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

####### VISUALIZATIONS #######################################################

mass = sns.displot(train, x="max_mass", kde=True)
plt.suptitle("Max Mass Values in Training Data")
plt.show()

train_count = train_labels.value_counts().tolist()
train_count
pie_labels = ["Heavy", "Small", "Meduim"]

fig, ax = plt.subplots()
ax.pie(train_count, labels=pie_labels, autopct='%1.1f%%')
plt.title("Value Counts for Training Labels")
plt.show()

cost = sns.jointplot(
    train, 
    x='max_cost', 
    y='price_per_kg'
    )
cost.fig.suptitle("Max Cost vs Price Per KG")
plt.show()

####### DECISION TREES ########################################################

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
    fontsize=11)
plt.show()
    
plt.savefig("Econ_Tree", dpi=500)
plt.close()

## PREDICTION 
pred = DT.predict(test)
print(pred)

## Confustion Matrix ##

act = test_labels.tolist()
conf_mat = confusion_matrix(act, pred)
print(conf_mat)

##Create the fancy CM using Seaborn
sns.heatmap(conf_mat, annot=True, cmap='Greens', xticklabels=names, yticklabels=names, cbar=False, fmt='d')
plt.title("Confusion Matrix For Rocket Ecomonics",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()


