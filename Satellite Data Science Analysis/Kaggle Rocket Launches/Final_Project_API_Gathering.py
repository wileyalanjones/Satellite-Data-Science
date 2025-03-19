#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:13:06 2025

@author: wileyjones
"""

## Kaggle API Key
##{"username":"wileyalanjones","key":"e1ba1ef1524d589752d2f17bcf8cb92e"}

## Query example
## https://wileyalanjones:
    #e1ba1ef1524d589752d2f17bcf8cb92e@www.kaggle.com/api/v1/datasets/list?
        #search=space+missions 
 
# import modules for api 
from kaggle.api.kaggle_api_extended import KaggleApi
import requests

# connect to API and authenicate
api = KaggleApi()
api.authenticate()

# Save Endpoint and Key for get request
username = "wileyalanjones"
key = "e1ba1ef1524d589752d2f17bcf8cb92e"
End = "https://www.kaggle.com/api/v1/datasets/list"

# Query parameters
params = {"search": "space launches"}

# Authenticate and make the GET request
response = requests.get(End, params=params, auth=(username, key))

# Print json reading of response
datasets = response.json()
print(datasets)

# Print the titles and urls only
for dataset in datasets:
    print(f"Title: {dataset['title']}, URL: {dataset['url']}")
   
# Find out name of file in dataset to download   
dataset_id = "sefercanapaydn/mission-launches"
files = api.dataset_list_files(dataset_id)
for file in files.files:
    print(f'{file.name}')

# download the csv file
api.dataset_download_file('sefercanapaydn/mission-launches', 
                          'mission_launches.csv', path='.')

# Check that csv file is correct
with open('mission_launches.csv', 'r') as file:
    content = file.read()
    print(content)
