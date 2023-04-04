import requests
import json
import pandas as pd
import random

# load the dataframe
df = pd.read_csv('data.csv')

# select a random row from the dataframe
row = df.sample().iloc[0]

# convert the row to a dictionary
data = row.to_dict()

# convert the dictionary to a JSON string
json_data = json.dumps(data)
print(json_data)

# set the URL for the Flask application
url = 'http://localhost:5000/predict'

# set the headers for the POST request
headers = {'Content-type': 'application/json'}

# send the POST request
response = requests.post(url, data=json_data, headers=headers)

# print the response from the Flask application
print(response.text)