import requests
import json
import pandas as pd
import numpy as np

# load the dataframe
df = pd.read_csv('data.csv')

while True:
    # select a random row from the dataframe
    row = df.sample().iloc[0]
    actual_type = row['type']
    print('Using row {}'.format(row.name))
    print('Actual Type: {}'.format(actual_type))


    # convert the row to a dictionary
    data = row.drop(columns=['type']).to_dict()

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
