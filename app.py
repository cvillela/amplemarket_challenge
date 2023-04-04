import pickle
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import torch
import argparse

from constants import specialty_labels, category_labels, industry_labels, tech_labels, tech_stack_dict

app = Flask(__name__)


def preprocess(data):
    # data to df - preprocessing made easier
    df = pd.DataFrame.from_dict(data, orient='columns')

    # categorical variables
    # funding, events, hubs
    df['has_funding'] = 0
    df.loc[df['total_funding'] > 0, 'has_funding'] = 1

    df['has_events'] = 0
    df.loc[df['events'].notnull(), 'has_events'] = 1

    df['has_hubs'] = 0
    df.loc[df['company_hubs'].notnull(), 'has_hubs'] = 1

    df = df.drop(columns=['name', 'alexa_rank', 'city', 'state', 'country', 'hq', 'website',
                          'linkedin_url', 'overview', 'sic_codes', 'company_hubs', 'events'])

    # size
    df['size'] = df['size'].str.replace(r'[^0-9\-]+', '')
    df.loc[df['size'] == '10001', 'size'] = '500+'
    df.loc[df['size'] == '5001-10000', 'size'] = '500+'
    df.loc[df['size'] == '1001-5000', 'size'] = '500+'
    df.loc[df['size'] == '501-1000', 'size'] = '500+'
    df.loc[df['size'] == '201-500', 'size'] = '201-500'
    df.loc[df['size'] == '0-1', 'size'] = '1-10'
    df.loc[df['size'] == '2-10', 'size'] = '1-10'

    # predefine categories so the one-hot encoded input is the same as in model training
    df['size'] = pd.Categorical(df['size'], categories=['1-10', '11-50', '51-200', '201-500', '500+'])

    # ownership type
    df.loc[df['ownership_type'] == 'Nonprofit', 'ownership_type'] = 'Non Profit'

    df.loc[(df['ownership_type'] == 'Self-Employed') |
           (df['ownership_type'] == 'Self Owned') |
           (df['ownership_type'] == 'Sole Proprietorship'), 'ownership_type'] = 'Self Employed'

    df.loc[(df['ownership_type'] == 'Government Agency') |
           (df['ownership_type'].isnull()) |
           (df['ownership_type'] == 'Non Profit') |
           (df['ownership_type'] == 'Educational'), 'ownership_type'] = 'Other'

    # predefine categories so the one-hot encoded input is the same as in model training
    df['ownership_type'] = pd.Categorical(df['ownership_type'],
                                          categories=['Partnership', 'Privately Held', 'Public Company',
                                                      'Self Employed', 'Other'])
    # one-hot-encoding
    cat_cols = ['ownership_type', 'size']

    for col in cat_cols:
        one_hot = pd.get_dummies(df[col])
        # Drop column B as it is now encoded
        df = df.drop(columns=[col], axis=1)
        # Join the encoded df
        df = df.join(one_hot)

    # standard scaling, using the same scaler from training
    scaler = joblib.load('models/num_std_scaler.joblib')
    cols_to_scale = ['employees_on_linkedin', 'followers', 'founded']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # preprocess text data using Transformers
    # This step is probably the slowest on the preprocessing pipeline, and could be optimized in the future

    df[industry_labels] = 0
    df[category_labels] = 0
    df[tech_labels] = 0
    df[specialty_labels] = 0

    if df['specialties'] is not None:
        cat = df['specialties']
        predicted_category = classifier(cat, candidate_labels=specialty_labels)
        for label, score in zip(predicted_category['labels'], predicted_category['scores']):
            df[label] = score

    if df['categories'] is not None:
        cat = df['categories']
        predicted_category = classifier(cat, candidate_labels=category_labels)
        for label, score in zip(predicted_category['labels'], predicted_category['scores']):
            df[label] = score

    if df['industry'] is not None:
        cat = df['industry']
        predicted_category = classifier(cat, candidate_labels=industry_labels)
        for label, score in zip(predicted_category['labels'], predicted_category['scores']):
            df[label] = score

    if df['technologies'] is not None:
        techs = df['technologies'].split(',')
        for tech in techs:
            if tech in tech_stack_dict.keys():
                for cat in tech_stack_dict[tech]:
                    df.loc[cat] = 1

    # scale text features
    cols_to_scale = category_labels + industry_labels + specialty_labels
    scaler = joblib.load('models/txt_std_scaler.joblib')
    cols_to_scale = ['employees_on_linkedin', 'followers', 'founded']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df_pp = df.drop(
        columns=['specialties', 'total_funding', 'technologies', 'categories', 'specialties', 'industry']).copy()

    return df_pp


@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data as a dictionary
    data = request.get_json()

    X = preprocess(data)

    # Make a prediction using the loaded model
    y_b2c = model_b2c.predict(X)
    print(y_b2c)

    y_b2b = model_b2b.predict(X)
    print(y_b2b)

    # Convert the prediction to a dictionary and return as JSON
    output = {'prediction_b2c': int(y_b2c), 'prediction_b2b': int(y_b2b) }
    return jsonify(output)


if __name__ == '__main__':

    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print('running on device = {}'.format(device_id))

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_pretrained_bart', type=str)
    args = parser.parse_args()
    if args.load_pretrained_bart:
        # simply load pretrained downloaded model.
        classifier = pipeline(task='zero-shot-classification', model='bart-classifier')
    else:
        # download BART model. This step may take long depending on computational capacity [1.63GB model] and it is
        # recomended to already have the BART model preinstalled on your machine
        classifier = pipeline(model="facebook/bart-large-mnli", device=device_id)

    # Load the LightGBM models
    model_b2b = pickle.load(open("models/b2b_model.pkl", "rb"))
    model_b2c = pickle.load(open("models/b2c_model.pkl", "rb"))

    app.run(port=5000, debug=True)

