import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the LightGBM model
model = pickle.load(open("model.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data as a dictionary
    data = request.get_json()

    # Convert the data to a numpy array
    features = np.array(list(data.values())).reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict(features)[0]

    # Convert the prediction to a dictionary and return as JSON
    output = {'prediction': int(prediction)}
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)