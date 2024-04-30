import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, json, send_from_directory

import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the preprocessor
with open('Models/data_preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)
model = pickle.load(open('Models/model.pkl', 'rb'))

# Create the home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('static/images', filename)

# Create the api end point for prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    # Convert JSON to DataFrame
    data_df = pd.DataFrame([data])
    transformed_data = preprocessor.transform(data_df)
    output = model.predict(transformed_data.squeeze().reshape(1, -1))
    prediction = ""
    if output[0] == 1:
        prediction = "Diabetes"
    else:
        prediction = "Clear"
    return jsonify(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    data = dict(request.form)
    data_df = pd.DataFrame([data])
    transformed_data = preprocessor.transform(data_df)
    output = model.predict(transformed_data.squeeze().reshape(1, -1))[0]
    if output == 1:
        prediction = "Diabetes"
    else:
        prediction = "Clear"
    return render_template("home.html", prediction_text=f"The result is {prediction}")

if __name__ == "__main__":
    app.run(debug=True)