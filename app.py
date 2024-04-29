import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, json

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

# Create the api end point for prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    print(data)
    # Convert JSON to DataFrame
    data_df = pd.DataFrame([data])
    transformed_data = preprocessor.transform(data_df)
    print(transformed_data.squeeze())
    output = model.predict(transformed_data.squeeze().reshape(1, -1))
    print(output[0])
    prediction = ""
    if output[0] == 1:
        prediction = "Diabetes"
    else:
        prediction = "Clear"
    return jsonify(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    print(data)
    data_df = pd.DataFrame(list(data.items(1)), columns=['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'])
    transformed_data = preprocessor.transform(data_df)
    output = model.predict(transformed_data.squeeze().reshape(1, -1))[0]
    return render_template("home.html", prediction_text=f"The result is {output}")

if __name__ == "__main__":
    app.run(debug=True)