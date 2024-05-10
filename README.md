# Diabetes Risk Prediction
<img src='static/images/dataset-cover.jpeg' width="50%"/>

Prediction of diabetes based on the signs and symptoms using machine learning algorithms.

## Problem Definition

Diabetes mellitus, often known simply as diabetes, is a group of common endocrine diseases characterized by sustained high blood sugar levels. Diabetes mellitus is diagnosed with a test for the glucose content in the blood [[wiki]](https://en.wikipedia.org/wiki/Diabetes). The dataset provides records of some signs and symthoms for people who have and have not diabetes. 

This is a binary classification problem and our aim, in this study, is to come up with a supervised ML model which predicts diabetes for a person who provides the information with high accuracy. 

## Software And Tools Requirements
<img src='static/images/project.png' width="50%" />

- Flask
- sklearn
- pandas
- numpy
- uvicorn

To install the required libraries, run the following command:
```bash
pip install -r requirements.txt
```

### How to run it locally

#### Terminal
```bash
uvicorn main:asgi_app --port 10000
```

#### IDE
```bash
Create a python run configuration and choose the main.py file
```

#### Docker
1. Run `docker build -t diabetes-prediction .`
2. Run `docker run -p 10000:10000 diabetes-prediction`
3. Open your browser and go to `http://localhost:10000/`

#### Online
App is also available online: https://diabetes-risk-prediction.onrender.com/

Or as API: https://diabetes-risk-prediction.onrender.com/predict_api
```JSON
{
    "Gender": "Female",
    "Polyuria": "No",
    "Polydipsia": "No",
    "sudden weight loss": "No",
    "weakness": "Yes",
    "Polyphagia": "No",
    "Genital thrush": "No",
    "visual blurring": "Yes",
    "Itching": "Yes",
    "Irritability": "No",
    "delayed healing": "Yes",
    "partial paresis": "No",
    "muscle stiffness": "No",
    "Alopecia": "Yes",
    "Obesity": "No",
    "Age": 48
}
```

