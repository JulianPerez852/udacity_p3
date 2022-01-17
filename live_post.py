import requests
import json

path = "https://udacityp3.herokuapp.com/predict"
json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
}

response = requests.post(url=path, json=json)

print(f"The status code is: {response.status_code}")

print(f"The response is: {response.text}")