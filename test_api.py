import pytest
import requests
import json

def test_get_method():
    path = "https://udacityp3.herokuapp.com/"
    response = requests.get(url=path)
    responseJson = json.loads(response.text)
    assert response.status_code == 200
    
def test_post_status_method():
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
    assert response.status_code == 200

def test_predict_method():
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
    assert response.text.replace('"',"") == "This person gain in one year  <=50K"

def test_predict_method_two():
    path = "https://udacityp3.herokuapp.com/predict"
    json={
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = requests.post(url=path, json=json)
    assert response.text.replace('"',"") == "This person gain in one year  >50K"