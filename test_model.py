import pytest
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference


def test_process_data():
    df=pd.read_csv("data/census_clean_copy.csv")
    cat_features = [
    " workclass",
    " education",
    " marital-status",
    " occupation",
    " relationship",
    " race",
    " sex",
    " native-country",
    ]

    X, y, encoder, lb = process_data(
    df, categorical_features=cat_features, label=" salary", training=True
    )
    assert df.shape[1] < len(X[0])


def test_compute_model_metrics():
    filename = 'ml/finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    data=pd.read_csv('data/census_clean_copy.csv')

    cat_features = [
    " workclass",
    " education",
    " marital-status",
    " occupation",
    " relationship",
    " race",
    " sex",
    " native-country",
    ]

    X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label=" salary", training=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    preds=model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test,preds)

    assert precision > 0


def test_inference():
    filename = 'ml/finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    data=pd.read_csv('data/census_clean_copy.csv')

    cat_features = [
    " workclass",
    " education",
    " marital-status",
    " occupation",
    " relationship",
    " race",
    " sex",
    " native-country",
    ]

    X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label=" salary", training=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    preds=inference(model,X_test)

    assert len(preds) == len(y_test)