from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import inference
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    #os.system("rm -r .dvc .apt/usr/lib/dvc")

class Item(BaseModel):
    age: int
    workclass: str
    fnlgt:int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
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
        }    


def change_names(obj):
    obj_change={
        "Unnamed: 0": 0,
        " age": [obj.age],
        " workclass": [obj.workclass],
        " fnlgt": [obj.fnlgt],
        " education": [obj.education],
        " education-num": [obj.education_num],
        " marital-status": [obj.marital_status],
        " occupation": [obj.occupation],
        " relationship": [obj.relationship],
        " race": [obj.race],
        " sex": [obj.sex],
        " capital-gain": [obj.capital_gain],
        " capital-loss": [obj.capital_loss],
        " hours-per-week": [obj.hours_per_week],
        " native-country": [obj.native_country]
    }
    return obj_change

def prepare_data(obj):

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

    dataframe=pd.DataFrame(obj)
    X_temp, y_temp, encoder1, lb1 = process_data(
            dataframe, categorical_features=cat_features, training=False, encoder=loaded_encoder, lb=loaded_lb,
            )

    return X_temp

#Load Models
filename="ml/finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))

filename="ml/OneHotEncoder.sav"
loaded_encoder = pickle.load(open(filename, 'rb'))

filename="ml/LabelBinarizer.sav"
loaded_lb = pickle.load(open(filename, 'rb'))

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    welcome_message="Welcome to the api for the course machine learning devops - module 3"
    return welcome_message

@app.post("/predict/")
async def create_item(item: Item):

    #add the column unnamed
    obj=change_names(item)

    X=prepare_data(obj)
    pred=inference(loaded_model,X)
    tag=loaded_lb.inverse_transform(pred[0])
    response_text="This person gain in one year " + tag[0]
    return response_text