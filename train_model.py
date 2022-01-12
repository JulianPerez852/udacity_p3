# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

# Add the necessary imports for the starter code.

data=pd.read_csv('data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X, y, encoder, lb = process_data(
data, categorical_features=cat_features, label=" salary", training=True
)

kf = KFold(n_splits=5)
kf.get_n_splits(X)
KFold(n_splits=5, random_state=None, shuffle=False)


best_precision=0
best_model=""
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index] 
        
    model=train_model(X_train, y_train)
    
    preds=model.predict(X_test)
    
    precision, recall, fbeta=compute_model_metrics(y_test, preds)
    
    print("Precision: ",precision,", Recall: ", recall, ", fbeta: ", fbeta)
    
    if precision > best_precision:
        best_precision=precision
        best_model=model

filename = 'ml/finalized_model.sav'
pickle.dump(best_model, open(filename, 'wb'))


def slice_data(df, cat_features, encoder, lb, model):
    """ Function for calculating descriptive stats on slices of the dataset."""
    for cat in cat_features:
        for cls in df[cat].unique():
            df_temp = df[df[cat] == cls]
            
            X_temp, y_temp, encoder1, lb1 = process_data(
            df_temp, categorical_features=cat_features, label=" salary", training=False, encoder=encoder, lb=lb,
            )
            
            preds=model.predict(X_temp)
    
            precision, recall, fbeta=compute_model_metrics(y_temp, preds)
        
            with open("ml/slice_output.txt", 'a') as f:
                f.write("\nCategory: "+ cat +", "+ cls+"\n")
                f.write(" -Precision: " + str(precision)+"\n")
                f.write(" -Recall: " + str(recall)+"\n")
                f.write(" -Fbeta: " + str(fbeta)+"\n")


slice_data(data,cat_features,encoder,lb,best_model)

# Proces the test data with the process_data function.

# Train and save a model.


