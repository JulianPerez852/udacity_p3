# Model Details:
## Julián David Pérez Hincapipé created the model. It is a Random Forest classifier using the default hyperparameters in scikit-learn 1.0.2
## Intended Use:
### Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataset.
## Metrics
### The model was evaluated using Accuracy Recall and FBeta with 0.77 , 0.64 and 0.68 respectively
## Data
### The data was obtanied from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/census+income). Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) Prediction task is to determine whether a person makes over 50K a year.
## Bias
### This model represent a lot of american, white, men, so the data set is unbalanced in this terms and it is posible that the data set had bias so the model is not 100% reliable
## Ethical Considerations
### Is important have a balanced data, so the model can have a lot of bias when it try to predict new data, if the data is bad the model can show not the reality, by the naturality of the data in this case it is not such a problem, but in models about races, countrys, gender or other sensible topics this can be a serius problem. 
## Caveats & Recommendations
### This data and this model only will be used for training purposes, non for production environments, take care with that, the model is maybe overfiting with some clases because the data is not adecuaded balanced, you can check info about the data in all the posible combinations and filters in ml/slice_output.txt