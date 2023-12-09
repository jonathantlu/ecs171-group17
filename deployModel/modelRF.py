
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# import data file
data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# split data into training and testing 
train, test = train_test_split(data, test_size=0.2, random_state=21)
X_train, y_train = train.drop(columns = ['Diabetes_binary']), train['Diabetes_binary']
X_test, y_test = test.drop(columns = ['Diabetes_binary']), test['Diabetes_binary']
assert X_train.shape[0] == len(y_train)

# create model
model = RandomForestClassifier(n_estimators=70, max_depth=13, min_samples_split=30, 
                               min_samples_leaf=10, max_features=5)


# fit model 
model.fit(X_train, y_train)




# make pickle file of our model
pickle.dump(model, open("modelRF.pkl", "wb"))