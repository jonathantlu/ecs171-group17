import pandas as pd
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# import data file
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# split data into training and testing 
train, test = train_test_split(df, test_size=0.2, random_state=21)
X_train, y_train = train.drop(columns = ['Diabetes_binary']), train['Diabetes_binary']
X_test, y_test = test.drop(columns = ['Diabetes_binary']), test['Diabetes_binary']
x_train_array = np.array(X_train)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# create model
clf = LogisticRegression()
clf.fit(X_train, y_train)



# make pickle file of our model
pickle.dump(clf, open("modelLogR.pkl", "wb"))