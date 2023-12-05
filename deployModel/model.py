import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 

# split data into training and testing 
train, test = train_test_split(df, test_size=0.2, random_state=21)
X_train, y_train = train.drop(columns = ['Diabetes_binary']), train['Diabetes_binary']
X_test, y_test = test.drop(columns = ['Diabetes_binary']), test['Diabetes_binary']
x_train_array = np.array(X_train)
assert X_train.shape[0] == len(y_train)

# create model
clf = SGDClassifier(alpha=0.245, class_weight='balanced', early_stopping=True,
              loss='modified_huber', n_jobs=3, validation_fraction=0.2)

# fit model 
clf.fit(X_train, y_train)




# make pickle file of our model
pickle.dump(clf, open("model.pkl", "wb"))