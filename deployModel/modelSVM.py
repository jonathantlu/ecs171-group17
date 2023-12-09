import pandas as pd
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# import data file
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# split data into training and testing 
train, test = train_test_split(df, test_size=0.2, random_state=21)
X_train, y_train = train.drop(columns = ['Diabetes_binary']), train['Diabetes_binary']
X_test, y_test = test.drop(columns = ['Diabetes_binary']), test['Diabetes_binary']
x_train_array = np.array(X_train)

# create model
clf = SGDClassifier(alpha=0.245, class_weight='balanced', early_stopping=True,
              loss='modified_huber', n_jobs=3, validation_fraction=0.2)

# fit model 
clf.fit(X_train, y_train)




# make pickle file of our model
pickle.dump(clf, open("modelSVM.pkl", "wb"))