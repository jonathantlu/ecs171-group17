import pandas as pd
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# import data file
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")


# split data into training and testing 
train, test = train_test_split(df, test_size=0.2, random_state=21)
X_train, y_train = train.drop(columns = ['Diabetes_binary']), train['Diabetes_binary']
X_test, y_test = test.drop(columns = ['Diabetes_binary']), test['Diabetes_binary']
x_train_array = np.array(X_train)

# create model
clf =  XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=2, min_child_weight=2,gamma=0.4, subsample=0.6, 
                     colsample_bytree=0.65, reg_alpha = 0.1, objective= 'binary:logistic',
                     nthread=4,scale_pos_weight=1,seed=27)

# fit model
clf.fit(X_train, y_train)



# make pickle file of our model
pickle.dump(clf, open("modelXGB.pkl", "wb"))