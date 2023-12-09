# ECS 171 Group 17 Diabetes Risk Predictor

The model `Diabetes Risk Predictor`can predict an individual's risk of 
diabetes and the likelhood of the prediction. Trained using a UC Irvine dataset,
the model utilizes `Logistic Regression`, `SVM`, `XgBoost`, and `Random Forest`
to make the prediction. 

## Project Description

1. `app.py` is a Python script that uses Flask to generates a web page application
used for the Diabetes Risk Prediction model. The application uses 4 models, 
 (Support Vector Machine, Random Forest, Logistic Regression, and XGBoost) to 
 make predictions and provides a classification based on the aggregated results.
    - These four machine learning models (SVM, RF, LogR, XGB) are loaded from 
    pickled files amd are stored in a list `models`.
    - There are two routes for defined in the application
        1. Home Route: displays the template from the HTML template `index.html`
        2. Predict Route: 
            - Retrieves values from the request using `Get`.
            - Converts values to integers and passes them into the stored models
            - Aggregates the predictions and probabilities from each model.
            - Calculates the average predictions and probabilities.
            - Generates a classification message based on the averages and 
            pushes the message to the index.html template.


2. `index.html` provides the structure of the web application for the diabetes
risk prediction model. The template displays a form for the users to input the 
individuals features needed for the prediction, as well as a button that 
triggers the `POST` request once all fields have been filled. Furthermore, the 
template displays tables that include further information that assists users 
with what categorical values to enter for certain fields. 

3. `modelLogR.py`is a python script which trains a logistic regression model 
based off of the dataset `diabetes_binary_5050split_health_indicators_BRFSS2015`.
The script uses an 80:20 split to split the data into training and testing data 
and scales the data using StandardScaler(). Once that is done, a logistic 
regression model is created and is trained using the training data. Finally, the
trained model is loaded into a pickle file to be used in the app.py

3. `modelRF.py`is a python script which trains a random forest model based off 
of the dataset `diabetes_binary_5050split_health_indicators_BRFSS2015`.The 
script uses an 80:20 split to split the data into training and testing data and 
creates a random forest model. The random forest model is trained using the 
training data is in then loaded into a pickle file to be used in the app.py

4. `modelSVM.py`is a python script which trains a SVM model based off 
of the dataset `diabetes_binary_5050split_health_indicators_BRFSS2015`.
Due to the large nature of the dataset as well as the accuracy not changing 
between different kernel function, we assume that the data is linearly separable.
Thus, to reduce time, a SGDClassifier model is used as the training time is 
significanlty faster than an SVC. 
The script uses an 80:20 split to split the data into training and testing data and 
creates the SGDClassifier. The SGDClassifier model is trained using the 
training data is in then loaded into a pickle file to be used in the app.py

5. `modelXBG.py`is a python script which trains a XGBoost model based off 
of the dataset `diabetes_binary_5050split_health_indicators_BRFSS2015`.The 
script uses an 80:20 split to split the data into training and testing data and 
creates a XGBoost model. The XGBoost model is trained using the 
training data is in then loaded into a pickle file to be used in the app.py

## How to Run

To run the program, make sure you have the following packages installed:
    1. Python 
    2. Sklearn (pip install scikit-learn)
    3. pickle  (pip install pickle)
    4. pandas   (pip install pandas)
    5. numpy    (pip install numpy)
    5. xgboost  (pip install xgboost)

Once confirm all packages are downloaded, next make sure that the pickle files 
(.pkl) have been generated for each of the models. If they have yet to be 
generated, run the model files (ex: modelLogR.py andmodelRF.py) to generate the
 pickle files. After the pickle files have been generated for all 4 models, run 
 `app.py`. This will prompt Flask to launch and create the application. To 
 access the application, follow the link generated in the command line 
`* Running on http://127.0.0.1:3000`, specifically following the
link `http://127.0.0.1:3000`.

Once you have followed the link, you will be brought to the webpage where to 
make a prediction, you have to fill out all of the 22 fields and then select 
`Predict`. The model will then generate and display the prediction as well as 
the probability of the prediction.


Here is a link to the video demo on what the application does:
https://www.youtube.com/watch?v=HdguFQyRlRU 


## Credit
Team Leader: William Vuong

Members: Xuanzhen Lao, Jonathan Lu, Minh Pham, and Brandon Huynh, 
