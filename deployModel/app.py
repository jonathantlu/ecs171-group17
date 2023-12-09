import numpy as np
from flask import Flask, render_template, request
import pickle

# create flask app
app = Flask(__name__)

# pickle model
models = []
models.append(pickle.load(open("modelSVM.pkl", "rb")))
models.append(pickle.load(open("modelRF.pkl", "rb")))
models.append(pickle.load(open("modelLogR.pkl", "rb")))
models.append(pickle.load(open("modelXGB.pkl", "rb")))


# home
@app.route('/')
def Home():
    return render_template("index.html")


# prediction
@app.route('/predict', methods=['POST'])
def predict():
    # obtains input values 
    form_values = request.form.values()

    # converts values to int and passes into model
    int_values = list([int(value) for value in form_values])
    features = [np.array(int_values)]

    # initializes total variables
    total_pred = 0
    total_prob_neg = 0
    total_prob_pos = 0

    # runs predictions
    for model in models:
        predition = model.predict(features)
        print(f"predicition: {predition}")
        probablilty = model.predict_proba(features)
        print(f"probability: {probablilty}")
        total_pred = total_pred + predition[0]
        total_prob_neg = total_prob_neg + probablilty[0][0]*100
        total_prob_pos = total_prob_pos + probablilty[0][1]*100

    # finds average predictions and probabilities
    avg_pred = total_pred/len(models)
    avg_prob_neg = total_prob_neg/len(models)
    avg_prob_pos = total_prob_pos/len(models)
    
    # creates classification
    if avg_pred < 0.75:
        classification = 'You have an average likelihood of %.2f%% that you are not at risk for diabetes (Negative)' % (avg_prob_neg)
    elif avg_pred >= 0.75:
        classification = 'You have an average likelihood of %.2f%% that you are at risk for diabetes (Positive)' % (avg_prob_pos)

    return render_template('index.html', prediction_text = classification)


if __name__ == "__main__":
    app.run(port=3000, debug=True)