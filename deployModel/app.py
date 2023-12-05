import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

# create flask app
app = Flask(__name__)

# model pickle model
modelSVM = pickle.load(open("model.pkl", "rb"))



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
    pred_list = modelSVM.predict(features)
    prob_list= modelSVM.predict_proba(features)
    prediction = pred_list[0]
    probability = prob_list[0][0] * 100
    
    match prediction:
        case 0.0:
            classification = 'You have %.2f%% liklihood that you are Negative for diabetes' % (probability)
        case 1.0:
            classification = 'You have a %.2f%% liklihood that you are Positive for diabetes' % (probability)
    

    return render_template('index.html', prediction_text = classification)


if __name__ == "__main__":
    app.run(port=3000, debug=True)