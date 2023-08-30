import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create flask app
app = Flask (__name__)

#load the pickle model
model = pickle.load(open("model1.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    return render_template("index.html",prediction_text="The college suggested is: {}".format(prediction))

if __name__ =="__main__":
    app.run(debug=True)
