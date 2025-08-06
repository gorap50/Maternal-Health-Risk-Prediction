# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The Result of Liver disease Is".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)

from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

def riskPredict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
    print("length = ",len(to_predict_list))
    loaded_model = joblib.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route("/predict",methods = ["POST"])
def predict():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    pred = riskPredict(to_predict_list) 
    

    if pred == 1:
        vr = "Low Risk"
    elif pred==0:
        vr="High Risk"
    elif pred==2:
        vr="Mid Risk"
   
    return render_template('index.html',prediction_text = vr)


if __name__ == '__main__':
    app.run()