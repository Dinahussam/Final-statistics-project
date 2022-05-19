from flask import Flask, redirect, url_for, request, render_template

import pickle
import numpy as np

model = pickle.load(open('Diabetes.classifier.pickle','rb'))
app = Flask(__name__, template_folder='templates')
@app.route('/')
def home():
    return render_template('home.html')


app.route('/predict', methods=['POST'])
def predict():
    Cholesterol = request.form['a']
    Glucose = request.form['b']
    hdl_chol = request.form['c']
    chol_hdl_ratio = request.form['d']
    age = request.form['e']
    gender = request.form['f']
    hight = request.form['g']
    weight = request.form['h']
    bmi = request.form['i']
    Systolic_bp = request.form['j']
    diastolic_bp = request.form['k']
    waist = request.form['l']
    hip = request.form['m']
    waist_hip_ratio = request.form['n']
    if (gender=="Male"):
        gender=1
    else:
        gender =0

    Cholesterol = int(Cholesterol)
    Glucose = int(Glucose)
    hdl_chol = int(hdl_chol)
    chol_hdl_ratio = float(chol_hdl_ratio)
    age = int(age)
    gender =(gender)
    hight = int(hight)
    weight = int(weight)
    bmi = float(bmi)
    Systolic_bp = int(Systolic_bp)
    diastolic_bp = int(diastolic_bp)
    waist = int(waist)
    hip = int(hip)
    waist_hip_ratio = float(waist_hip_ratio)

    features = np.array([(Cholesterol, Glucose, hdl_chol, chol_hdl_ratio, age, gender, hight, weight, bmi, Systolic_bp,
                          diastolic_bp, waist, hip, waist_hip_ratio)])
    prediction = model.predict(features)
    prediction = 'Diabetic' if prediction == 0 else "Not Diabetic"
    return render_template('prediction.html', prediction_text=' The pationt has diabetes:{}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True, port=7000)
