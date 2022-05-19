from flask import Flask, redirect, url_for, request, render_template

import pickle
import numpy as np
##def pre_process(data):
 ## data['waist_hip_ratio'] = data['waist_hip_ratio'].str.replace(',','.')
 ## data['waist_hip_ratio'] = data['waist_hip_ratio'].astype(float)
 ## data["chol_hdl_ratio"] = data["chol_hdl_ratio"].str.replace(",",".")
 ## data["chol_hdl_ratio"] = data["chol_hdl_ratio"].astype(float)
 ## data['bmi'] = data['bmi'].str.replace(',','.')
 ## data['bmi'] = data['bmi'].astype(float)
 ## data['gender'] = data['gender'].str.replace('female','1')
 ## data['gender'] = data['gender'].str.replace('male','0')
 ## data['gender'] = data['gender'].astype(int)
 ## data['diabetes'] = data['diabetes'].str.replace('Diabetes','1')
 ## data['diabetes'] = data['diabetes'].str.replace('No diabetes','0')
 ## data['diabetes'] = data['diabetes'].astype(int)
 ## data.drop(["patient_number"],axis=1,inplace=True)
 ## return data
# def load_model(data):
model = pickle.load(open('finalized_model.sav','rb'))
  ##out = model.predict(np.array(data).reshape(1,-1))
  ##return out[0]
app = Flask(__name__, template_folder='templates')
@app.route('/',methods=['GET'])
def home():
    return render_template('prediction.html')


##app.route('/predict', methods=['GET','POST'])
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

    Cholesterol = int(Cholesterol)
    Glucose = int(Glucose)
    hdl_chol = int(hdl_chol)
    chol_hdl_ratio = float(chol_hdl_ratio)
    age = int(age)
    gender = int(gender)
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
    return render_template('index.html', prediction_text=' The pationt has diabetes:{}'.format(prediction))
 ##if request == 'POST':
 ##  f = request.file['file']
 ##  data = pre_process(f)
 ##  out = load_model(data)
    ## if out == 0
    ##    return 'no diabetes'
    ## else
     ##   return 'diabetes'
 ##  return out
 ##return none
if __name__ == 'main':
    app.run(debug=True, port=7000)


#########
#To use the predict button in our web-app
# @app.route('/predict',methods=['POST'])
# def predict():
#     #For rendering results on HTML GUI
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0], 2)
#     return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))
