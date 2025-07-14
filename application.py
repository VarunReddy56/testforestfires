import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

# import ridge regressor and standard scaler pickle

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            print("Form submitted ✅")

            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            print("Received inputs:", Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region)

            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            scaled_data = standard_scaler.transform(input_data)
            prediction = ridge_model.predict(scaled_data)
            result = round(prediction[0], 2)

            print("Prediction result:", result)
            return render_template('home.html', result=result)

        except Exception as e:
            print("❌ Error occurred:", e)
            return f"<h1 style='color:red'>Error: {str(e)}</h1>"

    else:
        print("GET method accessed 🟢")
        return render_template('home.html', result=None)
    
if __name__ == '__main__':
    app.run(debug=True, port=5053)  # or 5050, 8080, etc.