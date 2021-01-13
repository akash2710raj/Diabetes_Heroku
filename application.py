# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'prediction.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        # redirect(url_for())
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)