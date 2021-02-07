import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import waitress
import os

port_= int(os.environ.get("PORT", 5000))

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = [np.array([int(x) for x in request.form.values()])]
    prediction = model.predict(data)
    
    return render_template('index.html', prediction_text=f"Its a {prediction[0]}")


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    serve(app,port=port_)

