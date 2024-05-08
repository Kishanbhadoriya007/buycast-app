from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("purchase_history.csv")

# Load the trained model and scaler
with open('knn_model.pickle', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

# Function to make predictions
def predict_purchase(gender, age, salary, price):
    # Encode gender
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Scale the input features
    input_features = scaler.transform([[gender_encoded, age, salary, price]])
    
    # Make prediction
    prediction = knn_model.predict(input_features)[0]
    
    return prediction

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        price = float(request.form['price'])
        
        prediction = predict_purchase(gender, age, salary, price)
        prediction_text = "Likely to purchase" if prediction == 1 else "Not likely to purchase"
        
        return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
