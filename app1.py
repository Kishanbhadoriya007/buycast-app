import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D:/Transfer_for_saving_space/Downloads/purchase_history.csv")

# Preprocess the data
gender_encoded = pd.get_dummies(df['Gender'], drop_first=True)
df = pd.concat([df, gender_encoded], axis=1)
x = df[['Male', 'Age', 'Salary', 'Price']].to_numpy()
y = df['Purchased'].to_numpy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# Function to get accuracy score
def get_accuracy_score():
    y_pred = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f"Customer behaviour model with {accuracy*100:.0f}% accuracy"

# Function to predict purchases for customers in CSV file
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)
    predictions = []
    for index, row in df.iterrows():
        gender = row['Gender']
        age = row['Age']
        salary = row['Salary']
        price = row['Price']
        prediction = predict_purchase(gender, age, salary, price)
        predictions.append(prediction)
    return predictions

# Set up the Streamlit app
st.title("Purchase Prediction")

# Accuracy Score Display
st.subheader(get_accuracy_score())

# Gender Selector
gender = st.selectbox("Gender:", ["Male", "Female"])

# Age Input
age = st.number_input("Age:", value=25)

# Salary Input
salary = st.number_input("Salary:", value=50000)

# Price Input
price = st.number_input("Price:", value=50)

# Predict Button for single customer prediction
if st.button("Predict"):
    prediction = predict_purchase(gender, age, salary, price)
    prediction_text = "Likely to purchase" if prediction == 1 else "Not likely to purchase"
    st.write(f'Predicted Purchase: {prediction_text}')

# OR Label
st.subheader("OR")

# Upload CSV Button
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file is not None:
    predictions = predict_from_csv(uploaded_file)
    for i, pred in enumerate(predictions):
        prediction_text = 'Likely to purchase' if pred == 1 else 'Not likely to purchase'
        st.write(f'Customer {i+1}: {prediction_text}')