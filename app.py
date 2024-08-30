from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [float(request.form.get(feature)) for feature in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    
    # Convert data to numpy array and reshape for the model
    input_data = np.array([input_data])
    
    # Predict using the model
    prediction = model.predict(input_data)[0]
    
    # Return prediction to the UI
    result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
