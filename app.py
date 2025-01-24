from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the pre-trained and pickled model (replace 'model.pkl' with the actual file path)
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.json

    # Extract feature values from the input JSON (update keys to match your feature names)
    feature_values = [
        data['Gender'],
        data['AttendanceRate'],
        data['StudyHoursPerWeek'],
        data['PreviousGrade'],
        data['ExtracurricularActivities'],
        data['ParentalSupport']
    ]

    # Convert to NumPy array and reshape for prediction
    feature_array = np.array(feature_values).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(feature_array)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
