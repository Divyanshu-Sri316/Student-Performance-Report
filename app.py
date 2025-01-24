from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model using pickle
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature values from the request data
        data = request.json

        # Feature extraction
        feature_values = [
            data['Gender'],                  # 1 for Male, 0 for Female
            data['AttendanceRate'],          # Numeric value (e.g., percentage)
            data['StudyHoursPerWeek'],       # Numeric value
            data['PreviousGrade'],           # Numeric value
            data['ExtracurricularActivities'],  # Numeric value (can be greater than 3)
            data['ParentalSupport']          # Values between 1 to 3
        ]

        # Input validation
        if not (1 <= data['ParentalSupport'] <= 3):
            return jsonify({"error": "ParentalSupport must be between 1 and 3"}), 400

        # Convert the features to a numpy array for prediction
        feature_array = np.array(feature_values).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(feature_array)[0]
        return jsonify({"prediction": prediction})
    except KeyError as ke:
        return jsonify({"error": f"Missing key in input data: {ke}"}), 400
    except ValueError as ve:
        return jsonify({"error": f"Invalid value in input data: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
