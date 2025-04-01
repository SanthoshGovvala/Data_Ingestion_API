from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'txt'}
MODEL_PATH = 'data_validation_model.pkl'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load or create ML model for data validation
def get_validation_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Create a new model (this would be trained on your specific data schema)
        model = IsolationForest(contamination=0.1, random_state=42)
        # Note: In production, you'd train this on your known good data first
        return model


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_data_structure(df):
    """
    Validate the structure of the uploaded data using ML
    """
    try:
        # Convert dataframe to numerical features (simplified example)
        # In production, you'd have more sophisticated feature extraction
        features = df.select_dtypes(include=[np.number]).fillna(0).values

        if len(features) == 0:
            return False, "No numerical data found for validation"

        # Get the validation model
        model = get_validation_model()

        # Fit the model if it hasn't been fit (for demo purposes)
        # In production, this would be pre-trained
        if not hasattr(model, 'estimators_'):
            model.fit(features[:min(100, len(features))])  # Small sample to fit

        # Predict anomalies
        preds = model.predict(features)

        # Calculate anomaly percentage
        anomaly_perc = (preds == -1).mean()

        if anomaly_perc > 0.3:  # If more than 30% anomalies
            return False, f"High percentage of anomalous data ({anomaly_perc:.1%})"

        return True, "Data structure validated successfully"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


@app.route('/data/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If user does not select file, browser submits empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read the file based on extension
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            # Basic data check
            if df.empty:
                return jsonify({"error": "File is empty"}), 400

            # Validate data structure with ML
            is_valid, validation_msg = validate_data_structure(df)

            if not is_valid:
                os.remove(filepath)  # Clean up invalid file
                return jsonify({
                    "error": "Data validation failed",
                    "details": validation_msg
                }), 400

            # If we get here, file is valid
            return jsonify({
                "message": "File successfully uploaded and validated",
                "filename": filename,
                "validation": validation_msg,
                "columns": list(df.columns),
                "sample_data": df.head().to_dict(orient='records')
            }), 200

        except Exception as e:
            os.remove(filepath)  # Clean up on error
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    app.run(debug=True)