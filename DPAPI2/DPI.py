from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'txt'}
MODEL_PATH = 'data_validation_model.pkl'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def get_validation_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
       
        model = IsolationForest(contamination=0.1, random_state=42)
     
        return model


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_data_structure(df):
    """
    Validate the structure of the uploaded data using ML
    """
    try:
       
        features = df.select_dtypes(include=[np.number]).fillna(0).values

        if len(features) == 0:
            return False, "No numerical data found for validation"

    
        model = get_validation_model()

       
        if not hasattr(model, 'estimators_'):
            model.fit(features[:min(100, len(features))])  

        
        preds = model.predict(features)

      
        anomaly_perc = (preds == -1).mean()

        if anomaly_perc > 0.3:  
            return False, f"High percentage of anomalous data ({anomaly_perc:.1%})"

        return True, "Data structure validated successfully"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


@app.route('/data/upload', methods=['POST'])
def upload_file():
   
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

   
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
          
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            
            if df.empty:
                return jsonify({"error": "File is empty"}), 400

            is_valid, validation_msg = validate_data_structure(df)

            if not is_valid:
                os.remove(filepath)  
                return jsonify({
                    "error": "Data validation failed",
                    "details": validation_msg
                }), 400

            
            return jsonify({
                "message": "File successfully uploaded and validated",
                "filename": filename,
                "validation": validation_msg,
                "columns": list(df.columns),
                "sample_data": df.head().to_dict(orient='records')
            }), 200

        except Exception as e:
            os.remove(filepath)  
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    app.run(debug=True)
