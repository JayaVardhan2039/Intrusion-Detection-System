from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import lime
import lime.lime_tabular
import os
from PIL import Image

# Load trained model and components
model_data = joblib.load("intrusion_detection.pkl")
scaler = model_data['scaler']
pca = model_data['pca']
model = model_data['model']
label_encoders = model_data['label_encoders']
for feature, encoder in label_encoders.items():
    print(f"Feature: {feature}")
    print(f"Classes: {encoder.classes_}")
    print(f"Encoded values: {encoder.transform(encoder.classes_)}")
    print("-" * 30)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Login credentials
users = {"admin": "jayavardhan"}

@app.route('/static0/<path:filename>')
def static0(filename):
    return send_from_directory('static0', filename)

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    session.clear()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('predict'))
        return "Invalid Credentials!"
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Ensure the file has only one row
        if df.shape[0] != 1:
            return "File must contain exactly one row of data"
        
        # Expected features
        expected_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Check if all expected features are present in the CSV
        if not all(feature in df.columns for feature in expected_features):
            return "CSV file is missing some required features"
        
        # Collect and process input data
        feature_values = []
        for feature in expected_features:
            value = df[feature].values[0]
            if feature in label_encoders:
                try:
                    value = label_encoders[feature].transform([value])[0]
                except ValueError:
                    return f"Error: Unknown value '{value}' for '{feature}'"
            feature_values.append(float(value))
        
        features = np.array(feature_values).reshape(1, -1)
        
        # Convert features to a DataFrame with column names before scaling
        features_df = pd.DataFrame(features, columns=expected_features)
        features_scaled = scaler.transform(features_df)
        features_pca = pca.transform(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_pca)[0]
        prediction_proba = model.predict_proba(features_pca).max() * 100

        try:
            # Define the prediction function for LIME
            def predict_fn(x):
                x_scaled = scaler.transform(x)
                x_pca = pca.transform(x_scaled)
                return model.predict_proba(x_pca)

            # Initialize LIME explainer (use encoded feature_values as training data)
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=features_df.values,  # Use DataFrame values with feature names
                feature_names=expected_features,
                class_names=label_encoders['labels'].classes_,
                mode='classification',
                discretize_continuous=False
            )

            # Generate explanation for the instance
            exp = explainer.explain_instance(
                features[0], 
                predict_fn, 
                num_features=10,
                top_labels=1
            )
            exp_html = exp.as_html()

            # Extract textual explanation
            lime_explanation_text = exp.as_list(label=exp.available_labels()[0])
            
            # Generate an informative explanation
            lime_explanation_informative = []
            for feature, weight in lime_explanation_text:
                impact = "positively" if weight > 0 else "negatively"
                lime_explanation_informative.append(
                    f"The feature '{feature}' {impact} influenced the prediction with a weight of {abs(weight):.4f}."
                )
            
            lime_explanation_text = "\n".join(lime_explanation_informative)

        except Exception as e:
            exp_html = f"Could not generate explanation: {str(e)}"
            lime_explanation_text = f"Could not generate explanation: {str(e)}"
                       
        return render_template('resultx.html',
                               lime_html=exp_html,
                               lime_text=lime_explanation_text)
           
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)