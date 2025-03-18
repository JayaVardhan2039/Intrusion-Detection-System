from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import joblib
import pandas as pd
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
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_pca)[0]
        print("yubcysbh",prediction)
        prediction_proba = model.predict_proba(features_pca).max() * 100
        
        # Precaution messages
        precaution = {
            11: "No intrusion detected. Continue monitoring network activity for potential threats.",
    
            1: "Buffer overflow detected. Implement stack canaries, address space layout randomization (ASLR), and non-executable memory protection. Regularly update and patch vulnerable software.",
            
            2: "Unauthorized FTP write access detected. Restrict file upload permissions, enforce strong authentication (SFTP/FTPS), and monitor FTP logs for unusual activity.",
            
            3: "Brute-force password attack detected. Implement account lockout policies, use CAPTCHA for login attempts, enable multi-factor authentication (MFA), and monitor failed login attempts.",
            
            4: "IMAP attack detected. Enforce strong passwords and MFA, disable plain-text authentication, and use encrypted IMAP connections (IMAPS) to protect email access.",
            
            5: "Network reconnaissance (IP sweep) detected. Deploy Intrusion Detection and Prevention Systems (IDPS), monitor for unusual network scanning behavior, and use firewall rules to block repeated scan attempts.",
            
            6: "Land attack detected. Configure firewalls to block packets where the source and destination addresses are the same, and update the network stack to prevent exploitation.",
            
            7: "Loadable kernel module attack detected. Disable unnecessary kernel module loading, use Secure Boot, enforce strict kernel module signing, and regularly audit running modules.",
            
            8: "Multi-hop attack detected. Restrict proxy chaining, limit external connections from internal systems, and enforce network segmentation to prevent unauthorized access.",
            
            9: "SYN flood (DoS) attack detected. Enable SYN cookies, deploy rate limiting, use load balancers, and configure firewalls to block excessive half-open connections.",
            
            10: "Port scanning detected. Implement IP-based rate limiting, use network behavior analysis tools, and restrict unnecessary open ports with firewall rules.",
            
            12: "Perl script exploit detected. Restrict execution of server-side scripts, apply input validation and sanitization, and disable outdated CGI scripting when not needed.",
            
            13: "Phf vulnerability attack detected. Disable vulnerable CGI scripts, patch web servers, and enforce strict access control for web applications.",
            
            14: "Ping of Death attack detected. Configure firewalls to filter oversized ICMP packets, update the network stack to prevent vulnerability exploitation, and monitor ICMP traffic patterns.",
            
            15: "Port sweep detected. Deploy honeypots to detect scanning attempts, implement firewall rules to block scanning IPs, and conduct network traffic analysis for early detection.",
            
            16: "Rootkit activity detected. Use file integrity monitoring tools, deploy endpoint detection and response (EDR) solutions, and periodically scan systems for rootkit infections.",
            
            17: "Satan vulnerability scan detected. Regularly patch vulnerabilities, disable unnecessary services, and conduct internal penetration testing to assess security weaknesses.",
            
            18: "Smurf attack detected. Disable ICMP broadcast responses, configure anti-spoofing rules in firewalls, and implement network ingress/egress filtering.",
            
            19: "Spyware detected. Deploy endpoint security solutions, restrict software installation rights for users, and educate employees on phishing threats and suspicious downloads.",
            
            20: "Teardrop attack detected. Patch operating systems and networking devices to prevent fragmented packet vulnerabilities, and configure firewalls to block malformed packet sequences.",
            
            21: "Unauthorized warez client activity detected. Monitor file transfer logs, restrict anonymous FTP access, and enforce network usage policies to prevent illegal file sharing.",
            
            22: "Warezmaster intrusion detected. Strengthen authentication mechanisms for FTP services, implement strict access control policies, and regularly audit file-sharing services for unauthorized activities."
        }
        attack_label = prediction
        print("Decoded top Attack Label:", attack_label)
        precaution_message = precaution.get(attack_label, "No specific precaution available.")

        # Generate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_pca)
        print("SHAP Values Shape:", np.shape(shap_values))
        print("Expected Value Shape:", np.shape(explainer.expected_value))

        # Ensure shap_values has the correct shape
        if isinstance(shap_values, list):
            shap_values_single = shap_values[prediction]  # Extract SHAP values for the predicted class
        elif isinstance(shap_values, np.ndarray):
            shap_values_single = shap_values[:, :, prediction]  # Select the correct class
        else:
            shap_values_single = np.array([shap_values])  # Convert scalar to array

        # Ensure explainer.expected_value is indexed correctly
        if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.shape[0] == 23:
            base_value = explainer.expected_value[prediction]  # Select expected value for predicted class
        else:
            base_value = explainer.expected_value  # Use as-is if not multi-class

        # Reshape SHAP values if necessary
        shap_values_single = np.array(shap_values_single)

        if shap_values_single.ndim == 1:
            shap_values_single = shap_values_single.reshape(1, -1)
        
        # Create SHAP Explanation Object
        shap_explanation = shap.Explanation(
            values=shap_values_single[0],
            base_values=base_value,
            data=features_pca[0],
            feature_names=[f"PC{i+1}" for i in range(features_pca.shape[1])]
        )
        # Add this after generating SHAP values but before rendering the template

        # 1. Waterfall plot explanation
        waterfall_text = (
            f"This waterfall plot shows how each principal component contributes to pushing the model's output "
            f"from the base value ({base_value:.2f}) to the final prediction. Values above zero increase "
            f"the likelihood of this prediction, while values below zero decrease it."
        )
        
        # 2. Summary bar plot explanation
        top_pcs = np.argsort(np.abs(shap_values_single).mean(0))[::-1][:3]  # Top 3 components
        bar_text = (
            f"The most significant principal components influencing this prediction were: "
            f"{', '.join([f'PC{i+1} ({shap_values_single[0,i]:.2f})' for i in top_pcs])}. "
            "Longer bars indicate greater impact on the model's decision."
        )
        
        # 3. PCA component interpretation (if you have access to original feature loadings)
        # Modified code to explain top 3 PCs
        try:
            pca_loadings = pca.components_
            pc_text = "Key original features contributing to principal components: "
            for i in range(10):  # For PC1, PC2, PC3
                top_features = np.argsort(np.abs(pca_loadings[i]))[::-1][:3]
                pc_text += (
                    f"PC{i+1} uses {expected_features[top_features[0]]}, "
                    f"{expected_features[top_features[1]]}, {expected_features[top_features[2]]}. "
                )
        except Exception as e:
            pc_text = "Principal components combine original network features"

        
        # Generate SHAP Waterfall Plot
        plt.figure()
        shap.plots.waterfall(shap_explanation)

        # Save the plot
        shap_plot_path1 = "static/shap_plot1.png"
        plt.savefig(shap_plot_path1, bbox_inches='tight', dpi=300)
        plt.close()

        # Generate SHAP Waterfall Plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_pca)
        
        # Define the path for saving the SHAP plot
        shap_plot_path = "static/shap_plot.png"
        
        # Generate SHAP summary bar plot
        plt.figure(figsize=(14, 6))  
        shap.summary_plot(shap_values, features_pca, plot_type="bar", show=False)
        plt.savefig(shap_plot_path, bbox_inches='tight', dpi=400)  
        plt.close()  # Close the plot to prevent overlap 


        # Extract SHAP values for the predicted class
        shap_values_single = shap_values[0, :, prediction]  # Shape: (10,)
        # Reshape SHAP values to match features_pca
        shap_values_single = shap_values_single.reshape(1, -1)
        pca_feature_names = [f"PC{i+1}" for i in range(features_pca.shape[1])]
        shap_plot_path2 = "static/shap_plot2.png"
        plt.figure(figsize=(14, 6)) 
        shap.summary_plot(shap_values_single, features_pca, plot_type="bar", feature_names=pca_feature_names, show=False)
        plt.xticks(rotation=-90)
        plt.yticks(rotation=-90)
        plt.savefig(shap_plot_path2, bbox_inches='tight', dpi=400)  
        plt.close()
        
        # Rotate the saved image by -90 degrees
        with Image.open(shap_plot_path2) as img:
            rotated_img = img.rotate(90, expand=True)  # Rotate and expand to fit the image
            rotated_img.save(shap_plot_path2)  # Save the rotated image

        # Global explanation text
        try:
            # Calculate mean absolute SHAP values across all classes
            if isinstance(shap_values, list):
                shap_array = np.array(shap_values)  # Convert list to array for multi-class
                global_mean_shap = np.abs(shap_array).mean(axis=(0, 1))
            else:
                global_mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Get top 3 components
            top_global_indices = np.argsort(global_mean_shap)[::-1][:3]
            top_global_pcs = [f"PC{i+1}" for i in top_global_indices]
            
            # Create explanation text
            global_explanation = (
                f"The model's decisions are primarily driven by these principal components: "
                f"{', '.join(top_global_pcs)}. These components capture the most significant "
                "network traffic patterns for intrusion detection across all predictions."
            )
        except Exception as e:
            print(f"Global explanation error: {str(e)}")
            global_explanation = "Overall model decisions are influenced by combinations of network traffic characteristics."

        # Get actual attack labels from the trained model
        attack_labels = {i: label for i, label in enumerate(model.classes_)}  # Dynamically map indices to names

        # Convert prediction index to actual attack name using label encoder
        # Convert prediction index to actual attack name using label encoder
        if 'labels' in label_encoders:  # Use 'labels' instead of 'attack_type'
            attack_label = label_encoders['labels'].inverse_transform([prediction])[0]
        else:
            attack_label = attack_labels.get(prediction, "Unknown Attack Type")  # Fallback if no encoder
        
        
        print(f"Decoded Attack Label: {attack_label}")  # Should now show actual attack name like "neptune" or "normal"
       
        return render_template('result.html', 
                               attack_type=attack_label,  # Displays actual attack name!
                               confidence=round(prediction_proba, 2), 
                               precaution=precaution_message,  
                               shap_img=shap_plot_path,
                              shap_img1=shap_plot_path1,
                              shap_img2=shap_plot_path2,
                              waterfall_explanation=waterfall_text,
                              bar_explanation=bar_text,
                              pca_interpretation=pc_text,
                              global_explanation=global_explanation)
           
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)