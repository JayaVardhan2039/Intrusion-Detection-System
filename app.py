from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import lime
import lime.lime_tabular
import os
from PIL import Image
import requests
import subprocess
import time
import json
from helpers import get_confidence_based_data

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

# Start Ollama Server
def start_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("Ollama is already running.")
            return None
    except requests.exceptions.RequestException:
        print("Starting Ollama server...")
        return subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

ollama_process = start_ollama()
time.sleep(5)  # Wait for Ollama to start

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
        return render_template('predict.html',xpected_features=[
                                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
                                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
                                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
                             ])
     
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
    try:
        df = None
        
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename == '':
                return "No file selected"
            
            df = pd.read_csv(file)
            
            if df.shape[0] != 1:
                return "File must contain exactly one row of data"
            
            if not all(feature in df.columns for feature in expected_features):
                return "CSV file is missing some required features"

        # Handle form input
        else:
            form_data = {}
            for feature in expected_features:
                value = request.form.get(feature)
                if not value:
                    return f"Missing value for {feature}"
                form_data[feature] = [value]
            
            df = pd.DataFrame(form_data)

        
        
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
    0: "Back Attack (Denial of Service - DoS): Back attack detected. Configure firewalls to block malicious HTTP requests, use rate limiting to prevent abuse, and monitor server logs for unusual patterns. \n\nExplanation: A back attack involves sending a large number of HTTP requests to a web server, overwhelming it and causing denial of service.",
    11: "No intrusion detected. Continue monitoring network activity for potential threats.",

    1: "Buffer Overflow Attack (User-to-Root - U2R): Buffer overflow detected. Implement stack canaries, address space layout randomization (ASLR), and non-executable memory protection. Regularly update and patch vulnerable software. \n\nExplanation: A buffer overflow attack occurs when an attacker sends more data to a program than it can handle, causing it to overwrite adjacent memory. This can lead to arbitrary code execution, crashes, or privilege escalation.",

    2: "Unauthorized FTP Write Attack (Remote-to-Local - R2L): Unauthorized FTP write access detected. Restrict file upload permissions, enforce strong authentication (SFTP/FTPS), and monitor FTP logs for unusual activity. \n\nExplanation: This attack involves exploiting misconfigured FTP servers to upload malicious files, which can be used to compromise the system or distribute malware.",

    3: "Brute-Force Password Attack (Remote-to-Local - R2L): Brute-force password attack detected. Implement account lockout policies, use CAPTCHA for login attempts, enable multi-factor authentication (MFA), and monitor failed login attempts. \n\nExplanation: Attackers systematically try multiple password combinations to gain unauthorized access to accounts or systems.",

    4: "IMAP Exploitation Attack (Remote-to-Local - R2L): IMAP attack detected. Enforce strong passwords and MFA, disable plain-text authentication, and use encrypted IMAP connections (IMAPS) to protect email access. \n\nExplanation: Attackers exploit vulnerabilities in IMAP services to gain unauthorized access to email accounts, often through weak authentication mechanisms.",

    5: "IP Sweep (Probing/Scanning): Network reconnaissance (IP sweep) detected. Deploy Intrusion Detection and Prevention Systems (IDPS), monitor for unusual network scanning behavior, and use firewall rules to block repeated scan attempts. \n\nExplanation: IP sweeps are used to identify active hosts on a network, often as a precursor to more targeted attacks.",

    6: "Land Attack (Denial of Service - DoS): Land attack detected. Configure firewalls to block packets where the source and destination addresses are the same, and update the network stack to prevent exploitation. \n\nExplanation: A land attack sends spoofed TCP SYN packets with identical source and destination IPs, causing the target system to crash or freeze.",

    7: "Loadable Kernel Module (LKM) Attack (User-to-Root - U2R): Loadable kernel module attack detected. Disable unnecessary kernel module loading, use Secure Boot, enforce strict kernel module signing, and regularly audit running modules. \n\nExplanation: Attackers load malicious kernel modules to gain root-level access or hide their presence on the system.",

    8: "Multi-Hop Attack (Remote-to-Local - R2L): Multi-hop attack detected. Restrict proxy chaining, limit external connections from internal systems, and enforce network segmentation to prevent unauthorized access. \n\nExplanation: Attackers use multiple compromised systems to hide their origin and gain access to sensitive systems.",

    9: "SYN Flood (Denial of Service - DoS): SYN flood (DoS) attack detected. Enable SYN cookies, deploy rate limiting, use load balancers, and configure firewalls to block excessive half-open connections. \n\nExplanation: A SYN flood overwhelms a target system with half-open TCP connections, exhausting resources and causing denial of service.",

    10: "Port Scanning Attack (Probing/Scanning): Port scanning detected. Implement IP-based rate limiting, use network behavior analysis tools, and restrict unnecessary open ports with firewall rules. \n\nExplanation: Port scanning is used to identify open ports and services on a target system, often as a precursor to exploitation.",

    12: "Perl Script Exploit Attack (User-to-Root - U2R): Perl script exploit detected. Restrict execution of server-side scripts, apply input validation and sanitization, and disable outdated CGI scripting when not needed. \n\nExplanation: Attackers exploit vulnerabilities in Perl scripts to execute arbitrary code or gain unauthorized access to the system.",

    13: "Phf Vulnerability Attack (Remote-to-Local - R2L): Phf vulnerability attack detected. Disable vulnerable CGI scripts, patch web servers, and enforce strict access control for web applications. \n\nExplanation: The phf vulnerability allows attackers to exploit a CGI script to execute commands or access sensitive files on the server.",

    14: "Ping of Death Attack (Denial of Service - DoS): Ping of Death attack detected. Configure firewalls to filter oversized ICMP packets, update the network stack to prevent vulnerability exploitation, and monitor ICMP traffic patterns. \n\nExplanation: This attack sends oversized ICMP packets to crash or freeze the target system by exploiting vulnerabilities in the network stack.",

    15: "Port Sweep Attack (Probing/Scanning): Port sweep detected. Deploy honeypots to detect scanning attempts, implement firewall rules to block scanning IPs, and conduct network traffic analysis for early detection. \n\nExplanation: A port sweep scans multiple ports on a single host to identify open services and potential vulnerabilities.",

    16: "Rootkit Attack (User-to-Root - U2R): Rootkit activity detected. Use file integrity monitoring tools, deploy endpoint detection and response (EDR) solutions, and periodically scan systems for rootkit infections. \n\nExplanation: Rootkits are malicious tools that provide attackers with persistent, stealthy access to a compromised system.",

    17: "Satan Vulnerability Scan (Probing/Scanning): Satan vulnerability scan detected. Regularly patch vulnerabilities, disable unnecessary services, and conduct internal penetration testing to assess security weaknesses. \n\nExplanation: Satan is a network scanning tool used to identify vulnerabilities in systems and services.",

    18: "Smurf Attack (Denial of Service - DoS): Smurf attack detected. Disable ICMP broadcast responses, configure anti-spoofing rules in firewalls, and implement network ingress/egress filtering. \n\nExplanation: A smurf attack floods the target with ICMP echo requests using IP spoofing, overwhelming the network with traffic.",

    19: "Spyware Attack (Data Theft/Espionage): Spyware detected. Deploy endpoint security solutions, restrict software installation rights for users, and educate employees on phishing threats and suspicious downloads. \n\nExplanation: Spyware is malicious software that secretly monitors user activity and collects sensitive data.",

    20: "Teardrop Attack (Denial of Service - DoS): Teardrop attack detected. Patch operating systems and networking devices to prevent fragmented packet vulnerabilities, and configure firewalls to block malformed packet sequences. \n\nExplanation: A teardrop attack sends malformed fragmented packets to crash the target system by exploiting vulnerabilities in packet reassembly.",

    21: "Warez Client Activity (Remote-to-Local - R2L): Unauthorized warez client activity detected. Monitor file transfer logs, restrict anonymous FTP access, and enforce network usage policies to prevent illegal file sharing. \n\nExplanation: Warez clients are used to download or distribute pirated software, often leading to legal and security risks.",

    22: "Warezmaster Intrusion (Remote-to-Local - R2L): Warezmaster intrusion detected. Strengthen authentication mechanisms for FTP services, implement strict access control policies, and regularly audit file-sharing services for unauthorized activities. \n\nExplanation: Warezmaster attacks involve unauthorized access to FTP servers to distribute pirated software or malware."
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
            f"The most significant principal components influencing this prediction were:"
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
            
            # First format the components with line breaks
            components_list = '\n'.join([f'- {pc}' for pc in top_global_pcs])

            # Then use in the f-string
            global_explanation = f"""
            The model's decision-making process is primarily influenced by these key principal components:
            {components_list}

            These components represent the most impactful network traffic patterns identified through PCA analysis, 
            capturing the essential characteristics that differentiate normal behavior from various attack types 
            across all predictions in the intrusion detection system.
            """
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
        '''
        
        print(f"Decoded Attack Label: {attack_label}")  # Should now show actual attack name like "neptune" or "normal"
        questions = "\n".join([
            f"1. Tell what {attack_label} attack is in 1 line.",
            f"2. Explain {attack_label} attack in 4-5 lines.",
            f"3. Summarize the impact of {attack_label} attack in 4-5 lines.",
            f"4. How does {attack_label} attack work? Answer in 4-5 lines.",
            f"5. Provide a short overview of {attack_label} attack in 4-5 lines."
        ])
        
        predefined_question = f"""Answer each numbered question directly following this format:
        [number]: [answer]
        Do not explain your answers. Be concise.
        {questions}"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": predefined_question,
                    "max_tokens": 400  # Increased for multiple answers
                },
                stream=True
            )
        
            generated_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = line.decode("utf-8")
                        data = json.loads(json_line)
                        if "response" in data:
                            generated_text += data["response"]
                    except json.JSONDecodeError as e:
                        print("JSON Decode Error:", e)
        
        except Exception as e:
            generated_text = f"Error: {str(e)}"
        '''
        '''
        questions = "\n".join([
            f"Tell what {attack_label} attack is in only 1 line",
            f"Tell what {attack_label} attack is in 4 to 5 lines.",
            f"Explain {attack_label} attack in in 4 to 5 lines.",
            f"Summarize the impact of a {attack_label} attack in 4 to 5 lines..",
            f"How does a {attack_label} attack work? in 4 to 5 lines.",
            f"Provide a short overview of {attack_label} attack in 4 to 5 lines."
        ])
        predefined_question = f"Answer these questions concisely:\n{questions}"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "deepseek-r1:7b", "prompt": predefined_question},  # Increased tokens
                stream=True
            )
        
            generated_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = line.decode("utf-8")
                        data = json.loads(json_line)
                        if "response" in data:
                            generated_text += data["response"]
                    except json.JSONDecodeError as e:
                        print("JSON Decode Error:", e)
        
        except Exception as e:
            generated_text = f"Error: {str(e)}"
        '''
        # In your prediction route
        confidence_value = round(prediction_proba, 2)
        attack_data = get_confidence_based_data(attack_label, confidence_value)

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
                              global_explanation=global_explanation,
                              lime_html=exp_html,
                              lime_text=lime_explanation_text,
                              causes=attack_data['causes'],
                              steps=attack_data['steps'],
                              explanations=attack_data['explanations'],
                              precautions=attack_data['precautions'],)
           
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if ollama_process:
            ollama_process.terminate()  # Ensure Ollama stops when Flask stops