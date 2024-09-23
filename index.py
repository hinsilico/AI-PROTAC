import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import matplotlib.pyplot as plt
import base64
import threading
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

        
# Load dataset with caching
@st.cache_data
def load_data():
    data_path = "protac_activity_db.csv"
    df = pd.read_csv(data_path)
    return df

# Convert SMILES string to molecular fingerprint
def mol_to_fp(mol):
    mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return list(fp)

# Predict PROTAC activity for a given SMILES string
def predict_protac_activity(models, scaler, smiles):
    mol_fp = mol_to_fp(smiles)
    mol_fp_scaled = scaler.transform([mol_fp])
    predictions = {}
    for name, model in models.items():
        y_prob = model.predict_proba(mol_fp_scaled)[:, 1]
        predictions[name] = y_prob[0]
    return predictions

# Extract molecular properties from a SMILES string
def extract_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {}
    properties['Molecular Weight'] = Descriptors.MolWt(mol)
    properties['Number of H-Bond Donors'] = Descriptors.NumHDonors(mol)
    properties['Number of H-Bond Acceptors'] = Descriptors.NumHAcceptors(mol)
    return properties

# SHAP analysis to explain model predictions
import shap
import numpy as np

def shap_analysis(model, X_train_scaled, X_test_scaled):
    # Using the TreeExplainer for the RandomForest model
    explainer = shap.TreeExplainer(model)
    
    # Get SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)

    # Check if the output is a list (for binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get SHAP values for the positive class

    # Check if shapes are correct
    if shap_values.shape[0] != X_test_scaled.shape[0]:
        raise ValueError("Mismatch between SHAP values and feature matrix dimensions.")

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar")

    # Ensure dependence plot is using the same number of samples
    shap.dependence_plot(0, shap_values, X_test_scaled)


# Describe PROTAC activity based on probability
def describe_activity(probability):
    return "Active" if probability >= 0.5 else "Inactive"

# Plot bar chart for predicted activity probability
def plot_activity_probability(prob):
    fig, ax = plt.subplots()
    ax.bar(['Inactive', 'Active'], [1 - prob, prob], color=['blue', 'green'])
    ax.set_xlabel('Activity')
    ax.set_ylabel('Probability')
    ax.set_title('Predicted Activity Probability')
    return fig

# Plot ROC-AUC curve for models
def plot_roc_auc_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt

# Function to display training and test set compounds
def display_train_test_sets(X_train, X_test, y_train, y_test, df):
    st.subheader("Training Set Compounds")
    train_indices = y_train.index
    st.write(df.iloc[train_indices][['mol', 'flag']])

    st.subheader("Test Set Compounds")
    test_indices = y_test.index
    st.write(df.iloc[test_indices][['mol', 'flag']])

# Function to download file as CSV
def get_binary_file_downloader_html(bin_file, file_name):
    bin_data = bin_file.to_csv(index=False).encode()
    bin_str = base64.b64encode(bin_data).decode()
    href = f'<a href="data:file/csv;base64,{bin_str}" download="{file_name}">Download {file_name}</a>'
    return href

def main():
    # CSS customization
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background: #2d2d2d;
        }
        .stButton>button {
            color: #fff;
            background-color: #6c757d;
        }
        .stButton>button:hover {
            background-color: #4e4e4e;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    
    # Call the background fetch function
    fetch_monitor_in_background()
    
    # Tool description
    st.markdown("""
        AI-DPAPT is a Machine Learning-based tool developed for predicting PROTAC activity. 
        The tool leverages multiple models such as Random Forest, Gradient Boosting, 
        AdaBoost, SVM, and Multi-layer Perceptron to predict the likelihood of a 
        given molecule being active or inactive as a PROTAC.
        """)
    
    
    
    # Load dataset
    df = load_data()
    
    df.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    df['mol_encoded'] = label_encoder.fit_transform(df['mol'])

    df['mol_fp'] = df['mol'].apply(mol_to_fp)

    X = np.array(df['mol_fp'].to_list())
    y = df['flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RF-AI-DPAPT": RandomForestClassifier(),
        "GBoost-AI-DPAPT": GradientBoostingClassifier(),
        "AdaBoost-AI-DPAPT": AdaBoostClassifier(),
        "SVM-AI-DPAPT": SVC(probability=True),
        "MLP-AI-DPAPT": MLPClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    user_smiles = st.text_input("Enter a SMILES string:", value="CCN(CCCC#CC1=C[N](CCC(=O)N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)N[C@@H](C)C3=CC=C(C=C3)C4=C(C)N=CS4)C(C)(C)C)N=C1)CCOC5=CC=C(C=C5)C(=O)C6=C(SC7=CC(=CC=C67)O)C8=CC=C(O)C=C8")
    
    if st.button("Predict"):
        if user_smiles:
            predictions = predict_protac_activity(trained_models, scaler, user_smiles)
            st.subheader("Molecular Properties:")
            properties = extract_molecular_properties(user_smiles)
            for prop, value in properties.items():
                st.write(f"{prop}: {value}")

            st.subheader("Predictions:")
            for name, prob in predictions.items():
                activity = describe_activity(prob)
                st.write(f"{name}: {prob:.4f} ({activity})")

            df_pred = pd.DataFrame(predictions.items(), columns=['Model', 'Predicted Probability'])
            st.write(df_pred)

            st.subheader("Predicted Activity Probability:")
            activity_prob_plot = plot_activity_probability(predictions["RF-AI-DPAPT"])
            st.pyplot(activity_prob_plot)

            st.subheader("AUC-ROC Curves:")
            auc_plot = plot_roc_auc_curve(trained_models, X_test_scaled, y_test)
            st.pyplot(auc_plot)
            
            st.markdown(get_binary_file_downloader_html(df_pred, file_name="predictions.csv"), unsafe_allow_html=True)

    if st.button("Compare Models"):
        performance = compare_model_performance(trained_models, X_test_scaled, y_test)
        
        st.subheader("Model Accuracy Comparison")

        fig, ax = plt.subplots()
        ax.bar(performance.keys(), performance.values(), color='purple')
        ax.set_xlabel('Models/Tools')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy vs Public Tools')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

	# Run SHAP analysis on the RandomForest model
	print("\nRunning SHAP analysis for RandomForest:")
	try:
    	shap_analysis(trained_models["RF-AI-DPAPT"], X_train_scaled, X_test_scaled)
	except ValueError as e:
    	print(f"Error during SHAP analysis: {e}")

    if st.checkbox("Show Training and Test Set Compounds"):
        display_train_test_sets(X_train, X_test, y_train, y_test, df)

if __name__ == "__main__":
    main()
