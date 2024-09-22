# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:35:20 2024

@author: user
"""


import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load the trained model and training data
model = load_model('best_heart_disease_model')
train_df = joblib.load('heart_train_data.pkl')

# Function to get user input
def get_user_input():
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    sex = st.selectbox('Sex', [0, 1])  # 1 = male, 0 = female
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=0)
    chol = st.number_input('Serum Cholesterol', min_value=0, max_value=600, value=0)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=0)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])
    
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return pd.DataFrame(user_data, index=[0])

# Prediction Page
def prediction():
    st.title("Heart Disease Prediction")
    user_input = get_user_input()
    
    if st.button("Predict"):
        prediction_result = predict_model(model, data=user_input)
        
        if 'prediction_label' in prediction_result.columns and 'prediction_score' in prediction_result.columns:
            predicted_class = int(prediction_result['prediction_label'].iloc[0])
            predicted_prob = prediction_result['prediction_score'].iloc[0]
            
            st.write("Prediction Result:")
            st.write(f"Predicted Class: {'Heart Disease' if predicted_class == 1 else 'No Heart Disease'}")
            st.write(f"Probability: {predicted_prob:.4f}")
            
            # Explanation of the prediction
            st.subheader("Prediction Explanation")
            if predicted_class == 1:
                st.write("The model predicts that the patient is likely to have heart disease.")
            else:
                st.write("The model predicts that the patient is unlikely to have heart disease.")
            
            st.subheader("Probability Explanation")
            st.write("""
                The probability score indicates the model's confidence in its prediction. 
                A higher score closer to 1 means the model is more confident that the patient has heart disease, 
                while a score closer to 0 means the model is more confident that the patient does not have heart disease.
            """)
            
            # Risk Factor Analysis using SHAP
            st.subheader("Risk Factor Analysis")
            shap.initjs()

            # Explaining the model's predictions using SHAP
            explainer = shap.Explainer(model.predict, train_df.drop(columns=['target']))
            shap_values = explainer(user_input)
            
            st.write("SHAP Values for Risk Factors:")
            st.write(shap_values.values)

            # Feature Importance Plot
            st.subheader("Feature Importance")
            st.write("""
                The Feature Importance plot shows the most significant factors contributing to the prediction.
                The higher the bar, the more influence that feature has on the model's prediction.
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
            st.pyplot(fig)

            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            st.write("""
                The SHAP Summary plot provides a visualization of the impact of each feature on the model's output.
                Each dot represents a feature's impact on the prediction for a specific instance. 
                The color represents the value of the feature (red for high, blue for low).
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, show=False)
            st.pyplot(fig)

            # SHAP Force Plot
            st.subheader("SHAP Force Plot")
            st.write("""
                The SHAP Force plot illustrates the impact of each feature on the model's prediction for a single instance.
                Features pushing the prediction towards a higher value (indicating heart disease) are shown in red, 
                while those pushing towards a lower value (indicating no heart disease) are in blue.
            """)
            st_shap(shap.force_plot(shap_values[0]), height=400)

            # SHAP Values Explanation
            st.subheader("SHAP Values Explanation")
            st.write("""
                SHAP values help in understanding the model's prediction by showing the contribution of each feature.
                - Positive SHAP values indicate that the feature contributes to predicting a higher probability of heart disease.
                - Negative SHAP values indicate that the feature contributes to predicting a lower probability of heart disease.
                - The magnitude of the SHAP value shows the strength of the contribution.
            """)
            
        else:
            st.error("Error: Prediction did not return 'prediction_label' or 'prediction_score' columns.")

# Helper function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Main function
def main():
    prediction()

if __name__ == "__main__":
    main()
