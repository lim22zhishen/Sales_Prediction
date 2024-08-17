
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to get user inputs
def get_user_inputs():
    st.title("Revenue Prediction Model")

    Administrative = st.number_input("Enter your number of administrative pages visited:", min_value=0.0)
    Administrative_Duration = st.number_input("Enter the total duration (in seconds) spent on administrative pages:", min_value=0.0)
    Informational = st.number_input("Enter the number of informational pages visited:", min_value=0.0)
    Informational_Duration = st.number_input("Enter the total duration (in seconds) spent on informational pages:", min_value=0.0)
    ProductRelated = st.number_input("Enter the number of product-related pages visited:", min_value=0.0)
    ProductRelated_Duration = st.number_input("Enter the total duration (in seconds) spent on product-related pages:", min_value=0.0)

    # Likert scale inputs
    BounceRates = st.slider("Rate the bounce rates on a scale of 1-5:", 1, 5)
    ExitRates = st.slider("Rate the exit rates on a scale of 1-5:", 1, 5)
    PageValues = st.slider("Rate the page values on a scale of 1-5:", 1, 5)
    SpecialDay = st.slider("Rate the special day indicator (1 to 5):", 1, 5)

    Month = st.selectbox("Select the month:", ['Feb', 'Mar', 'May', 'Oct', 'June', 'Jul', 'Aug', 'Nov', 'Sep', 'Dec'])
    OperatingSystems = st.selectbox("Select the operating system:", [1, 2, 3, 4, 5, 6, 7, 8])
    Browser = st.selectbox("Select the browser:", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Region = st.selectbox("Select the region:", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    TrafficType = st.selectbox("Select the traffic type:", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    Weekend = st.selectbox("Is it the weekend?", ['Yes', 'No'])

    return {
        'Administrative': Administrative,
        'Administrative_Duration': Administrative_Duration,
        'Informational': Informational,
        'Informational_Duration': Informational_Duration,
        'ProductRelated': ProductRelated,
        'ProductRelated_Duration': ProductRelated_Duration,
        'BounceRates': BounceRates,
        'ExitRates': ExitRates,
        'PageValues': PageValues,
        'SpecialDay': SpecialDay,
        'Month': Month,
        'OperatingSystems': OperatingSystems,
        'Browser': Browser,
        'Region': Region,
        'TrafficType': TrafficType,
        'Weekend': Weekend
    }

# Function to preprocess and predict
def preprocess_and_predict(user_input):
    # Encode categorical features
    categorical_data = {
        'Month': user_input['Month'],
        'OperatingSystems': user_input['OperatingSystems'],
        'Browser': user_input['Browser'],
        'Region': user_input['Region'],
        'TrafficType': user_input['TrafficType'],
        'Weekend': 1 if user_input['Weekend'] == 'Yes' else 0
    }
    numerical_data = {k: v for k, v in user_input.items() if k not in categorical_data}

    label_encoders = {}
    for feature_name in categorical_data:
        label_encoders[feature_name] = LabelEncoder()
        categorical_data[feature_name] = label_encoders[feature_name].fit_transform([categorical_data[feature_name]])

    # Preprocess categorical features
    processed_cat_data = pd.DataFrame(categorical_data)
    scaled_numerical_data = pd.DataFrame([numerical_data])

    # Combine numerical and categorical DataFrames
    final_input_data = pd.concat([scaled_numerical_data, processed_cat_data], axis=1)
    
    final_input_data = final_input_data.values.reshape(-1, 1)

    # Load model and make prediction
    loaded_model = joblib.load('best_model.pkl')
    prediction = loaded_model.predict(final_input_data)

    # SHAP explanation
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(final_input_data)

    st.write("### SHAP Explanation:")
    shap.force_plot(explainer.expected_value, shap_values[0], final_input_data, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')

    # Summary plot
    st.subheader("Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, final_input_data, show=False)
    st.pyplot(fig)

    input_df = pd.DataFrame(final_input_data, index=[0])
    shap_values_input = explainer.shap_values(input_df)

    # Force plot
    st.subheader("Force Plot")
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

    st.subheader("Decision Plot")
    st_shap(shap.decision_plot(explainer.expected_value[0], shap_values_input[0], final_input_data.columns))
    
    # Display result
    st.write("### Prediction Result:")
    if prediction[0] == 1:
        st.success("Positive Revenue Prediction")
    else:
        st.error("Negative Revenue Prediction")

def main():
    user_input = get_user_inputs()

    if st.button("Predict Revenue"):
        preprocess_and_predict(user_input)

if __name__ == "__main__":
    main()
