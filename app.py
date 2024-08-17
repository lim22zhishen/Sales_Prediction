
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
    BounceRates = st.number_input("Enter the bounce rates:", min_value=0.0)
    ExitRates = st.number_input("Enter the exit rates:", min_value=0.0)
    PageValues = st.number_input("Enter the page values:", min_value=0.0)
    SpecialDay = st.number_input("Enter the special day indicator (0 to 1):", min_value=0.0, max_value=1.0)

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

    # Load model and make prediction
    loaded_model = joblib.load('best_model.pkl')
    prediction = loaded_model.predict(final_input_data)

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
