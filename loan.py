import streamlit as st
from PIL import Image
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

def load_image(image_path):
    """Load and return image if available."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image file '{image_path}' is missing.")
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


@st.cache_resource
def load_model():
    """Load model and scaler with caching."""
    try:
        if not os.path.exists('model.pkl') or not os.path.exists('scale.pkl'):
            raise FileNotFoundError("Model or scaler file not found.")
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scale.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"An error occurred while loading the model or scaler: {e}")
        return None, None


def main():
    st.title(":rainbow[Loan Approval Prediction]")
    st.markdown("Predict whether a loan will be approved based on personal and financial details.")

    # Load and display image
    image = load_image('download.jpg')
    if image:
        st.image(image, use_column_width=True)

    # Sidebar inputs
    st.sidebar.title("Applicant Information")

    person_age = st.sidebar.number_input("Person Age", min_value=18, max_value=100, value=30)
    person_gender = st.sidebar.selectbox("Person Gender", options=["Male", "Female", "Other"])
    person_education = st.sidebar.selectbox("Education", options=["High School", "Undergraduate", "Graduate", "Postgraduate"])
    person_income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_exp = st.sidebar.number_input("Employment Experience (years)", min_value=0, value=5)
    person_home_ownership = st.sidebar.selectbox("Home Ownership", options=["Own", "Rent", "Mortgage"])
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=10000)
    loan_intent = st.sidebar.selectbox("Loan Intent", options=["Personal", "Business", "Education"])
    loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    loan_percent_income = st.sidebar.number_input("Loan % of Income", min_value=0.0, max_value=100.0, value=10.0)
    cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=1, value=5)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.sidebar.selectbox("Previous Loan Defaults", options=["No", "Yes"])

    # Validate inputs
    if loan_amnt > person_income:
        st.warning("⚠️ Loan amount exceeds applicant's annual income.")

    # Encode categorical features
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    education_map = {"High School": 0, "Undergraduate": 1, "Graduate": 2, "Postgraduate": 3}
    home_ownership_map = {"Own": 0, "Rent": 1, "Mortgage": 2}
    loan_intent_map = {"Personal": 0, "Business": 1, "Education": 2}
    previous_defaults_map = {"No": 0, "Yes": 1}

    # Build feature vector (13 + 1 dummy = 14)
    dummy_feature = 0  # TEMP: placeholder to match scaler’s expected input
    features = [
        person_age,
        gender_map[person_gender],
        education_map[person_education],
        person_income,
        person_emp_exp,
        home_ownership_map[person_home_ownership],
        loan_amnt,
        loan_intent_map[loan_intent],
        loan_int_rate / 100,  # Convert percent to decimal
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_defaults_map[previous_loan_defaults_on_file],
        dummy_feature  # Add this to match the scaler's expected 14 features
    ]

    model, scaler = load_model()
    if model is None or scaler is None:
        return

    if st.button("🔍 Predict Loan Approval"):
        try:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                st.success("✅ Loan Approved!")
            else:
                st.error("❌ Loan Denied.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
