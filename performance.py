import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_90_modelpp.pkl")

# Dropdown options
departments = ['Sales', 'Human Resources', 'Development', 'Data Science', 'Research & Development', 'Finance']
job_roles = ['Sales Executive', 'Manager', 'Developer', 'Sales Representative', 'Human Resources',
             'Senior Developer', 'Data Scientist', 'Senior Manager R&D', 'Laboratory Technician',
             'Manufacturing Director', 'Research Scientist', 'Healthcare Representative',
             'Research Director', 'Manager R&D', 'Finance Manager', 'Technical Architect',
             'Business Analyst', 'Technical Lead', 'Delivery Manager']
travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
overtime_options = ['No', 'Yes']

# Title
st.title("PerformEdge – Predict Employee Performance Rating")
st.write("Use your trained XGBoost model to predict performance rating of employees based on their profile.")

# Mode selection
mode = st.radio("Select input mode", ["Upload CSV file", "Enter manually"])

# Manual input UI
def get_user_input():
    st.subheader("Enter employee data:")
    EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    EmpDepartment = st.selectbox("Department", departments)
    ExperienceYearsInCurrentRole = st.slider("Years in Current Role", 0, 30, 2)
    EmpWorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
    OverTime = st.selectbox("OverTime", overtime_options)
    YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 2)
    TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 2)
    EmpJobLevel = st.slider("Job Level", 1, 5, 2)
    EmpJobRole = st.selectbox("Job Role", job_roles)
    ExperienceYearsAtThisCompany = st.slider("Years at Company", 0, 30, 3)
    EmpJobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    BusinessTravelFrequency = st.selectbox("Business Travel Frequency", travel_options)
    EmpRelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    EmpJobInvolvement = st.slider("Job Involvement", 1, 4, 3)

    data = {
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'EmpDepartment': departments.index(EmpDepartment),
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'OverTime': overtime_options.index(OverTime),
        'YearsWithCurrManager': YearsWithCurrManager,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobRole': job_roles.index(EmpJobRole),
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'BusinessTravelFrequency': travel_options.index(BusinessTravelFrequency),
        'EmpRelationshipSatisfaction': EmpRelationshipSatisfaction,
        'EmpJobInvolvement': EmpJobInvolvement
    }

    return pd.DataFrame([data])

# Batch CSV Upload Mode
if mode == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        preds = model.predict(input_df)
        input_df['PredictedPerformanceRating'] = preds
        st.success("Predictions complete!")
        st.dataframe(input_df)
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

# Manual Input Mode
else:
    input_df = get_user_input()
if st.button("Predict Performance Rating"):
    prediction_encoded = int(model.predict(input_df)[0])
    prediction = prediction_encoded + 2  # Convert back to original rating (2, 3, 4)

    st.subheader(f"Predicted Performance Rating: **{prediction}**")

    if prediction == 2:
        st.warning("Low performer. Recommend performance improvement plan or training.")
    elif prediction == 3:
        st.info("ℹAverage performer. Encourage growth through learning & development.")
    elif prediction == 4:
        st.success("High performer! Recommend bonus, promotion, or retention strategy.")
    else:
        st.error("Unexpected prediction value.")
