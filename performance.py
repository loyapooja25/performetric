import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_90_modelpp.pkl")

# Manual categorical mappings (reverse from encoding)
gender_options = ['Male', 'Female']
edu_background = ['Marketing', 'Life Sciences', 'Human Resources', 'Medical', 'Other', 'Technical Degree']
marital_status = ['Single', 'Married', 'Divorced']
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

# Upload or manual entry
mode = st.radio("Select input mode", ["Upload CSV file", "Enter manually"])

def get_user_input():
    st.subheader("Enter employee data:")
    Age = st.slider("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", gender_options)
    EducationBackground = st.selectbox("Education Background", edu_background)
    MaritalStatus = st.selectbox("Marital Status", marital_status)
    EmpDepartment = st.selectbox("Department", departments)
    EmpJobRole = st.selectbox("Job Role", job_roles)
    BusinessTravelFrequency = st.selectbox("Business Travel Frequency", travel_options)
    OverTime = st.selectbox("OverTime", overtime_options)
    TotalWorkExperienceInYears = st.slider("Total Work Experience (Years)", 0, 40, 5)
    ExperienceYearsAtThisCompany = st.slider("Years at Company", 0, 30, 3)
    ExperienceYearsInCurrentRole = st.slider("Years in Current Role", 0, 30, 2)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
    YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 2)
    EmpEnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", 0, 25, 12)
    EmpWorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)

    data = {
        'Age': Age,
        'Gender': gender_options.index(Gender),
        'EducationBackground': edu_background.index(EducationBackground),
        'MaritalStatus': marital_status.index(MaritalStatus),
        'EmpDepartment': departments.index(EmpDepartment),
        'EmpJobRole': job_roles.index(EmpJobRole),
        'BusinessTravelFrequency': travel_options.index(BusinessTravelFrequency),
        'OverTime': overtime_options.index(OverTime),
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'EmpWorkLifeBalance': EmpWorkLifeBalance
    }

    return pd.DataFrame([data])

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
else:
    input_df = get_user_input()
    if st.button("Predict Performance Rating"):
        prediction = int(model.predict(input_df)[0])
        st.subheader(f"Predicted Performance Rating: **{prediction}**")
        if prediction == 2:
            st.warning("Low performance detected. Recommend performance improvement plan or training support.")
        elif prediction == 3:
            st.info(" Average performer. Encourage continued growth through learning & development.")
        elif prediction == 4:
            st.success("High performer! Consider promotion, bonus, or retention strategy.")
        else:
            st.error(" Unexpected value predicted. Check the model or input.")

