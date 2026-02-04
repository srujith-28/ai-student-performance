import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("AI-Based Student Academic Performance Analysis System")

# Load dataset
data = pd.read_excel("student_data.xlsx")

# Rule-based labeling
def classify_performance(row):
    if row['mid_1_marks'] < 12 or row['attendance'] < 65:
        return "Poor"
    else:
        return "Good"

data['performance_status'] = data.apply(classify_performance, axis=1)

# Features and target
X = data[['attendance', 'mid_1_marks', 'assignment_marks', 'quiz_marks', 'previous_gpa']]
y = data['performance_status'].map({'Good':1, 'Poor':0})

# Train model
model = LogisticRegression()
model.fit(X, y)

# UI
student_id = st.text_input("Enter Student ID (e.g., S10)")

if st.button("Analyze Student"):
    student_row = data[data['student_id'] == student_id]

    if student_row.empty:
        st.error("Student ID not found!")
    else:
        features = student_row[['attendance', 'mid_1_marks', 'assignment_marks', 'quiz_marks', 'previous_gpa']]
        prediction = model.predict(features)

        st.subheader("Analysis Result")

        if prediction[0] == 1:
            st.success("Performance Status: GOOD")
        else:
            st.warning("Performance Status: AT RISK (POOR)")

        st.write("Student Details:")
        st.dataframe(student_row)
