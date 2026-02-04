import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("AI-Based Student Academic Performance Analysis & Guidance System")

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

# Video recommendation dictionary
video_links = {
    "Maths": "https://www.youtube.com/results?search_query=engineering+maths+mid1+important+topics",
    "Physics": "https://www.youtube.com/results?search_query=engineering+physics+mid1+important+topics",
    "Chemistry": "https://www.youtube.com/results?search_query=engineering+chemistry+mid1+important+topics",
    "DSA": "https://www.youtube.com/results?search_query=data+structures+mid1+important+topics"
}

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

        # CGPA Prediction (simple logic)
        total = student_row['mid_1_marks'].values[0] + student_row['assignment_marks'].values[0] + student_row['quiz_marks'].values[0]
        predicted_cgpa = round((total / 30) * 10, 2)
        st.write("Predicted CGPA:", predicted_cgpa)

        # Required marks logic
        if predicted_cgpa < 7:
            st.error("You need to score at least 15+ in Mid-2 to reach good CGPA.")
        else:
            st.success("You are on track for a good CGPA.")

        # Weak subject & videos
       subject = student_row['subjects'].values[0]
if student_row['mid_1_marks'].values[0] < 12:
    st.warning(f"Weak Subject Detected: {subject}")
    st.write("Recommended Learning Resources:")
    st.markdown(f"[Click here for YouTube videos]({video_links.get(subject)})")

