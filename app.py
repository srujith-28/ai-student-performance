import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Student Performance System", layout="centered")

st.title("AI-Based Student Academic Performance Analysis & Guidance System")

# Tabs
tab1, tab2 = st.tabs(["📂 Data & Model", "🎓 Student Analysis"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Dataset & Model Training")

    data = pd.read_excel("student_data.xlsx")

    st.write("Preview of Dataset:")
    st.dataframe(data.head())

    def classify_performance(row):
        if row['mid_1_marks'] < 12 or row['attendance'] < 65:
            return "Poor"
        else:
            return "Good"

    data['performance_status'] = data.apply(classify_performance, axis=1)

    X = data[['attendance', 'mid_1_marks', 'assignment_marks', 'quiz_marks', 'previous_gpa']]
    y = data['performance_status'].map({'Good': 1, 'Poor': 0})

    model = LogisticRegression()
    model.fit(X, y)

    st.success("AI model trained successfully.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Student Performance Analysis")

    video_links = {
        "Maths": "https://www.youtube.com/results?search_query=engineering+maths+mid1+important+topics",
        "Physics": "https://www.youtube.com/results?search_query=engineering+physics+mid1+important+topics",
        "Chemistry": "https://www.youtube.com/results?search_query=engineering+chemistry+mid1+important+topics",
        "DSA": "https://www.youtube.com/results?search_query=data+structures+mid1+important+topics"
    }

    student_id = st.text_input("Enter Student ID (e.g., S10)")

    if st.button("Analyze Student"):

        student_row = data[data['student_id'] == student_id]

        if student_row.empty:
            st.error("Student ID not found!")
        else:
            features = student_row[['attendance', 'mid_1_marks', 'assignment_marks', 'quiz_marks', 'previous_gpa']]

            probability = model.predict_proba(features)[0][1] * 100
            st.write(f"Performance Probability: {round(probability, 2)} %")

            if probability >= 75:
                st.success("Risk Level: LOW RISK (Good Performance)")
            elif probability >= 50:
                st.warning("Risk Level: MEDIUM RISK (Needs Improvement)")
            else:
                st.error("Risk Level: HIGH RISK (At Academic Risk)")

            total = (
                student_row['mid_1_marks'].values[0]
                + student_row['assignment_marks'].values[0]
                + student_row['quiz_marks'].values[0]
            )

            predicted_cgpa = round((total / 30) * 10, 2)
            st.write("Predicted CGPA:", predicted_cgpa)

            if predicted_cgpa < 7:
                st.error("You need to score at least 15+ in Mid-2 to reach good CGPA.")
            else:
                st.success("You are on track for a good CGPA.")

            subject = student_row['subjects'].values[0]

            if student_row['mid_1_marks'].values[0] < 12:
                st.warning(f"Weak Subject Detected: {subject}")
                st.markdown(
                    f"[Click here for YouTube videos]({video_links.get(subject)})"
                )

            st.subheader("Student Data")
            st.dataframe(student_row)
            st.subheader("📊 Academic Performance Overview")

            marks = [
                student_row['mid_1_marks'].values[0],
                student_row['assignment_marks'].values[0],
                student_row['quiz_marks'].values[0]
            ]

            labels = ["Mid-1", "Assignments", "Quiz"]

            chart_data = pd.DataFrame(
                {"Marks": marks},
                index=labels
            )

            st.bar_chart(chart_data)




