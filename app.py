import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Academic Dashboard", layout="wide")
st.title("🎓 AI-Based Student Academic Performance System")

# -------------------------------
# FILE PATH
# -------------------------------
FILE_PATH = "student_data_multiple_subjects.xlsx"

# -------------------------------
# LOAD DATA (FIXED)
# -------------------------------
if os.path.exists(FILE_PATH):
    data = pd.read_excel(FILE_PATH)
    st.success("Loaded local dataset")
else:
    st.warning("Local dataset not found. Please upload file.")
    uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx"])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.success("Uploaded dataset loaded")
    else:
        st.stop()

# -------------------------------
# PDF GENERATOR
# -------------------------------
def generate_pdf(student_id, weak_subjects, risk_level):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Student Performance Report", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Student ID: {student_id}", styles['Normal']))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))

    if weak_subjects:
        elements.append(Paragraph("Weak Subjects:", styles['Heading3']))
        for sub in weak_subjects:
            elements.append(Paragraph(sub, styles['Normal']))
    else:
        elements.append(Paragraph("No weak subjects", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------
# LABELING
# -------------------------------
data['performance_status'] = data.apply(
    lambda row: "Poor" if row['mid_1_marks'] < 12 or row['attendance'] < 65 else "Good",
    axis=1
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
y = data['performance_status'].map({'Good':1,'Poor':0})

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📊 Dataset", "🎯 Analysis"])

# -------------------------------
# TAB 1: DATA + ADD STUDENT
# -------------------------------
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(data))
    col2.metric("Students", data['student_id'].nunique())

    st.divider()

    # -------------------------------
    # ADD STUDENT
    # -------------------------------
    st.markdown("## ➕ Add New Student Record")

    with st.form("add_student_form"):

        new_id = st.text_input("Student ID")
        new_subject = st.text_input("Subject")

        col1, col2, col3 = st.columns(3)

        attendance = col1.number_input("Attendance", 0, 100)
        mid = col2.number_input("Mid-1 Marks", 0, 25)
        assignment = col3.number_input("Assignment Marks", 0, 10)

        quiz = st.number_input("Quiz Marks", 0, 10)
        gpa = st.number_input("Previous GPA", 0.0, 10.0)

        submitted = st.form_submit_button("Add Student")

        if submitted:

            new_row = pd.DataFrame([{
                "student_id": new_id,
                "subject": new_subject,
                "attendance": attendance,
                "mid_1_marks": mid,
                "assignment_marks": assignment,
                "quiz_marks": quiz,
                "previous_gpa": gpa
            }])

            data = pd.concat([data, new_row], ignore_index=True)

            # Save ONLY if local file exists
            if os.path.exists(FILE_PATH):
                data.to_excel(FILE_PATH, index=False)
                st.success("✅ Student added and saved to Excel!")
            else:
                st.success("✅ Student added (not saved locally)")

# -------------------------------
# TAB 2: ANALYSIS
# -------------------------------
with tab2:

    student_id = st.text_input("Enter Student ID (e.g., S1)")

    if st.button("Analyze Student"):

        student_rows = data[data['student_id'] == student_id]

        if student_rows.empty:
            st.error("Student not found!")
        else:

            weak_subjects = []
            avg_marks = student_rows['mid_1_marks'].mean()

            st.markdown("## 📚 Subject-wise Analysis")

            for _, row in student_rows.iterrows():

                features = pd.DataFrame([[
                    row['attendance'],
                    row['mid_1_marks'],
                    row['assignment_marks'],
                    row['quiz_marks'],
                    row['previous_gpa']
                ]], columns=[
                    'attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa'
                ])

                prob = model.predict_proba(features)[0][1] * 100

                col1, col2, col3 = st.columns(3)

                col1.metric("Subject", row['subject'])
                col2.metric("Marks", row['mid_1_marks'])
                col3.metric("AI Score", f"{round(prob,2)}%")

                if row['mid_1_marks'] < avg_marks:
                    st.error("Needs Improvement")
                    weak_subjects.append(row['subject'])
                else:
                    st.success("Good Performance")

                st.divider()

            # -------------------------------
            # RISK SCORE
            # -------------------------------
            weak_count = len(weak_subjects)
            total = len(student_rows)
            ratio = weak_count / total

            st.markdown("## 🎯 Overall Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("Subjects", total)
            col2.metric("Weak Subjects", weak_count)

            if ratio < 0.3:
                risk_level = "LOW"
                col3.success("LOW")
            elif ratio < 0.6:
                risk_level = "MEDIUM"
                col3.warning("MEDIUM")
            else:
                risk_level = "HIGH"
                col3.error("HIGH")

            # -------------------------------
            # VIDEO RECOMMENDATION
            # -------------------------------
            if weak_subjects:
                st.markdown("## 🎥 Recommended Videos")

                for subject in weak_subjects:
                    query = f"{subject} important topics"
                    st.markdown(
                        f"[Watch {subject} videos](https://www.youtube.com/results?search_query={query})"
                    )

            else:
                st.success("No recommendations needed!")

            # -------------------------------
            # CHART
            # -------------------------------
            st.markdown("## 📈 Performance Chart")

            chart_data = student_rows[['subject','mid_1_marks']].set_index('subject')
            st.bar_chart(chart_data)

            # -------------------------------
            # PDF
            # -------------------------------
            st.markdown("## 📄 Download Report")

            pdf = generate_pdf(student_id, weak_subjects, risk_level)

            st.download_button(
                "Download Report",
                data=pdf,
                file_name=f"{student_id}_report.pdf",
                mime="application/pdf"
            )
