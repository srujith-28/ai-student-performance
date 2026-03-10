import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="AI Academic Dashboard", layout="wide")

st.title("🎓 AI-Based Student Academic Performance Dashboard")

st.markdown(
"""
Upload a student dataset to analyze academic performance across multiple subjects.
The system uses AI to detect weak subjects and recommend learning resources.
"""
)

# --------------------------------------------------
# RULE BASED LABEL FUNCTION
# --------------------------------------------------

def classify_performance(row):
    if row['mid_1_marks'] < 12 or row['attendance'] < 65:
        return "Poor"
    else:
        return "Good"

# --------------------------------------------------
# PDF REPORT GENERATOR
# --------------------------------------------------

def generate_pdf(student_id, weak_subjects):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Student Performance Report", styles['Title']))
    elements.append(Spacer(1,20))
    elements.append(Paragraph(f"Student ID: {student_id}", styles['Normal']))
    elements.append(Spacer(1,10))

    if weak_subjects:
        elements.append(Paragraph("Subjects requiring improvement:", styles['Heading3']))
        for subject in weak_subjects:
            elements.append(Paragraph(subject, styles['Normal']))
    else:
        elements.append(Paragraph("All subjects performing well.", styles['Normal']))

    doc.build(elements)

    buffer.seek(0)

    return buffer


# --------------------------------------------------
# VIDEO RESOURCES
# --------------------------------------------------

video_links = {
    "Maths": "https://www.youtube.com/results?search_query=engineering+maths+important+topics",
    "Physics": "https://www.youtube.com/results?search_query=engineering+physics+important+topics",
    "Chemistry": "https://www.youtube.com/results?search_query=engineering+chemistry+important+topics",
    "DSA": "https://www.youtube.com/results?search_query=data+structures+important+topics",
    "English": "https://www.youtube.com/results?search_query=english+communication+skills"
}


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader("📂 Upload Student Dataset (Excel)", type=["xlsx"])

if uploaded_file is None:

    st.info("Please upload the student dataset to start analysis.")

else:

    data = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully")

    # --------------------------------------------------
    # CREATE LABELS
    # --------------------------------------------------

    data['performance_status'] = data.apply(classify_performance, axis=1)

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
    y = data['performance_status'].map({'Good':1,'Poor':0})

    model = LogisticRegression()
    model.fit(X,y)

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    tab1, tab2 = st.tabs(["📊 Dataset Overview","🎓 Student Analysis"])

    # --------------------------------------------------
    # DATASET TAB
    # --------------------------------------------------

    with tab1:

        st.markdown("## Dataset Preview")

        st.dataframe(data.head())

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric("Total Records", len(data))
        col2.metric("Unique Students", data['student_id'].nunique())


    # --------------------------------------------------
    # STUDENT ANALYSIS TAB
    # --------------------------------------------------

    with tab2:

        st.markdown("## Student Performance Analysis")

        student_id = st.text_input("Enter Student ID (Example: S1)")

        if st.button("Analyze Student"):

            student_rows = data[data['student_id'] == student_id]

            if student_rows.empty:

                st.error("Student ID not found")

            else:

                weak_subjects = []

                st.markdown("### Subject-wise AI Evaluation")

                for index, row in student_rows.iterrows():

                    features = pd.DataFrame([[
                        row['attendance'],
                        row['mid_1_marks'],
                        row['assignment_marks'],
                        row['quiz_marks'],
                        row['previous_gpa']
                    ]], columns=[
                        'attendance',
                        'mid_1_marks',
                        'assignment_marks',
                        'quiz_marks',
                        'previous_gpa'
                    ])

                    probability = model.predict_proba(features)[0][1] * 100

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Subject", row['subject'])
                    col2.metric("Mid-1 Marks", row['mid_1_marks'])
                    col3.metric("Performance Probability", f"{round(probability,2)}%")

                    if probability >= 75:
                        st.success("Good Performance")

                    elif probability >= 50:
                        st.warning("Average Performance")

                    else:
                        st.error("Needs Improvement")
                        weak_subjects.append(row['subject'])

                    st.divider()

                # --------------------------------------------------
                # WEAK SUBJECT SECTION
                # --------------------------------------------------

                if weak_subjects:

                    st.markdown("## ⚠ Subjects Requiring Improvement")

                    for subject in weak_subjects:

                        with st.expander(f"Improve {subject}"):

                            st.write("Recommended learning resources:")

                            st.markdown(
                                f"[Watch videos for {subject}]({video_links.get(subject)})"
                            )

                else:

                    st.success("🎉 All subjects are performing well.")

                # --------------------------------------------------
                # CHART
                # --------------------------------------------------

                st.markdown("## 📈 Subject Marks Comparison")

                chart_data = student_rows[['subject','mid_1_marks']]
                chart_data = chart_data.set_index('subject')

                st.bar_chart(chart_data)

                st.divider()

                # --------------------------------------------------
                # PDF REPORT
                # --------------------------------------------------

                st.markdown("## 📄 Download Performance Report")

                pdf = generate_pdf(student_id, weak_subjects)

                st.download_button(
                    label="Download Report",
                    data=pdf,
                    file_name=f"{student_id}_report.pdf",
                    mime="application/pdf"
                )
