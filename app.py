import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Student Performance System", layout="centered")

st.title("AI-Based Student Academic Performance Analysis & Guidance System")

# -----------------------------
# Load dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload Student Data", type=["xlsx"])

if uploaded_file is not None:

    data = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully")

    # rule-based classification
    data['performance_status'] = data.apply(classify_performance, axis=1)

    # train model
    X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
    y = data['performance_status'].map({'Good':1,'Poor':0})

    model = LogisticRegression()
    model.fit(X,y)

    tab1, tab2 = st.tabs(["Dataset Overview","Student Analysis"])

    with tab1:
        st.dataframe(data.head())

    with tab2:
        student_id = st.text_input("Enter Student ID")

        if st.button("Analyze Student"):
            # student analysis code here

# -----------------------------
# Rule-based classification
# -----------------------------
def classify_performance(row):
    if row['mid_1_marks'] < 12 or row['attendance'] < 65:
        return "Poor"
    else:
        return "Good"

data['performance_status'] = data.apply(classify_performance, axis=1)

# -----------------------------
# Train AI model
# -----------------------------
X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
y = data['performance_status'].map({'Good':1,'Poor':0})

model = LogisticRegression()
model.fit(X,y)

# -----------------------------
# YouTube recommendation links
# -----------------------------
video_links = {
    "Maths": "https://www.youtube.com/results?search_query=engineering+maths+important+topics",
    "Physics": "https://www.youtube.com/results?search_query=engineering+physics+important+topics",
    "Chemistry": "https://www.youtube.com/results?search_query=engineering+chemistry+important+topics",
    "DSA": "https://www.youtube.com/results?search_query=data+structures+important+topics",
    "English": "https://www.youtube.com/results?search_query=english+communication+skills"
}

# -----------------------------
# PDF Report Generator
# -----------------------------
def generate_pdf(student_id, weak_subjects):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Student Academic Performance Report", styles['Title']))
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

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Dataset Overview","Student Analysis"])

# -----------------------------
# Tab 1 : Dataset
# -----------------------------
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.write("Total Records:", len(data))

# -----------------------------
# Tab 2 : Student Analysis
# -----------------------------
with tab2:

    st.subheader("Student Performance Analysis")

    student_id = st.text_input("Enter Student ID (example: S1)")

    if st.button("Analyze Student"):

        student_rows = data[data['student_id'] == student_id]

        if student_rows.empty:

            st.error("Student ID not found!")

        else:

            weak_subjects = []

            st.subheader("Subject-wise Performance")

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

                probability = model.predict_proba(features)[0][1]*100

                st.write(f"Subject: {row['subject']}")
                st.write(f"Performance Probability: {round(probability,2)}%")

                if probability >= 75:
                    st.success("Good Performance")
                elif probability >= 50:
                    st.warning("Average Performance")
                else:
                    st.error("Needs Improvement")
                    weak_subjects.append(row['subject'])

                st.write("---")

            # -----------------------------
            # Weak Subject Recommendations
            # -----------------------------
            if weak_subjects:

                st.subheader("⚠ Subjects Requiring Improvement")

                for subject in weak_subjects:

                    st.warning(subject)

                    st.markdown(
                        f"[Recommended Videos for {subject}]({video_links.get(subject)})"
                    )

            else:

                st.success("All subjects are performing well.")

            # -----------------------------
            # Chart
            # -----------------------------
            chart_data = student_rows[['subject','mid_1_marks']]
            chart_data = chart_data.set_index('subject')

            st.subheader("Marks Comparison Across Subjects")
            st.bar_chart(chart_data)

            # -----------------------------
            # Download PDF
            # -----------------------------
            pdf = generate_pdf(student_id, weak_subjects)

            st.download_button(
                label="Download Performance Report",
                data=pdf,
                file_name=f"{student_id}_performance_report.pdf",
                mime="application/pdf"
            )



