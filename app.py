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

FILE_PATH = "student_data_multiple_subjects.xlsx"

# -------------------------------
# LOAD DATA
# -------------------------------
if os.path.exists(FILE_PATH):
    data = pd.read_excel(FILE_PATH)
    st.success("Dataset loaded")
else:
    st.error("Dataset not found!")
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
# MODEL
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
# TAB 1: ADD + UPDATE
# -------------------------------
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.divider()

    # ===============================
    # ADD NEW STUDENT (MULTI SUBJECT)
    # ===============================
    st.markdown("## ➕ Add New Student")

    with st.form("add_student"):

        student_id = st.text_input("Student ID")

        subjects = ["Maths", "Physics", "Chemistry", "DSA", "English"]
        rows = []

        for subject in subjects:

            st.markdown(f"### {subject}")

            col1, col2, col3 = st.columns(3)

            attendance = col1.number_input(f"{subject} Attendance", 0, 100, key=f"{subject}_a")
            mid = col2.number_input(f"{subject} Mid", 0, 25, key=f"{subject}_m")
            assignment = col3.number_input(f"{subject} Assignment", 0, 10, key=f"{subject}_as")

            quiz = st.number_input(f"{subject} Quiz", 0, 10, key=f"{subject}_q")
            gpa = st.number_input(f"{subject} GPA", 0.0, 10.0, key=f"{subject}_g")

            rows.append({
                "student_id": student_id,
                "subject": subject,
                "attendance": attendance,
                "mid_1_marks": mid,
                "assignment_marks": assignment,
                "quiz_marks": quiz,
                "previous_gpa": gpa
            })

            st.divider()

        if st.form_submit_button("Add Student"):
            new_df = pd.DataFrame(rows)
            data = pd.concat([data, new_df], ignore_index=True)
            data.to_excel(FILE_PATH, index=False)
            st.success("✅ Student added successfully!")

    st.divider()

    # ===============================
    # UPDATE EXISTING STUDENT
    # ===============================
    st.markdown("## ✏️ Update Student Data")

    update_id = st.text_input("Enter Student ID to Update")

    student_data = data[data['student_id'] == update_id]

    if not student_data.empty:

        for idx, row in student_data.iterrows():

            st.markdown(f"### {row['subject']}")

            col1, col2, col3 = st.columns(3)

            attendance = col1.number_input("Attendance", 0, 100, int(row['attendance']), key=f"u_att_{idx}")
            mid = col2.number_input("Mid Marks", 0, 25, int(row['mid_1_marks']), key=f"u_mid_{idx}")
            assignment = col3.number_input("Assignment", 0, 10, int(row['assignment_marks']), key=f"u_ass_{idx}")

            quiz = st.number_input("Quiz", 0, 10, int(row['quiz_marks']), key=f"u_quiz_{idx}")
            gpa = st.number_input("GPA", 0.0, 10.0, float(row['previous_gpa']), key=f"u_gpa_{idx}")

            if st.button(f"Update {row['subject']}", key=f"btn_{idx}"):

                data.loc[idx, 'attendance'] = attendance
                data.loc[idx, 'mid_1_marks'] = mid
                data.loc[idx, 'assignment_marks'] = assignment
                data.loc[idx, 'quiz_marks'] = quiz
                data.loc[idx, 'previous_gpa'] = gpa

                data.to_excel(FILE_PATH, index=False)

                st.success(f"✅ {row['subject']} updated!")

    else:
        st.info("Enter valid Student ID")

# -------------------------------
# TAB 2: ANALYSIS
# -------------------------------
with tab2:

    student_id = st.text_input("Enter Student ID")

    if st.button("Analyze"):

        student_rows = data[data['student_id'] == student_id]

        if student_rows.empty:
            st.error("Student not found!")
        else:

            weak = []
            avg = student_rows['mid_1_marks'].mean()

            for _, row in student_rows.iterrows():

                prob = model.predict_proba(pd.DataFrame([[
                    row['attendance'],
                    row['mid_1_marks'],
                    row['assignment_marks'],
                    row['quiz_marks'],
                    row['previous_gpa']
                ]], columns=X.columns))[0][1] * 100

                st.write(f"### {row['subject']}")
                st.metric("Marks", row['mid_1_marks'])
                st.metric("AI Score", round(prob,2))

                if row['mid_1_marks'] < avg:
                    st.error("Weak")
                    weak.append(row['subject'])
                else:
                    st.success("Good")

                st.divider()

            ratio = len(weak)/len(student_rows)

            if ratio < 0.3:
                risk = "LOW"
                st.success("LOW RISK")
            elif ratio < 0.6:
                risk = "MEDIUM"
                st.warning("MEDIUM RISK")
            else:
                risk = "HIGH"
                st.error("HIGH RISK")

            if weak:
                for sub in weak:
                    st.markdown(f"[Watch {sub}](https://www.youtube.com/results?search_query={sub}+important+topics)")

            chart = student_rows[['subject','mid_1_marks']].set_index('subject')
            st.bar_chart(chart)

            pdf = generate_pdf(student_id, weak, risk)

            st.download_button("Download Report", pdf, f"{student_id}.pdf")
