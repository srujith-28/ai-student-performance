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

# -------------------------------
# SIMPLE LOGIN SYSTEM
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🎓 AI-Based Student Academic Performance System")

FILE_PATH = "student_data_multiple_subjects.xlsx"

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded_file = st.file_uploader("Upload Dataset (if file not found)", type=["xlsx"])

if os.path.exists(FILE_PATH):
    data = pd.read_excel(FILE_PATH)
elif uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
else:
    st.warning("Upload dataset to continue")
    st.stop()

# -------------------------------
# PDF
# -------------------------------
def generate_pdf(student_id, weak, risk):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Student Report", styles['Title']))
    elements.append(Paragraph(f"ID: {student_id}", styles['Normal']))
    elements.append(Paragraph(f"Risk: {risk}", styles['Normal']))

    for w in weak:
        elements.append(Paragraph(w, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------
# LABEL + MODEL
# -------------------------------
data['performance_status'] = data.apply(
    lambda r: "Poor" if r['mid_1_marks'] < 12 or r['attendance'] < 65 else "Good", axis=1
)

X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
y = data['performance_status'].map({'Good':1,'Poor':0})

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📊 Manage Data", "🎯 Analysis"])

# ===============================
# TAB 1: ADD / UPDATE / DELETE
# ===============================
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.divider()

    # ADD STUDENT
    st.markdown("## ➕ Add Student")

    with st.form("add"):

        sid = st.text_input("Student ID")
        subjects = ["Maths","Physics","Chemistry","DSA","English"]

        rows = []

        for sub in subjects:
            st.markdown(f"### {sub}")
            col1,col2,col3 = st.columns(3)

            att = col1.number_input("Attendance",0,100,key=sub+"a")
            mid = col2.number_input("Mid",0,25,key=sub+"m")
            ass = col3.number_input("Assignment",0,10,key=sub+"as")

            quiz = st.number_input("Quiz",0,10,key=sub+"q")
            gpa = st.number_input("GPA",0.0,10.0,key=sub+"g")

            rows.append({
                "student_id":sid,
                "subject":sub,
                "attendance":att,
                "mid_1_marks":mid,
                "assignment_marks":ass,
                "quiz_marks":quiz,
                "previous_gpa":gpa
            })

        if st.form_submit_button("Add Student"):
            data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
            if os.path.exists(FILE_PATH):
                data.to_excel(FILE_PATH,index=False)
            st.success("Added!")

    st.divider()

    # UPDATE
    st.markdown("## ✏️ Update Student")

    uid = st.text_input("Student ID to update")

    df = data[data['student_id']==uid]

    if not df.empty:

        for i,row in df.iterrows():

            st.markdown(f"### {row['subject']}")

            att = st.number_input("Attendance",0,100,int(row['attendance']),key="u"+str(i))
            mid = st.number_input("Mid",0,25,int(row['mid_1_marks']),key="m"+str(i))

            if st.button(f"Update {row['subject']}",key="btn"+str(i)):

                data.loc[i,'attendance']=att
                data.loc[i,'mid_1_marks']=mid

                if os.path.exists(FILE_PATH):
                    data.to_excel(FILE_PATH,index=False)

                st.success("Updated!")

    st.divider()

    # DELETE
    st.markdown("## ❌ Delete Student")

    did = st.text_input("Student ID to delete")

    if st.button("Delete"):
        data = data[data['student_id'] != did]

        if os.path.exists(FILE_PATH):
            data.to_excel(FILE_PATH,index=False)

        st.success("Deleted!")

# ===============================
# TAB 2: ANALYSIS
# ===============================
with tab2:

    sid = st.text_input("Enter Student ID")

    if st.button("Analyze"):

        df = data[data['student_id']==sid]

        if df.empty:
            st.error("Not found")
        else:

            weak=[]
            avg=df['mid_1_marks'].mean()

            for _,r in df.iterrows():

                prob = model.predict_proba(pd.DataFrame([[
                    r['attendance'],
                    r['mid_1_marks'],
                    r['assignment_marks'],
                    r['quiz_marks'],
                    r['previous_gpa']
                ]],columns=X.columns))[0][1]*100

                st.metric(r['subject'],r['mid_1_marks'])

                if r['mid_1_marks']<avg:
                    weak.append(r['subject'])
                    st.error("Weak")
                else:
                    st.success("Good")

                st.divider()

            ratio=len(weak)/len(df)

            risk="LOW" if ratio<0.3 else "MEDIUM" if ratio<0.6 else "HIGH"
            st.subheader(f"Risk: {risk}")

            for w in weak:
                st.markdown(f"[Learn {w}](https://www.youtube.com/results?search_query={w})")

            st.bar_chart(df[['subject','mid_1_marks']].set_index('subject'))

            pdf=generate_pdf(sid,weak,risk)
            st.download_button("Download Report",pdf,f"{sid}.pdf")
