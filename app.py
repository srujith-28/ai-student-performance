import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("🎓 AI Student Performance System")

# -------------------------------
# UPLOAD DATASET
# -------------------------------
uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if uploaded_file is None:
    st.warning("Upload dataset to continue")
    st.stop()

data = pd.read_excel(uploaded_file)

# -------------------------------
# MODEL
# -------------------------------
data['performance_status'] = data.apply(
    lambda r: "Poor" if r['mid_1_marks'] < 12 or r['attendance'] < 65 else "Good",
    axis=1
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
    st.dataframe(data)

    st.divider()

    # ===============================
    # ADD MULTI SUBJECT STUDENT
    # ===============================
    st.markdown("## ➕ Add Student (Multiple Subjects)")

    with st.form("add_student"):

        sid = st.text_input("Student ID")

        subjects = ["Maths","Physics","Chemistry","DSA","English"]
        rows = []

        for sub in subjects:

            st.markdown(f"### {sub}")

            col1,col2,col3 = st.columns(3)

            att = col1.number_input(f"{sub} Attendance",0,100,key=sub+"a")
            mid = col2.number_input(f"{sub} Mid Marks",0,25,key=sub+"m")
            ass = col3.number_input(f"{sub} Assignment",0,10,key=sub+"as")

            quiz = st.number_input(f"{sub} Quiz",0,10,key=sub+"q")
            gpa = st.number_input(f"{sub} GPA",0.0,10.0,key=sub+"g")

            rows.append({
                "student_id":sid,
                "subject":sub,
                "attendance":att,
                "mid_1_marks":mid,
                "assignment_marks":ass,
                "quiz_marks":quiz,
                "previous_gpa":gpa
            })

            st.divider()

        if st.form_submit_button("Add Student"):
            data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
            st.success("✅ Student added!")

    st.divider()

    # ===============================
    # UPDATE
    # ===============================
    st.markdown("## ✏️ Update Student")

    uid = st.text_input("Enter Student ID to update")

    df = data[data['student_id']==uid]

    if not df.empty:

        for i,row in df.iterrows():

            st.markdown(f"### {row['subject']}")

            new_mid = st.number_input("Mid Marks",0,25,int(row['mid_1_marks']),key=f"mid{i}")

            if st.button(f"Update {row['subject']}",key=f"btn{i}"):
                data.loc[i,'mid_1_marks'] = new_mid
                st.success("Updated!")

    st.divider()

    # ===============================
    # DELETE
    # ===============================
    st.markdown("## ❌ Delete Student")

    did = st.text_input("Student ID to delete")

    if st.button("Delete"):
        data = data[data['student_id'] != did]
        st.success("Deleted!")

    st.divider()

    # ===============================
    # DOWNLOAD UPDATED DATA
    # ===============================
    st.markdown("## 💾 Download Updated Dataset")

    output = BytesIO()
    data.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Updated Excel",
        output,
        "updated_student_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===============================
# TAB 2: ANALYSIS
# ===============================
with tab2:

    sid = st.text_input("Enter Student ID")

    if st.button("Analyze"):

        df = data[data['student_id']==sid]

        if df.empty:
            st.error("Student not found")
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

            risk="LOW" if len(weak)==0 else "MEDIUM" if len(weak)<3 else "HIGH"
            st.subheader(f"Risk: {risk}")

            for w in weak:
                st.markdown(f"[Learn {w}](https://www.youtube.com/results?search_query={w})")

            st.bar_chart(df[['subject','mid_1_marks']].set_index('subject'))
