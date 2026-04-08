import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🎓 AI Student Performance System")

# -------------------------------
# UPLOAD DATASET
# -------------------------------
uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if uploaded_file is None:
    st.warning("Upload dataset to continue")
    st.stop()

# -------------------------------
# SESSION STATE
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.read_excel(uploaded_file)

if "backup" not in st.session_state:
    st.session_state.backup = st.session_state.data.copy()

data = st.session_state.data

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
# TAB 1: MANAGE DATA
# ===============================
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(data)

    st.divider()

    # -------------------------------
    # ADD STUDENT
    # -------------------------------
    st.markdown("## ➕ Add Student (Multiple Subjects)")

    with st.form("add_form"):

        sid = st.text_input("Student ID", key="add_id")

        subjects = ["Maths","Physics","Chemistry","DSA","English"]
        rows = []

        for sub in subjects:

            st.markdown(f"### {sub}")

            col1,col2,col3 = st.columns(3)

            att = col1.number_input(f"{sub} Attendance",0,100,key=f"{sub}_a")
            mid = col2.number_input(f"{sub} Mid",0,25,key=f"{sub}_m")
            ass = col3.number_input(f"{sub} Assignment",0,10,key=f"{sub}_as")

            quiz = st.number_input(f"{sub} Quiz",0,10,key=f"{sub}_q")
            gpa = st.number_input(f"{sub} GPA",0.0,10.0,key=f"{sub}_g")

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

            if sid.strip() == "":
                st.warning("⚠️ Enter Student ID")
            else:
                st.session_state.backup = st.session_state.data.copy()

                new_df = pd.DataFrame(rows)
                st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)

                st.success("✅ Student added")

    st.divider()

    # -------------------------------
    # UPDATE
    # -------------------------------
    st.markdown("## ✏️ Update Student")

    uid = st.text_input("Enter Student ID", key="update_id")

    df = st.session_state.data[st.session_state.data['student_id']==uid]

    if not df.empty:

        for i,row in df.iterrows():

            st.markdown(f"### {row['subject']}")

            col1,col2,col3 = st.columns(3)

            att = col1.number_input("Attendance",0,100,int(row['attendance']),key=f"att_{i}")
            mid = col2.number_input("Mid",0,25,int(row['mid_1_marks']),key=f"mid_{i}")
            ass = col3.number_input("Assignment",0,10,int(row['assignment_marks']),key=f"ass_{i}")

            quiz = st.number_input("Quiz",0,10,int(row['quiz_marks']),key=f"quiz_{i}")
            gpa = st.number_input("GPA",0.0,10.0,float(row['previous_gpa']),key=f"gpa_{i}")

            if st.button(f"Update {row['subject']}", key=f"btn_{i}"):

                st.session_state.backup = st.session_state.data.copy()

                st.session_state.data.loc[i] = [
                    row['student_id'], row['subject'],
                    att, mid, ass, quiz, gpa
                ]

                st.success("✅ Updated")

    st.divider()

    # -------------------------------
    # DELETE
    # -------------------------------
    st.markdown("## ❌ Delete Student")

    did = st.text_input("Student ID to delete", key="delete_id")

    if st.button("Delete", key="delete_btn"):

        if did.strip() == "":
            st.warning("⚠️ Enter Student ID")
        elif did not in st.session_state.data['student_id'].values:
            st.error("❌ Student not found")
        else:
            st.session_state.backup = st.session_state.data.copy()

            st.session_state.data = st.session_state.data[
                st.session_state.data['student_id'] != did
            ]

            st.success("✅ Deleted")

    st.divider()

    # -------------------------------
    # UNDO
    # -------------------------------
    if st.button("↩️ Undo", key="undo_btn"):
        st.session_state.data = st.session_state.backup.copy()
        st.success("Undo done")

    st.divider()

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    st.markdown("## 📥 Download Updated Excel")

    output = BytesIO()
    st.session_state.data.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Excel",
        output,
        "updated_student_data.xlsx",
        key="download_btn"
    )

# ===============================
# TAB 2: ANALYSIS
# ===============================
with tab2:

    sid = st.text_input("Enter Student ID", key="analysis_id")

    if st.button("Analyze", key="analyze_btn"):

        df = st.session_state.data[st.session_state.data['student_id']==sid]

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

                st.metric(r['subject'], r['mid_1_marks'])

                if r['mid_1_marks'] < avg:
                    weak.append(r['subject'])
                    st.error("Weak")
                else:
                    st.success("Good")

            risk = "LOW" if len(weak)==0 else "MEDIUM" if len(weak)<3 else "HIGH"
            st.subheader(f"Risk Level: {risk}")

            for w in weak:
                st.markdown(f"[Learn {w}](https://www.youtube.com/results?search_query={w})")

            st.bar_chart(df[['subject','mid_1_marks']].set_index('subject'))
