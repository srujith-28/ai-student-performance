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
# LABEL + MODEL
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

    # ADD
    st.markdown("## ➕ Add Student")

    sid = st.text_input("Student ID")
    subject = st.text_input("Subject")

    att = st.number_input("Attendance",0,100)
    mid = st.number_input("Mid Marks",0,25)
    ass = st.number_input("Assignment",0,10)
    quiz = st.number_input("Quiz",0,10)
    gpa = st.number_input("GPA",0.0,10.0)

    if st.button("Add"):
        new = pd.DataFrame([{
            "student_id":sid,
            "subject":subject,
            "attendance":att,
            "mid_1_marks":mid,
            "assignment_marks":ass,
            "quiz_marks":quiz,
            "previous_gpa":gpa
        }])
        data = pd.concat([data,new],ignore_index=True)
        st.success("Added!")

    st.divider()

    # UPDATE
    st.markdown("## ✏️ Update Student")

    uid = st.text_input("Enter Student ID to update")

    rows = data[data['student_id']==uid]

    for i,row in rows.iterrows():

        st.write(row['subject'])

        new_mid = st.number_input("New Mid Marks",0,25,int(row['mid_1_marks']),key=i)

        if st.button(f"Update {row['subject']}",key=f"u{i}"):
            data.loc[i,'mid_1_marks'] = new_mid
            st.success("Updated!")

    st.divider()

    # DELETE
    st.markdown("## ❌ Delete Student")

    did = st.text_input("Student ID to delete")

    if st.button("Delete"):
        data = data[data['student_id'] != did]
        st.success("Deleted!")

    st.divider()

    # DOWNLOAD UPDATED FILE
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
