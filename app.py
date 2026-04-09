import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from io import BytesIO

st.set_page_config(page_title="AI Dashboard", layout="wide")

# -------------------------------
# LOGIN
# -------------------------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------------------
# TITLE
# -------------------------------
st.title("🎓 AI Student Performance Dashboard")

# -------------------------------
# UPLOAD DATA
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
# LABEL FUNCTION
# -------------------------------
def update_labels():
    st.session_state.data['performance_status'] = st.session_state.data.apply(
        lambda r: "Poor" if r['mid_1_marks'] < 12 or r['attendance'] < 65 else "Good",
        axis=1
    )

update_labels()

# -------------------------------
# MODEL
# -------------------------------
X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
y = data['performance_status'].map({'Good':1,'Poor':0})

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# DASHBOARD METRICS
# -------------------------------
st.subheader("📊 Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(data))
col2.metric("Students", data['student_id'].nunique())
col3.metric("Avg Marks", round(data['mid_1_marks'].mean(),2))

st.divider()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Manage Data", "🎯 Analysis", "📊 Comparison"])

# ===============================
# TAB 1: MANAGE DATA
# ===============================
with tab1:

    st.dataframe(data)

    st.divider()

    # ADD
    st.markdown("## ➕ Add Student")

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
            if sid.strip()=="":
                st.warning("Enter Student ID")
            else:
                st.session_state.backup = st.session_state.data.copy()
                st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame(rows)], ignore_index=True)
                update_labels()
                st.success("Student added")

    st.divider()

    # UPDATE
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

                st.session_state.data.loc[i,'attendance'] = att
                st.session_state.data.loc[i,'mid_1_marks'] = mid
                st.session_state.data.loc[i,'assignment_marks'] = ass
                st.session_state.data.loc[i,'quiz_marks'] = quiz
                st.session_state.data.loc[i,'previous_gpa'] = gpa

                update_labels()
                st.success("Updated")

    st.divider()

    # DELETE
    st.markdown("## ❌ Delete Student")

    did = st.text_input("Student ID", key="delete_id")

    if st.button("Delete", key="delete_btn"):
        if did.strip()=="":
            st.warning("Enter ID")
        elif did not in st.session_state.data['student_id'].values:
            st.error("Not found")
        else:
            st.session_state.backup = st.session_state.data.copy()
            st.session_state.data = st.session_state.data[st.session_state.data['student_id']!=did]
            st.success("Deleted")

    st.divider()

    # UNDO
    if st.button("↩️ Undo", key="undo_btn"):
        st.session_state.data = st.session_state.backup.copy()
        st.success("Undo done")

    st.divider()

    # DOWNLOAD
    output = BytesIO()
    st.session_state.data.to_excel(output,index=False)
    output.seek(0)

    st.download_button("📥 Download Excel",output,"updated_data.xlsx")

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

            st.markdown("## 🎥 Recommendations")
            for w in weak:
                st.markdown(f"[Learn {w}](https://www.youtube.com/results?search_query={w})")

            st.bar_chart(df[['subject','mid_1_marks']].set_index('subject'))

# ===============================
# TAB 3: COMPARISON
# ===============================
with tab3:

    st.markdown("## 📊 Comparison Module")

    option = st.radio(
        "Select Data Source",
        ["Use Existing Data", "Upload New Dataset"],
        key="compare_option"
    )

    if option == "Use Existing Data":
        comp_data = st.session_state.data
    else:
        uploaded_comp = st.file_uploader("Upload Comparison Dataset", type=["xlsx"], key="comp_upload")

        if uploaded_comp is not None:
            comp_data = pd.read_excel(uploaded_comp)
        else:
            st.warning("Upload dataset to continue")
            st.stop()

    st.dataframe(comp_data)

    students = comp_data['student_id'].unique()

    selected_students = st.multiselect(
        "Select Students",
        students,
        default=list(students)[:3],
        key="compare_students_tab3"
    )

    if selected_students:

        filtered = comp_data[comp_data['student_id'].isin(selected_students)]

        avg_marks = filtered.groupby('student_id')['mid_1_marks'].mean().reset_index()

        st.markdown("### 📊 Average Comparison")
        st.bar_chart(avg_marks.set_index('student_id'))

        threshold = avg_marks['mid_1_marks'].mean()

        weak_students = avg_marks[avg_marks['mid_1_marks'] < threshold]
        top_students = avg_marks[avg_marks['mid_1_marks'] >= threshold]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚠️ Weak Students")
            for _, r in weak_students.iterrows():
                st.error(f"{r['student_id']} → {round(r['mid_1_marks'],2)}")

        with col2:
            st.markdown("### 🏆 Top Students")
            for _, r in top_students.iterrows():
                st.success(f"{r['student_id']} → {round(r['mid_1_marks'],2)}")

        st.metric("Class Average", round(threshold,2))
