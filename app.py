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

st.markdown("Upload dataset to analyze student performance and compare sections.")

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def classify_performance(row):
    if row['mid_1_marks'] < 12 or row['attendance'] < 65:
        return "Poor"
    else:
        return "Good"

def get_weak_students(df):
    return df[(df['mid_1_marks'] < 12) | (df['attendance'] < 65)]

def generate_pdf(student_id, weak_subjects):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Student Performance Report", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Student ID: {student_id}", styles['Normal']))
    elements.append(Spacer(1, 10))

    if weak_subjects:
        elements.append(Paragraph("Subjects needing improvement:", styles['Heading3']))
        for sub in weak_subjects:
            elements.append(Paragraph(sub, styles['Normal']))
    else:
        elements.append(Paragraph("All subjects performing well.", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

video_links = {
    "Maths": "https://www.youtube.com/results?search_query=engineering+maths",
    "Physics": "https://www.youtube.com/results?search_query=engineering+physics",
    "Chemistry": "https://www.youtube.com/results?search_query=engineering+chemistry",
    "DSA": "https://www.youtube.com/results?search_query=data+structures",
    "English": "https://www.youtube.com/results?search_query=communication+skills"
}

# --------------------------------------------------
# FILE UPLOAD (MAIN DATASET)
# --------------------------------------------------

uploaded_file = st.file_uploader("📂 Upload Main Student Dataset", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload dataset to begin analysis.")

else:
    data = pd.read_excel(uploaded_file)

    data['performance_status'] = data.apply(classify_performance, axis=1)

    X = data[['attendance','mid_1_marks','assignment_marks','quiz_marks','previous_gpa']]
    y = data['performance_status'].map({'Good':1,'Poor':0})

    model = LogisticRegression()
    model.fit(X,y)

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset Overview",
        "🎓 Student Analysis",
        "🏫 Section Comparison"
    ])

    # --------------------------------------------------
    # TAB 1
    # --------------------------------------------------
    with tab1:
        st.dataframe(data.head())

        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(data))
        col2.metric("Students", data['student_id'].nunique())

    # --------------------------------------------------
    # TAB 2 (STUDENT ANALYSIS)
    # --------------------------------------------------
    with tab2:

        st.subheader("Student Performance Analysis")

        student_id = st.text_input("Enter Student ID")

        if st.button("Analyze Student"):

            student_rows = data[data['student_id'] == student_id]

            if student_rows.empty:
                st.error("Student not found")

            else:
                weak_subjects = []

                for _, row in student_rows.iterrows():

                    features = pd.DataFrame([[
                        row['attendance'],
                        row['mid_1_marks'],
                        row['assignment_marks'],
                        row['quiz_marks'],
                        row['previous_gpa']
                    ]], columns=X.columns)

                    prob = model.predict_proba(features)[0][1]*100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Subject", row['subject'])
                    col2.metric("Marks", row['mid_1_marks'])
                    col3.metric("Performance", f"{round(prob,2)}%")

                    if prob < 50:
                        weak_subjects.append(row['subject'])
                        st.error("Needs Improvement")
                    elif prob < 75:
                        st.warning("Average")
                    else:
                        st.success("Good")

                    st.divider()

                if weak_subjects:
                    st.subheader("⚠ Weak Subjects")
                    for sub in weak_subjects:
                        with st.expander(sub):
                            st.markdown(f"[Learn {sub}]({video_links.get(sub)})")

                chart_data = student_rows[['subject','mid_1_marks']].set_index('subject')
                st.bar_chart(chart_data)

                pdf = generate_pdf(student_id, weak_subjects)
                st.download_button("Download Report", pdf, f"{student_id}.pdf")

    # --------------------------------------------------
    # TAB 3 (SECTION COMPARISON)
    # --------------------------------------------------
    with tab3:

        st.subheader("Multi-Section Comparison")

        uploaded_files = st.file_uploader(
            "Upload Section Files",
            type=["xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files:

            sections = {}
            summary = []

            for i, file in enumerate(uploaded_files):
                name = f"Section {chr(65+i)}"
                df = pd.read_excel(file)
                sections[name] = df

            for sec, df in sections.items():

                weak = get_weak_students(df)

                summary.append({
                    "Section": sec,
                    "Total": df['student_id'].nunique(),
                    "Weak": weak['student_id'].nunique(),
                    "Avg Marks": round(df['mid_1_marks'].mean(),2)
                })

            summary_df = pd.DataFrame(summary)

            st.dataframe(summary_df)

            best = summary_df.loc[summary_df['Avg Marks'].idxmax()]
            worst = summary_df.loc[summary_df['Weak'].idxmax()]

            col1, col2 = st.columns(2)
            col1.metric("Best Section", best["Section"])
            col2.metric("Most Weak Students", worst["Section"])

            st.bar_chart(summary_df.set_index("Section")["Weak"])

            for sec, df in sections.items():
                weak = get_weak_students(df)

                with st.expander(f"Weak Students - {sec}"):
                    st.dataframe(weak[['student_id','subject','mid_1_marks','attendance']])
