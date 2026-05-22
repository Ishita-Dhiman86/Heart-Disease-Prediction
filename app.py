import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
model = joblib.load("heart_model.pkl")

# --------------------------------
# BACKGROUND + CSS
# --------------------------------
page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1516549655169-df83a0774514?q=80&w=2070&auto=format&fit=crop");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.main-box{
background: rgba(0,0,0,0.75);
padding: 40px;
border-radius: 25px;
box-shadow: 0px 0px 30px rgba(0,0,0,0.6);
}

.title{
text-align:center;
font-size:60px;
font-weight:bold;
color:#ff4b4b;
}

.subtitle{
text-align:center;
font-size:22px;
color:white;
margin-bottom:30px;
}

.stButton>button{
width:100%;
height:60px;
border:none;
border-radius:15px;
background:linear-gradient(to right,#ff416c,#ff4b2b);
color:white;
font-size:22px;
font-weight:bold;
}

.result-success{
background: linear-gradient(to right,#16a34a,#22c55e);
padding:25px;
border-radius:20px;
text-align:center;
font-size:35px;
font-weight:bold;
color:white;
}

.result-danger{
background: linear-gradient(to right,#dc2626,#ef4444);
padding:25px;
border-radius:20px;
text-align:center;
font-size:35px;
font-weight:bold;
color:white;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# --------------------------------
# MAIN BOX
# --------------------------------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Smart Health Care System</div>',
    unsafe_allow_html=True
)

# --------------------------------
# PATIENT NAME
# --------------------------------
patient_name = st.text_input("👤 Enter Patient Name")

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------
# INPUT SECTION
# --------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 1, 100, 30)

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    sex_value = 0 if sex == "Female" else 1

    chest_pain = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4]
    )

    bp = st.slider(
        "Blood Pressure",
        80,
        200,
        120
    )

    cholesterol = st.slider(
        "Cholesterol",
        100,
        400,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

with col2:

    ekg = st.selectbox(
        "EKG Results",
        [0, 1, 2]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    st_depression = st.slider(
        "ST Depression",
        0.0,
        6.0,
        1.0
    )

    slope = st.selectbox(
        "Slope of ST",
        [1, 2, 3]
    )

    vessels = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thallium = st.selectbox(
        "Thallium Test",
        [1, 2, 3, 4]
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
if st.button("🔍 Predict Heart Disease"):

    input_data = np.array([[
        age,
        sex_value,
        chest_pain,
        bp,
        cholesterol,
        fbs,
        ekg,
        max_hr,
        exercise_angina,
        st_depression,
        slope,
        vessels,
        thallium
    ]])

    # --------------------------------
    # PREDICTION
    # --------------------------------
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")

    # --------------------------------
    # RESULT
    # --------------------------------
    if prediction[0] == 1:

        result_text = "Heart Disease Detected"

        st.markdown(
            '<div class="result-danger">💔 Heart Disease Found</div>',
            unsafe_allow_html=True
        )

        st.error("Please consult a doctor immediately.")

    else:

        result_text = "No Heart Disease Detected"

        st.markdown(
            '<div class="result-success">🎉 No Heart Disease Found</div>',
            unsafe_allow_html=True
        )

        st.success("Your heart looks healthy ❤️")

    # --------------------------------
    # RISK SCORE
    # --------------------------------
    st.progress(int(probability))

    st.markdown(
        f"<h2 style='text-align:center;color:white;'>Risk Score: {probability:.2f}%</h2>",
        unsafe_allow_html=True
    )

    # --------------------------------
    # PIE CHART
    # --------------------------------
    st.markdown("## 📊 Prediction Analysis")

    labels = ['Healthy', 'Risk']
    sizes = [100 - probability, probability]

    fig, ax = plt.subplots()

    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

    # --------------------------------
    # PDF REPORT
    # --------------------------------
    pdf = FPDF()

    pdf.add_page()

    # TITLE
    pdf.set_font("Arial", "B", 20)

    pdf.cell(
        200,
        15,
        "HEART HEALTH MEDICAL REPORT",
        ln=True,
        align='C'
    )

    pdf.ln(10)

    # --------------------------------
    # PATIENT DETAILS
    # --------------------------------
    pdf.set_font("Arial", "", 12)

    pdf.cell(
        200,
        10,
        f"Patient Name: {patient_name}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        f"Date: {datetime.now().strftime('%d-%m-%Y')}",
        ln=True
    )

    pdf.ln(10)

    # --------------------------------
    # TABLE HEADER
    # --------------------------------
    pdf.set_fill_color(0, 128, 128)

    pdf.set_text_color(255,255,255)

    pdf.cell(80, 10, "Test", 1, 0, 'C', True)
    pdf.cell(50, 10, "Result", 1, 0, 'C', True)
    pdf.cell(50, 10, "Status", 1, 1, 'C', True)

    pdf.set_text_color(0,0,0)

    # --------------------------------
    # TABLE DATA
    # --------------------------------
    tests = [

        ("Age", str(age), "NORMAL"),

        ("Sex", sex, "NORMAL"),

        ("Chest Pain Type",
         str(chest_pain),
         "CHECK"),

        ("Blood Pressure",
         str(bp),
         "HIGH" if bp > 140 else "NORMAL"),

        ("Cholesterol",
         str(cholesterol),
         "HIGH" if cholesterol > 240 else "NORMAL"),

        ("Fasting Blood Sugar",
         str(fbs),
         "HIGH" if fbs == 1 else "NORMAL"),

        ("EKG Results",
         str(ekg),
         "CHECK"),

        ("Maximum Heart Rate",
         str(max_hr),
         "NORMAL"),

        ("Exercise Induced Angina",
         str(exercise_angina),
         "CHECK"),

        ("ST Depression",
         str(st_depression),
         "CHECK"),

        ("Number of Major Vessels",
         str(vessels),
         "CHECK")

    ]

    # --------------------------------
    # ADD TABLE ROWS
    # --------------------------------
    for test, result, status in tests:

        pdf.cell(80, 10, test, 1)

        pdf.cell(50, 10, result, 1)

        if status == "NORMAL":

            pdf.set_text_color(0, 128, 0)

        else:

            pdf.set_text_color(255, 0, 0)

        pdf.cell(50, 10, status, 1, 1)

        pdf.set_text_color(0, 0, 0)

    # --------------------------------
    # FINAL RESULT
    # --------------------------------
    pdf.ln(15)

    pdf.set_font("Arial", "B", 16)

    if prediction[0] == 1:

        pdf.set_text_color(255, 0, 0)

        pdf.cell(
            200,
            10,
            "FINAL RESULT: HEART DISEASE DETECTED",
            ln=True
        )

    else:

        pdf.set_text_color(0, 128, 0)

        pdf.cell(
            200,
            10,
            "FINAL RESULT: NO HEART DISEASE DETECTED",
            ln=True
        )

    pdf.set_text_color(0,0,0)

    # --------------------------------
    # SAVE PDF
    # --------------------------------
    pdf.output("report.pdf")

    # --------------------------------
    # DOWNLOAD BUTTON
    # --------------------------------
    with open("report.pdf", "rb") as file:

        st.download_button(
            label="📥 Download Medical Report",
            data=file,
            file_name="Heart_Report.pdf",
            mime="application/pdf"
        )

st.markdown('</div>', unsafe_allow_html=True)
