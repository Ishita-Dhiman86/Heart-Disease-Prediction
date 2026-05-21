import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit.components.v1 as components
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
background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f");
background-size: cover;
background-position: center;
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

.patient-box{
background: rgba(255,255,255,0.1);
padding: 15px;
border-radius: 15px;
margin-bottom: 25px;
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
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.03);
background:linear-gradient(to right,#ff4b2b,#ff416c);
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

.report-box{
background:white;
padding:30px;
border-radius:20px;
margin-top:30px;
color:black;
}

.report-title{
text-align:center;
font-size:38px;
font-weight:bold;
color:#003366;
margin-bottom:30px;
}

.report-table{
width:100%;
border-collapse: collapse;
}

.report-table th{
background:#0f766e;
color:white;
padding:12px;
font-size:18px;
}

.report-table td{
padding:12px;
border:1px solid #ddd;
font-size:16px;
}

.normal{
background:#16a34a;
color:white;
font-weight:bold;
}

.high{
background:#dc2626;
color:white;
font-weight:bold;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# --------------------------------
# MAIN UI
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
# PATIENT DETAILS
# --------------------------------
st.markdown('<div class="patient-box">', unsafe_allow_html=True)

patient_name = st.text_input("👤 Enter Patient Name")

st.markdown('</div>', unsafe_allow_html=True)

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

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)[0][1] * 100

    st.balloons()

    st.markdown("---")

    # --------------------------------
    # RESULT SECTION
    # --------------------------------
    if prediction[0] == 1:

        result_text = "Heart Disease Detected"
        result_status = "HIGH RISK"

        st.markdown(
            '<div class="result-danger">💔 Heart Disease Found 😢</div>',
            unsafe_allow_html=True
        )

        st.error("Please consult a cardiologist immediately.")

    else:

        result_text = "No Heart Disease Detected"
        result_status = "NORMAL"

        st.markdown(
            '<div class="result-success">🎉 No Disease Found 😍</div>',
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
    # MEDICAL REPORT DISPLAY
    # --------------------------------
    st.markdown('<div class="report-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="report-title">🩺 HEART HEALTH MEDICAL REPORT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h4>Patient Name: {patient_name}</h4>
        <h4>Date: {datetime.now().strftime("%d-%m-%Y")}</h4>
        """,
        unsafe_allow_html=True
    )

    # Status helper
    def status(value, low, high):
        if value < low or value > high:
            return "CHECK"
        return "NORMAL"

    report_html = f"""
    <table class="report-table">
        <tr>
            <th>Test</th>
            <th>Result</th>
            <th>Status</th>
        </tr>

        <tr>
            <td>Age</td>
            <td>{age}</td>
            <td class="normal">NORMAL</td>
        </tr>

        <tr>
            <td>Blood Pressure</td>
            <td>{bp}</td>
            <td class="{'high' if bp > 140 else 'normal'}">
                {"HIGH" if bp > 140 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>Cholesterol</td>
            <td>{cholesterol}</td>
            <td class="{'high' if cholesterol > 240 else 'normal'}">
                {"HIGH" if cholesterol > 240 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>Maximum Heart Rate</td>
            <td>{max_hr}</td>
            <td class="{'high' if max_hr < 100 else 'normal'}">
                {"CHECK" if max_hr < 100 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>Chest Pain Type</td>
            <td>{chest_pain}</td>
            <td class="normal">NORMAL</td>
        </tr>

        <tr>
            <td>Fasting Blood Sugar</td>
            <td>{fbs}</td>
            <td class="{'high' if fbs == 1 else 'normal'}">
                {"HIGH" if fbs == 1 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>EKG Results</td>
            <td>{ekg}</td>
            <td class="normal">NORMAL</td>
        </tr>

        <tr>
            <td>Exercise Induced Angina</td>
            <td>{exercise_angina}</td>
            <td class="{'high' if exercise_angina == 1 else 'normal'}">
                {"CHECK" if exercise_angina == 1 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>ST Depression</td>
            <td>{st_depression}</td>
            <td class="{'high' if st_depression > 2 else 'normal'}">
                {"CHECK" if st_depression > 2 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>Number of Vessels</td>
            <td>{vessels}</td>
            <td class="{'high' if vessels >= 2 else 'normal'}">
                {"CHECK" if vessels >= 2 else "NORMAL"}
            </td>
        </tr>

        <tr>
            <td>Final Prediction</td>
            <td>{result_text}</td>
            <td class="{'high' if prediction[0] == 1 else 'normal'}">
                {result_status}
            </td>
        </tr>

    </table>
    """

    st.markdown(report_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------
    # PDF REPORT
    # --------------------------------
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, "HEART HEALTH MEDICAL REPORT", ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", "", 12)

    # IMPORTANT: no emojis in PDF
    pdf.cell(200, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True)

    pdf.ln(5)

    report_lines = [
        f"Age: {age}",
        f"Sex: {sex}",
        f"Blood Pressure: {bp}",
        f"Cholesterol: {cholesterol}",
        f"Chest Pain Type: {chest_pain}",
        f"Fasting Blood Sugar: {fbs}",
        f"EKG Results: {ekg}",
        f"Maximum Heart Rate: {max_hr}",
        f"Exercise Angina: {exercise_angina}",
        f"ST Depression: {st_depression}",
        f"Slope: {slope}",
        f"Number of Vessels: {vessels}",
        f"Thallium Test: {thallium}",
        f"Risk Score: {probability:.2f}%",
        f"Final Result: {result_text}"
    ]

    for line in report_lines:
        pdf.cell(200, 10, txt=line, ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as file:

        st.download_button(
            label="📥 Download Medical Report",
            data=file,
            file_name="Heart_Report.pdf",
            mime="application/pdf"
        )

st.markdown('</div>', unsafe_allow_html=True)
