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
background-image: url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.main-box{
background: rgba(0,0,0,0.78);
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

.report-title{
font-size:28px;
font-weight:bold;
color:#0f172a;
text-align:center;
margin-bottom:20px;
}

table{
width:100%;
border-collapse: collapse;
background:white;
}

th{
background:#0f766e;
color:white;
padding:12px;
border:1px solid #ccc;
}

td{
padding:10px;
border:1px solid #ccc;
background:#f8fafc;
}

.high{
background:#dc2626;
color:white;
font-weight:bold;
}

.normal{
background:#16a34a;
color:white;
font-weight:bold;
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
# PATIENT DETAILS
# --------------------------------
st.markdown("## 👤 Patient Information")

patient_name = st.text_input("Enter Patient Name")

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
# PREDICT BUTTON
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

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1] * 100

    st.balloons()

    st.markdown("---")

    # --------------------------------
    # RESULT SECTION
    # --------------------------------
    if prediction[0] == 1:

        result_text = "Heart Disease Found 😢"
        result_status = "HIGH RISK"

        st.markdown(
            '<div class="result-danger">💔 Heart Disease Found 😢</div>',
            unsafe_allow_html=True
        )

        components.html(
        """
        <div style="text-align:center;">
        <img src="https://media.tenor.com/6oSedVpeB-YAAAAC/broken-heart.gif" width="350">
        </div>
        """,
        height=350
        )

    else:

        result_text = "No Heart Disease Found 😍"
        result_status = "NORMAL"

        st.markdown(
            '<div class="result-success">🎉 Hurrayy! No Disease Found 😍</div>',
            unsafe_allow_html=True
        )

        components.html(
        """
        <div style="text-align:center;">
        <img src="https://media.tenor.com/0AVbKGY_MxMAAAAC/heart.gif" width="350">
        </div>
        """,
        height=350
        )

    # --------------------------------
    # RISK BAR
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
    # MEDICAL REPORT TEMPLATE
    # --------------------------------
    st.markdown("## 🧾 Medical Report")

    report_html = f"""
    <div style="background:white;padding:20px;border-radius:15px;">

    <div class="report-title">
    HEART HEALTH MEDICAL REPORT
    </div>

    <p><b>Patient Name:</b> {patient_name}</p>
    <p><b>Date:</b> {datetime.now().strftime("%d-%m-%Y")}</p>

    <table>

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
    <td>Sex</td>
    <td>{sex}</td>
    <td class="normal">NORMAL</td>
    </tr>

    <tr>
    <td>Chest Pain Type</td>
    <td>{chest_pain}</td>
    <td class="high">CHECK</td>
    </tr>

    <tr>
    <td>Blood Pressure</td>
    <td>{bp}</td>
    <td class="high">HIGH</td>
    </tr>

    <tr>
    <td>Cholesterol</td>
    <td>{cholesterol}</td>
    <td class="high">HIGH</td>
    </tr>

    <tr>
    <td>EKG Results</td>
    <td>{ekg}</td>
    <td class="normal">NORMAL</td>
    </tr>

    <tr>
    <td>Maximum Heart Rate</td>
    <td>{max_hr}</td>
    <td class="normal">NORMAL</td>
    </tr>

    <tr>
    <td>Exercise Angina</td>
    <td>{exercise_angina}</td>
    <td class="high">CHECK</td>
    </tr>

    <tr>
    <td>ST Depression</td>
    <td>{st_depression}</td>
    <td class="high">CHECK</td>
    </tr>

    <tr>
    <td>Thallium Test</td>
    <td>{thallium}</td>
    <td class="high">CHECK</td>
    </tr>

    <tr>
    <td><b>Final Result</b></td>
    <td><b>{result_text}</b></td>
    <td class="high">{result_status}</td>
    </tr>

    </table>

    </div>
    """

    st.markdown(report_html, unsafe_allow_html=True)

    # --------------------------------
    # PDF REPORT
    # --------------------------------
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", 'B', 18)

    pdf.cell(200, 10, txt="Heart Health Medical Report", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)

    pdf.cell(
        200,
        10,
        txt=f"Date: {datetime.now().strftime('%d-%m-%Y')}",
        ln=True
    )

    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Sex: {sex}", ln=True)
    pdf.cell(200, 10, txt=f"Chest Pain Type: {chest_pain}", ln=True)
    pdf.cell(200, 10, txt=f"Blood Pressure: {bp}", ln=True)
    pdf.cell(200, 10, txt=f"Cholesterol: {cholesterol}", ln=True)
    pdf.cell(200, 10, txt=f"EKG Results: {ekg}", ln=True)
    pdf.cell(200, 10, txt=f"Maximum Heart Rate: {max_hr}", ln=True)
    pdf.cell(200, 10, txt=f"Exercise Angina: {exercise_angina}", ln=True)
    pdf.cell(200, 10, txt=f"ST Depression: {st_depression}", ln=True)
    pdf.cell(200, 10, txt=f"Thallium Test: {thallium}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", 'B', 16)

    pdf.cell(
        200,
        10,
        txt=f"FINAL RESULT: {result_text}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Risk Score: {probability:.2f}%",
        ln=True
    )

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as file:

        st.download_button(
            label="📥 Download Medical Report",
            data=file,
            file_name="Heart_Report.pdf",
            mime="application/pdf"
        )

st.markdown('</div>', unsafe_allow_html=True)
