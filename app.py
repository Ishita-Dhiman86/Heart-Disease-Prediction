import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit.components.v1 as components
import speech_recognition as sr

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
# CUSTOM CSS
# --------------------------------
page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1505751172876-fa1923c5c528");
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
box-shadow: 0px 0px 30px rgba(0,0,0,0.7);
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
    '<div class="subtitle">AI Powered Smart Healthcare System</div>',
    unsafe_allow_html=True
)

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Prediction", "BMI Calculator", "Voice Assistant"]
)

# --------------------------------
# PREDICTION PAGE
# --------------------------------
if page == "Prediction":

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 1, 100, 30)

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        sex = 0 if sex == "Female" else 1

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
            sex,
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

        st.balloons()

        st.markdown("---")

        # --------------------------------
        # RESULT
        # --------------------------------
        if prediction[0] == 1:

            st.markdown(
                '<div class="result-danger">💔 Heart Disease Found 😢</div>',
                unsafe_allow_html=True
            )

            st.image(
                "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
                width=250
            )

            st.error("Please consult a cardiologist immediately.")

            components.html(
            """
            <div style="text-align:center;">
            <img src="https://media.tenor.com/6oSedVpeB-YAAAAC/broken-heart.gif" width="350">
            </div>
            """,
            height=350
            )

        else:

            st.markdown(
                '<div class="result-success">🎉 Hurrayy! No Disease Found 😍</div>',
                unsafe_allow_html=True
            )

            st.image(
                "https://cdn-icons-png.flaticon.com/512/190/190411.png",
                width=250
            )

            st.success("Your heart looks healthy and strong ❤️")

            components.html(
            """
            <div style="text-align:center;">
            <img src="https://media.tenor.com/0AVbKGY_MxMAAAAC/heart.gif" width="350">
            </div>
            """,
            height=350
            )

        # --------------------------------
        # PIE CHART
        # --------------------------------
        st.markdown("## 📊 Health Analysis")

        labels = ['Healthy', 'Risk']

        if prediction[0] == 1:
            sizes = [30, 70]
        else:
            sizes = [85, 15]

        fig, ax = plt.subplots()

        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%'
        )

        st.pyplot(fig)

        # --------------------------------
        # DOCTOR RECOMMENDATION
        # --------------------------------
        st.markdown("## 🩺 Recommended Cardiologists")

        st.markdown("""
        - Dr. Sharma — Heart Specialist  
        - Dr. Verma — AIIMS Cardiologist  
        - Apollo Hospital Cardiac Unit  
        """)

        # --------------------------------
        # PDF REPORT
        # --------------------------------
        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=16)

        pdf.cell(
            200,
            10,
            txt="Heart Disease Prediction Report",
            ln=True
        )

        if prediction[0] == 1:
            result_text = "Heart Disease Found"
        else:
            result_text = "No Heart Disease Found"

        pdf.cell(
            200,
            10,
            txt=result_text,
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

# --------------------------------
# BMI CALCULATOR PAGE
# --------------------------------
elif page == "BMI Calculator":

    st.markdown("## ⚖️ BMI Calculator")

    weight = st.number_input(
        "Enter Weight (kg)",
        1.0,
        200.0
    )

    height = st.number_input(
        "Enter Height (m)",
        0.5,
        3.0
    )

    if height > 0:

        bmi = weight / (height ** 2)

        st.success(f"Your BMI is: {bmi:.2f}")

# --------------------------------
# VOICE ASSISTANT PAGE
# --------------------------------
elif page == "Voice Assistant":

    st.markdown("## 🎤 Voice Health Assistant")

    st.info("Click button and speak your health question")

    if st.button("🎙️ Start Listening"):

        recognizer = sr.Recognizer()

        try:

            with sr.Microphone() as source:

                st.write("Listening...")

                audio = recognizer.listen(source)

                text = recognizer.recognize_google(audio)

                st.success(f"You said: {text}")

                st.info(
                    "For proper medical advice please consult a doctor."
                )

        except Exception as e:

            st.error("Microphone not detected or speech not recognized.")

st.markdown('</div>', unsafe_allow_html=True)
