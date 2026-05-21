import streamlit as st
import numpy as np
import joblib
import streamlit.components.v1 as components

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("heart_model.pkl")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
background-image: url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.main-container{
background: rgba(0,0,0,0.75);
padding: 40px;
border-radius: 25px;
box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
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

.result-box-success{
background: linear-gradient(to right,#16a34a,#22c55e);
padding:40px;
border-radius:25px;
text-align:center;
font-size:40px;
font-weight:bold;
color:white;
animation: pulse 1s infinite;
}

.result-box-danger{
background: linear-gradient(to right,#dc2626,#ef4444);
padding:40px;
border-radius:25px;
text-align:center;
font-size:40px;
font-weight:bold;
color:white;
animation: pulse 1s infinite;
}

@keyframes pulse{
0%{transform:scale(1);}
50%{transform:scale(1.02);}
100%{transform:scale(1);}
}

.stButton>button{
width:100%;
height:60px;
font-size:24px;
font-weight:bold;
border-radius:15px;
border:none;
background:linear-gradient(to right,#ff416c,#ff4b2b);
color:white;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.03);
}

label{
color:white !important;
font-size:18px !important;
font-weight:bold !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Smart Health Care System</div>',
    unsafe_allow_html=True
)

# -----------------------------
# INPUTS
# -----------------------------
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

# -----------------------------
# BUTTON
# -----------------------------
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

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -----------------------------
    # RESULT PAGE
    # -----------------------------
    if prediction[0] == 0:

        st.balloons()

        st.markdown(
            """
            <div class="result-box-success">
            🎉 HURRAYYY!! 🎉<br><br>
            😊 No Heart Disease Found 😊
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            "https://cdn-icons-png.flaticon.com/512/4140/4140048.png",
            width=250
        )

        components.html(
        """
        <div style="text-align:center;">
        <img src="https://media.tenor.com/0AVbKGY_MxMAAAAC/heart.gif" width="350">
        </div>
        """,
        height=350
        )

    else:

        st.snow()

        st.markdown(
            """
            <div class="result-box-danger">
            😢 HEART DISEASE FOUND 😢<br><br>
            Please Consult A Doctor
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            "https://cdn-icons-png.flaticon.com/512/3209/3209265.png",
            width=250
        )

        components.html(
        """
        <div style="text-align:center;">
        <img src="https://media.tenor.com/6oSedVpeB-YAAAAC/broken-heart.gif" width="350">
        </div>
        """,
        height=350
        )

st.markdown('</div>', unsafe_allow_html=True)