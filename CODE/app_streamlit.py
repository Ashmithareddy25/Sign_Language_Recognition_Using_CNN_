import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import numpy as np
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(page_title="VisionAI | Sign Language Recognition", page_icon="🤟", layout="wide")

# -----------------------------------------------------------
# STYLING (Modern Dark Theme with Strong Contrast)
# -----------------------------------------------------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    color: #EAEAEA;
}
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.shutterstock.com/blog/wp-content/uploads/sites/5/2021/08/ASL-cover-image.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ---------- NAVIGATION BAR ---------- */
.navbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background-color: rgba(0, 0, 0, 0.85);
    padding: 18px;
    border-radius: 8px;
    text-align: center;
}
.navbar a {
    color: #00C6FF;
    text-decoration: none;
    font-weight: 600;
    font-size: 20px;
    margin: 0 30px;
    transition: 0.3s;
}
.navbar a:hover {
    color: #FFFFFF;
    text-shadow: 0px 0px 10px #00C6FF;
}

/* ---------- SECTION LAYOUT ---------- */
.section {
    background-color: rgba(10, 10, 15, 0.93);
    border-radius: 15px;
    padding: 50px;
    width: 85%;
    margin: 40px auto;
}
.section h1, .section h2 {
    color: #00C6FF;
    font-weight: 700;
}
.section p, .section li, .section ol {
    color: #E8E8E8;
    font-size: 18px;
    line-height: 1.8em;
}
.section th, .section td {
    color: #FFFFFF;
    border: 1px solid #444;
    padding: 8px;
}
.section table {
    width: 100%;
    border-collapse: collapse;
}

/* ---------- TITLE SECTION ---------- */
.title-section {
    text-align: center;
    padding: 90px;
    background: rgba(0, 0, 0, 0.78);
    border-radius: 25px;
    margin: 30px auto;
    width: 85%;
}
.title-section h1 {
    color: #00C6FF;
    font-size: 70px;
    font-weight: 900;
}
.title-section h3 {
    font-weight: 300;
    color: #F8F8F8;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    padding: 40px;
    background-color: rgba(0,0,0,0.85);
    border-radius: 15px;
    margin-top: 70px;
    color: #E8E8E8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD CNN MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("Model/FinalModel.h5")

cnn_model = load_cnn_model()
classes = [chr(i) for i in range(65, 91)]

# -----------------------------------------------------------
# NAVIGATION BAR
# -----------------------------------------------------------
st.markdown("""
<div class="navbar">
    <a href="#home">🏠 Home</a>
    <a href="#services">🧩 Services</a>
    <a href="#image-upload">🖼️ Image Upload</a>
    <a href="#live-prediction">🎥 Live Prediction</a>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HOME
# -----------------------------------------------------------
st.markdown("""
<a name="home"></a>
<div class="title-section">
    <h1>🤟 SIGN LANGUAGE DETECTION</h1>
    <h3>AI-Powered Communication for the Hearing and Speech Impaired</h3>
</div>

<div class="section">
    <h2>🏠 Project Overview</h2>
    <p>
    The <b>Sign Language Detection</b> system is a deep learning–based AI solution 
    designed to recognize American Sign Language (ASL) gestures and translate them 
    into text and audio output. It bridges communication barriers through 
    <b>gesture recognition, CNN classification, and speech synthesis.</b>
    </p>
    <h3>🔍 What Happens Here</h3>
    <ul>
        <li>Visitors are welcomed with a banner titled <b>“SIGN LANGUAGE DETECTION.”</b></li>
        <li>Explains CNN-based recognition of ASL gestures (A–Z).</li>
        <li>Highlights key innovations:
            <ul>
                <li>✅ High-accuracy CNN trained on ASL alphabets</li>
                <li>🔊 Real-time gesture-to-speech via gTTS</li>
                <li>🌐 Streamlit deployment for universal accessibility</li>
            </ul>
        </li>
    </ul>
    <p><b>🧠 Purpose:</b> Provide clear insight into system functionality and technologies powering it.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# SERVICES
# -----------------------------------------------------------
st.markdown("""
<a name="services"></a>
<div class="section">
    <h2>🧩 Services – System Capabilities</h2>
    <ul>
        <li>🖐️ <b>Sign-to-Text Recognition:</b> Converts static hand gestures into English alphabets.</li>
        <li>🔊 <b>Sign-to-Speech:</b> Converts model output to speech using Google gTTS.</li>
        <li>⚙️ <b>Real-Time Prediction:</b> Tracks live gestures via CVZone’s Hand Detector.</li>
        <li>📈 <b>Model Scalability:</b> Extendable to detect digits and common words (e.g., “Hello”).</li>
    </ul>
    <h3>🌍 Real-World Applications</h3>
    <ul>
        <li>🏫 Special-needs Education and Training</li>
        <li>🏛 Accessible Government Services</li>
        <li>🏥 Healthcare Communication Assistance</li>
        <li>🏢 Inclusive Workplace AI Tools</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# IMAGE UPLOAD
# -----------------------------------------------------------
st.markdown("""
<a name="image-upload"></a>
<div class="section">
    <h2>🖼️ Image Upload – Static Prediction Mode</h2>
    <p>Upload an ASL gesture image to let the model identify and pronounce the corresponding sign.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("📤 Upload a Sign Language Image", type=["jpg", "jpeg", "png"])
if uploaded:
    os.makedirs("images", exist_ok=True)
    path = os.path.join("images", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.image(path, caption="Uploaded ASL Gesture", width=350)
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = cnn_model.predict(img_array)
    prediction = classes[np.argmax(preds)]
    st.success(f"✅ Predicted Sign: **{prediction}**")
    tts = gTTS(text=prediction, lang="en", slow=False)
    tts.save("example.mp3")
    st.audio("example.mp3", format="audio/mp3")
    st.info("🗣️ The detected sign has been pronounced audibly.")
else:
    st.image("https://superstarworksheets.com/wp-content/uploads/2024/08/FreeASLAlphabetChart-1.jpg",
             caption="📘 ASL Reference Chart for Upload Guidance", width=700)

# -----------------------------------------------------------
# LIVE PREDICTION
# -----------------------------------------------------------
st.markdown("""
<a name="live-prediction"></a>
<div class="section">
    <h2>🎥 Live Prediction – Real-Time Gesture Recognition</h2>
    <p>
    Use your webcam to detect gestures live. The system identifies the ASL letter and overlays it in real time.
    </p>
</div>
""", unsafe_allow_html=True)

st.image("https://www.researchgate.net/publication/329550804/figure/fig1/AS%3A708746011480065%401545989625257/Static-Hand-Gestures-for-American-Sign-Language-Letters-6.png",
         caption="📘 ASL Gesture Reference for Live Mode", width=750)

st.info("Press 'Q' in the webcam window to exit live detection.")
if st.button("▶ Start Live Prediction"):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    offset = 20
    imgSize = 300
    labels = [chr(i) for i in range(65, 91)]

    while True:
        success, img = cap.read()
        if not success:
            break
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, labels[index], (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 255), 4)
            except:
                pass
        cv2.imshow("Live Sign Detection", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Live prediction ended successfully!")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div class="footer">
    <h3>👩‍💻 Developed by <b>Ashmitha Reddy Thota</b></h3>
    <p>M.S. in Computer Science — Data Science Concentration<br>
    University of North Carolina at Charlotte (UNCC)</p>
</div>
""", unsafe_allow_html=True)
