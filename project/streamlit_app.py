import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Rotten or Not 🍎", layout="wide")
st.title("🍓 Fruit Freshness Detector")
st.markdown("Detect whether a fruit is **fresh** or **rotten** using YOLO")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()
st.success("✅ Model loaded successfully!")

# =====================================================
# 📤 IMAGE UPLOAD DETECTION
# =====================================================
st.header("📤 Upload Fruit Image")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # resize large images
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    st.image(frame_rgb, caption="Uploaded Image", width="stretch")

    results = model.predict(frame_resized, conf=0.5, verbose=False)
    pred = results[0]

    if pred.boxes is not None and len(pred.boxes) > 0:
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = pred.names[cls_id]

            color = (0,255,0) if "fresh" in label.lower() else (0,0,255)

            cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame_resized,
                        f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        st.image(frame_resized,
                 caption="Detection Result",
                 width="stretch")

    else:
        st.warning("⚠️ No fruit detected.")


# =====================================================
# 🎥 WEBCAM DETECTION
# =====================================================
st.header("🎥 Live Webcam Detection")

start_detection = st.button("Start Webcam")
FRAME_WINDOW = st.image([], width="stretch")

if start_detection:
    cap = cv2.VideoCapture(0)
    stop_button = st.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error.")
            break

        frame = cv2.flip(frame, 1)

        results = model.predict(frame, conf=0.5, verbose=False)
        pred = results[0]

        if pred.boxes is not None and len(pred.boxes) > 0:
            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = pred.names[cls_id]

                color = (0,255,0) if "fresh" in label.lower() else (0,0,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, width="stretch")

        if stop_button:
            break

        time.sleep(0.03)

    cap.release()
    st.warning("🛑 Webcam stopped.")
