import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time

st.set_page_config(page_title="Rotten or Not 🍎", layout="wide")
st.title("🍓 Fruit Freshness Detector")
st.markdown("### Detect if your fruit is **fresh** or **rotten** using YOLO!")


st.sidebar.header("⚙️ Options")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)


@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()
st.success("✅ Model loaded successfully!")


st.markdown("#### 🎥 Live Webcam Detection")
start_detection = st.button("Start Webcam Detection")

FRAME_WINDOW = st.image([])

if start_detection:
    cap = cv2.VideoCapture(0)
    st.info("Press **Stop** or close the app to end the detection.")
    stop_button = st.button("Stop")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(frame, conf=confidence, verbose=False)
        pred = results[0]

        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = pred.names[cls_id]
                text = f"{label} {conf:.2f}"

                color = (0, 255, 0) if "fresh" in label.lower() else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        if stop_button:
            break

        time.sleep(0.03)

    cap.release()
    st.warning("🛑 Detection stopped.")

