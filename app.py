import io
import cv2
import numpy as np
import streamlit as st

# Built-in OpenCV Haar cascade (detects general license plates)
CASCADE = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

st.set_page_config(page_title="Simple Plate Blurrer", layout="centered")
st.title("🚗 Car Plate Blurrer")

uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    output = img.copy()

    # Detect license plates
    plate_cascade = cv2.CascadeClassifier(CASCADE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    # Blur detected plate regions
    for (x, y, w, h) in plates:
        roi = output[y:y+h, x:x+w]
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 30)
        output[y:y+h, x:x+w] = roi_blur

    # Show blurred image
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
             caption="Blurred Image", use_column_width=True)

    # Download button
    success, buffer = cv2.imencode(".jpg", output)
    bts = io.BytesIO(buffer.tobytes())
    st.download_button("Download Blurred Image",
                       data=bts, file_name="blurred_car.jpg", mime="image/jpeg")

else:
    st.info("📤 Drag and drop or browse to upload a car image.")
