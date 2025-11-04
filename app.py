import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import pytesseract

text_data = pytesseract.image_to_string(gray_roi)


CASCADE_1 = "/mnt/data/haarcascade_russian_plate_number.xml"
CASCADE_2 = "/mnt/data/haarcascade_licence_plate_rus_16stages.xml"

st.set_page_config(page_title="License Plate Blurrer with OCR", layout="centered")

st.title("License Plate Blurrer with OCR Verification")

st.markdown("""
Upload a car image. The app will detect possible license plates using Haar cascades and contour heuristics,
then use OCR (EasyOCR) to verify real plates before blurring.
""")

# Parameters
blur_type = st.radio("Blur type:", ["Pixelate", "Gaussian"], index=0)
pixel_size = st.slider("Pixelation block size", 3, 50, 15)
gaussian_ksize = st.slider("Gaussian kernel size", 3, 51, 21, step=2)
min_plate_area = st.slider("Min plate area", 500, 5000, 1000)
ocr_conf_thresh = st.slider("Min OCR confidence to accept text", 0.1, 1.0, 0.4)

cascade1 = cv2.CascadeClassifier(CASCADE_1)
cascade2 = cv2.CascadeClassifier(CASCADE_2)



uploaded = st.file_uploader("Upload car image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
orig = img.copy()

st.write("Original image:")
st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_column_width=True)

def blur_region(img, x, y, w, h, method="pixelate"):
    roi = img[y:y+h, x:x+w]
    if method == "pixelate":
        k = max(1, pixel_size)
        small = cv2.resize(roi, (max(1, w//k), max(1, h//k)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = pixelated
    else:
        k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize+1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        img[y:y+h, x:x+w] = blurred
    return img

def detect_with_haar(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,10)):
    rects = []
    for casc in (cascade1, cascade2):
        if casc.empty(): continue
        found = casc.detectMultiScale(image_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        for (x,y,w,h) in found:
            rects.append((x,y,w,h))
    return rects

def fallback_plate_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(gray, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < min_plate_area: continue
        aspect = w / float(h)
        if 2.0 < aspect < 8.5:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) >= 4:
                candidates.append((x,y,w,h))
    candidates = sorted(candidates, key=lambda r: r[2]*r[3], reverse=True)
    return candidates

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
haar_rects = detect_with_haar(gray)

detected = haar_rects if len(haar_rects)>0 else fallback_plate_detection(img)
if not detected:
    st.warning("No plate-like regions detected.")
    st.stop()

st.info(f"Detected {len(detected)} candidate regions. Running OCR...")

out = orig.copy()
ocr_results = []
for (x,y,w,h) in detected:
    roi = orig[y:y+h, x:x+w]
    if roi.size == 0: continue
    ocr = reader.readtext(roi)
    plate_texts = []
    for (bbox, text, conf) in ocr:
        if conf >= ocr_conf_thresh and any(ch.isalnum() for ch in text):
            plate_texts.append((text, conf))
    if plate_texts:
        out = blur_region(out, x, y, w, h, method=("pixelate" if blur_type=="Pixelate" else "gaussian"))
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        ocr_results.append((x,y,w,h,plate_texts))

if ocr_results:
    st.success(f"Blurred {len(ocr_results)} verified license plate(s):")
    for (x,y,w,h,texts) in ocr_results:
        for t,conf in texts:
            st.write(f"🔹 Text: **{t}** (conf={conf:.2f})")
else:
    st.warning("OCR found no valid plate text in detected regions. Nothing blurred.")

st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

is_success, buffer = cv2.imencode(".jpg", out)
bts = io.BytesIO(buffer.tobytes())
st.download_button("Download result", data=bts, file_name="plate_blurred.jpg", mime="image/jpeg")



