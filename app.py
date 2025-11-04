import io
import cv2
import numpy as np
from PIL import Image
import pytesseract
import streamlit as st

# Use OpenCV’s built-in Haar cascades (no /mnt/data/ dependency)
CASCADE_1 = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
CASCADE_2 = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

st.set_page_config(page_title="License Plate Blurrer (Tesseract OCR)", layout="centered")

st.title("License Plate Blurrer with OCR Verification (Tesseract)")

st.markdown("""
Upload a car image.  
The app will detect possible license plates using Haar cascades and contour heuristics,
then verify them using **Tesseract OCR** before blurring.
""")

# Parameters
blur_type = st.radio("Blur type:", ["Pixelate", "Gaussian"], index=0)
pixel_size = st.slider("Pixelation block size", 3, 50, 15)
gaussian_ksize = st.slider("Gaussian kernel size", 3, 51, 21, step=2)
min_plate_area = st.slider("Minimum plate area (for contour fallback)", 500, 5000, 1000)
min_text_len = st.slider("Minimum text length to consider as plate", 3, 10, 4)

# Load cascades
cascade1 = cv2.CascadeClassifier(CASCADE_1)
cascade2 = cv2.CascadeClassifier(CASCADE_2)

if cascade1.empty() or cascade2.empty():
    st.warning("⚠️ Could not load Haar cascades. Using OpenCV defaults.")

# File uploader
uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

# Read uploaded file
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
orig = img.copy()

st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

# -------------------- Helper Functions --------------------

def blur_region(img, x, y, w, h, method="pixelate"):
    """Blur or pixelate a region"""
    roi = img[y:y+h, x:x+w]
    if method == "pixelate":
        k = max(1, pixel_size)
        small = cv2.resize(roi, (max(1, w//k), max(1, h//k)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = pixelated
    else:
        k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        img[y:y+h, x:x+w] = blurred
    return img

def detect_with_haar(gray):
    """Detect plates using Haar cascades"""
    rects = []
    for casc in (cascade1, cascade2):
        if casc.empty():
            continue
        found = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 10))
        for (x, y, w, h) in found:
            rects.append((x, y, w, h))
    return rects

def fallback_plate_detection(image):
    """Fallback: contour-based detection for plate-like regions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(gray, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_plate_area:
            continue
        aspect = w / float(h)
        if 2.0 < aspect < 8.5:
            candidates.append((x, y, w, h))
    return sorted(candidates, key=lambda r: r[2]*r[3], reverse=True)

# -------------------- Detection Phase --------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected = detect_with_haar(gray)

if not detected:
    st.info("No plates detected with Haar cascades. Trying contour fallback...")
    detected = fallback_plate_detection(img)

if not detected:
    st.error("No plate-like regions found.")
    st.stop()

st.info(f"Detected {len(detected)} possible plate region(s). Running OCR...")

# -------------------- OCR + Blurring --------------------

out = orig.copy()
blurred_count = 0

for (x, y, w, h) in detected:
    roi = orig[y:y+h, x:x+w]
    if roi.size == 0:
        continue

    # Convert ROI to grayscale for OCR
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.equalizeHist(gray_roi)

    # Run OCR with pytesseract
    text = pytesseract.image_to_string(gray_roi)
    clean_text = "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()

    # Check if this looks like a plate
    if len(clean_text) >= min_text_len and any(c.isdigit() for c in clean_text):
        out = blur_region(out, x, y, w, h, "pixelate" if blur_type == "Pixelate" else "gaussian")
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.write(f"✅ Plate text detected: **{clean_text}**")
        blurred_count += 1

# -------------------- Display Results --------------------

if blurred_count:
    st.success(f"Blurred {blurred_count} verified license plate(s).")
else:
    st.warning("OCR found no valid plate text; nothing blurred.")

st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Blurred Output", use_column_width=True)

# Allow download
is_success, buffer = cv2.imencode(".jpg", out)
bts = io.BytesIO(buffer.tobytes())
st.download_button("Download Result", data=bts, file_name="plate_blurred.jpg", mime="image/jpeg")
