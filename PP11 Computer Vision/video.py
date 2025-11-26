# face_video.py
import cv2
import sys
import os

# Lokasi relative ke skrip ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(BASE_DIR, "cascades")

face_xml = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
eye_xml = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")

# Cek file cascade ada
if not os.path.exists(face_xml):
    print("ERROR: file haarcascade_frontalface_default.xml tidak ditemukan di:", face_xml)
    sys.exit(1)

if not os.path.exists(eye_xml):
    print("ERROR: file haarcascade_eye.xml tidak ditemukan di:", eye_xml)
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("ERROR: Tidak bisa membuka kamera.")
    sys.exit(1)

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()