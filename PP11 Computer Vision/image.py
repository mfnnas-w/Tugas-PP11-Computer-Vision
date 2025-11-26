import cv2

# Load kedua cascade
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

# Load gambar
img = cv2.imread('./images/sample.jpg')  # tetap harus seperti ini
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.08, 5)

for (x, y, w, h) in faces:
    # Gambar kotak wajah
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Region of Interest (ROI) untuk area wajah
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Deteksi mata dalam area wajah
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.08, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow('Face and Eye Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
