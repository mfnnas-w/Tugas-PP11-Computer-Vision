import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# Buka kamera (0 = webcam utama)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Tidak bisa membaca frame dari kamera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Gambar kotak di wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
