import streamlit as st
import cv2
import os
import numpy as np

# Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data_folder = "face_database"
os.makedirs(data_folder, exist_ok=True)

# Capture and save face to database
def add_face_to_database():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera ochilmadi!")
        return

    st.title("Yuzni bazaga kiritish")

    ret, frame = cap.read()
    if not ret:
        st.error("Rasm olishda xatolik!")
        cap.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(frame, channels="BGR", caption="Aniqlangan yuz bilan rasm", use_column_width=True)
    image_name = st.text_input("Rasmga ism bering:")

    if image_name and st.button("Saqlash"):
        image_path = os.path.join(data_folder, f"{image_name}.jpg")
        cv2.imwrite(image_path, frame)
        st.success(f"{image_name} bazaga qo'shildi!")
    elif not image_name:
        st.warning("Iltimos, ism kiriting!")

    cap.release()

# Recognize face from the database
def recognize_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera ochilmadi!")
        return

    st.title("Yuzni aniqlash")

    ret, frame = cap.read()
    if not ret:
        st.error("Rasm olishda xatolik!")
        cap.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(frame, channels="BGR", caption="Aniqlangan yuz", use_column_width=True)

    for stored_image in os.listdir(data_folder):
        stored_img_path = os.path.join(data_folder, stored_image)
        stored_img = cv2.imread(stored_img_path)

        stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)
        stored_faces = face_cascade.detectMultiScale(stored_gray, 1.1, 4)

        if len(stored_faces) > 0 and len(faces) > 0:
            stored_x, stored_y, stored_w, stored_h = stored_faces[0]
            current_x, current_y, current_w, current_h = faces[0]

            diff_x = abs(stored_x - current_x)
            diff_y = abs(stored_y - current_y)

            if diff_x < 50 and diff_y < 50:
                st.success(f"Bu rasm: {stored_image.split('.')[0]}")
                cap.release()
                return

    st.error("Mos keluvchi yuz topilmadi!")
    cap.release()

# Streamlit UI
task = st.radio("Tanlang:", ("Yuzni bazaga kiritish", "Yuzni aniqlash"))

if task == "Yuzni bazaga kiritish":
    add_face_to_database()
elif task == "Yuzni aniqlash":
    if st.button("Yuzni aniqlash"):
        recognize_face()
