import cv2
import glob
import pandas as pd
import numpy as np
import face_recognition

ruta = r"C:\Users\Erven\Desktop\MCC\SEMESTRE 2 HECTORIN\EMOCIONES\\"
emotions = ["bored", "engaged", "excited", "focused", "interested"]
classifier_face = cv2.CascadeClassifier(r"C:\OPENCV\opencv\build\etc\lbpcascades\lbpcascade_frontalface.xml")

data = []

for index, emotion in enumerate(emotions):
    rutaEmocion = ruta + emotion
    for img_path in glob.glob(rutaEmocion + "\\*.jpeg"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces = classifier_face.detectMultiScale(img)
        if len(faces) > 0:
            face = faces[0]
            img_face = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            img_face_resized = cv2.resize(img_face, (150, 150))
            
            # Detectar puntos de características faciales
            face_landmarks = face_recognition.face_landmarks(img)
            
            # Obtener los datos de la imagen (aplanar la matriz)
            img_data = img_face_resized.flatten()
            
            # Agregar los datos al conjunto de datos
            data.append([img, face_landmarks, img_face_resized, img_data, emotion])

# Crear DataFrame de pandas
df = pd.DataFrame(data, columns=['Original Image', 'Face Landmarks', 'Face Image', 'Image Data', 'Emotion'])

# Guardar el DataFrame como un archivo CSV
df.to_csv('emotions_dataset_table2.csv', index=False)
